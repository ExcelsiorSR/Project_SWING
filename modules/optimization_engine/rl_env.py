# ==================================================
#              MODULE IMPORTS
# ==================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import pandapower as pp

# ==================================================
#              FUNCTIONAL MODULE
# ==================================================

# Custom Gymnasium environment for training a continuous - MW corrective - action policy via SAC/PPO.

def build_observation_vector(net, target_bus: int) -> np.ndarray:
    """
    Shared observation-building logic, used by BOTH the training environment (on training_net) 
    and api.py's live inference call (on the real grid's net). 
    Keeping this as one function guarantees training and inference never silently drift apart in how they build the vector.
    """
    loadings = (net.res_line.loading_percent.fillna(0).values / 100.0).astype(np.float32)
    voltages = net.res_bus.vm_pu.fillna(0).values.astype(np.float32)
    n_buses = len(net.bus)
    target_norm = np.array([target_bus / max(n_buses - 1, 1)], dtype=np.float32)
    return np.concatenate([loadings, voltages, target_norm]).astype(np.float32)


class GridOptimizeEnv(gym.Env):
    """Single-step environment for training a continuous-MW corrective-action policy."""

    def __init__(self, base_net, action_type: str = "SHED_LOAD", search_ceiling_mw: float = 300.0,
                 load_multiplier_range=(1.0, 2.0), derate_range=(0.05, 0.5)):
        super().__init__()

        if action_type not in ("SHED_LOAD", "REDISPATCH_GEN"):
            raise ValueError(
                "GridOptimizeEnv only supports SHED_LOAD or REDISPATCH_GEN. "
                "LOAD_REDISTRIBUTION needs a discrete destination-bus choice and isn't "
                "a good fit for this continuous action space - keep using the bisection "
                "search in api.py for that action type."
            )

        self.base_net = base_net
        self.action_type = action_type
        self.search_ceiling_mw = search_ceiling_mw
        self.load_multiplier_range = load_multiplier_range
        self.derate_range = derate_range

        self.n_lines = len(base_net.line)
        self.n_buses = len(base_net.bus)
        self.is_case39 = (self.n_buses == 39)

        # Only buses that actually have the relevant facility are valid training targets 
        # - sampling a pure transmission junction bus for SHED_LOAD teaches the agent nothing.
        if action_type == "SHED_LOAD":
            self.valid_target_buses = sorted(base_net.load['bus'].unique().tolist())
        else:
            self.valid_target_buses = sorted(base_net.gen['bus'].unique().tolist())

        if not self.valid_target_buses:
            raise ValueError(f"No buses with a facility for {action_type} in this network.")

        ceiling = 100.0 if self.is_case39 else search_ceiling_mw
        self.search_ceiling_mw = ceiling
        self.action_space = spaces.Box(low=0.0, high=ceiling, shape=(1,), dtype=np.float32)

        obs_dim = self.n_lines + self.n_buses + 1
        self.observation_space = spaces.Box(low=-2.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        self.training_net = None
        self.target_bus = None

    def _get_observation(self) -> np.ndarray:
        return build_observation_vector(self.training_net, self.target_bus)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.training_net = copy.deepcopy(self.base_net)

        # DYNAMIC STRESS: Use narrower load multiplier for 39-bus
        mult_min, mult_max = (1.0, 1.4) if self.is_case39 else self.load_multiplier_range
        load_mult = self.np_random.uniform(mult_min, mult_max)

        # Random stress each episode so the policy generalizes across severities, 
        # not just whatever congestion level the live twin happens to be sitting at during training.
        derate_min, derate_max = (0.3, 0.7) if self.is_case39 else self.derate_range
        derate_mult = self.np_random.uniform(derate_min, derate_max) # DYNAMIC DERATING: Prevent mathematically unsolvable thermal bottlenecks on Case 39

        self.training_net.load['p_mw'] = self.training_net.load['p_mw'] * load_mult
        self.training_net.load['q_mvar'] = self.training_net.load['q_mvar'] * load_mult
        self.training_net.line['max_i_ka'] = self.training_net.line['max_i_ka'] * derate_mult

        self.target_bus = int(self.np_random.choice(self.valid_target_buses))

        try:
            pp.runpp(self.training_net, solver='nr', numba=True, check_connectivity=False)
        except pp.LoadflowNotConverged:
            # The random stress draw was severe enough that even the BASELINE (before any corrective action) doesn't converge 
            # - retry with a fresh draw rather than starting the agent from a broken state it has no way to recover from.
            return self.reset(seed=seed, options=options)

        return self._get_observation(), {}

    def step(self, action):
        mw_amount = float(np.clip(action[0], 0.0, self.search_ceiling_mw))

        if self.action_type == "SHED_LOAD":
            idx = self.training_net.load[self.training_net.load.bus == self.target_bus].index[0]
            current_mw = self.training_net.load.at[idx, "p_mw"]
            self.training_net.load.at[idx, "p_mw"] = max(0.0, current_mw - mw_amount)
        else:
            idx = self.training_net.gen[self.training_net.gen.bus == self.target_bus].index[0]
            self.training_net.gen.at[idx, "p_mw"] += mw_amount

        try:
            pp.runpp(self.training_net, solver='nr', numba=True, check_connectivity=False)
            max_loading = float(self.training_net.res_line.loading_percent.max())
            converged = True
        except pp.LoadflowNotConverged:
            # Treat non-convergence as maximally bad, not silently ignored or treated as a neutral/zero outcome.
            max_loading = 999.0
            converged = False

        '''
        Reward: minimize MW used (linear penalty), HEAVILY penalize remaining overload 
        (SQUARED - a 180% overload is much worse than 3.6x as bad as a 105% overload, and the reward should reflect that steepness), 
        bonus for a fully secure result.
        DYNAMIC REWARD SCALING: Lower coefficients for 39-bus
        '''
        lambda_1 = 0.1 if self.is_case39 else 1.0
        lambda_2 = 0.005 if self.is_case39 else 0.05
        
        overload_penalty = max(0.0, max_loading - 100.0) ** 2
        mw_penalty = mw_amount / self.search_ceiling_mw
        secure_bonus = 10.0 if (converged and max_loading <= 100.0) else 0.0

        reward = -(lambda_1 * mw_penalty) - (lambda_2 * overload_penalty) + secure_bonus

        '''
        Single-step: terminated is always True. Whether the agent "won" or "lost" lives in the reward's sign/magnitude, 
        not in episode length - keeps this from conflating "is the episode over" with "did the agent do well," 
        which can otherwise create subtle replay-buffer weirdness in SAC/PPO.
        '''
        terminated = True
        truncated = False

        observation = self._get_observation()
        info = {"resulting_max_loading": max_loading, "converged": converged, "mw_amount": mw_amount}

        return observation, reward, terminated, truncated, info