# ============================================
#              MODULE IMPORTS
# ============================================

import argparse
import sys
import csv
from pathlib import Path

import pandapower.networks as nw
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from modules.optimization_engine.rl_env import GridOptimizeEnv

# ==================================================
#              FUNCTIONAL SCRIPT
# ==================================================

# Offline training script for Project SWING's RL-based Optimization Engine.
# Run this manually whenever you want to (re)train a policy - it is NEVER imported or run by the live api.py process. 
# Training takes real wall-clock time (thousands of simulated episodes); the live API only ever LOADS an already-trained .zip via RLOptimizer.
# REMARK: If you want to train in cloud environment(Eg. Google Colab, etc.) Please use the interactive notebook located at:
#                                   notebooks/02_RL_Optimization_Cloud_Training.ipynb

class RewardCSVLogger(BaseCallback):
    """
    Writes one row per completed episode (timestep, reward, resulting max loading, mw_amount) to a CSV in models/. 
    Since GridOptimizeEnv is single-step, every step IS an episode 
    - this gives a genuine reward curve you can plot later without needing TensorBoard installed.
    """
    def __init__(self, csv_path: Path, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._file = None
        self._writer = None

    def _on_training_start(self):
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestep", "reward", "resulting_max_loading", "mw_amount", "converged"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        for info, reward in zip(infos, rewards):
            self._writer.writerow([
                self.num_timesteps,
                float(reward),
                info.get("resulting_max_loading"),
                info.get("mw_amount"),
                info.get("converged"),
            ])
        return True

    def _on_training_end(self):
        if self._file:
            self._file.close()


def main():
    parser = argparse.ArgumentParser(description="Train an RL policy for Project SWING's Optimization Engine.")
    parser.add_argument("--grid", choices=["39", "118"], required=True)
    parser.add_argument("--action", choices=["SHED_LOAD", "REDISPATCH_GEN"], required=True)
    parser.add_argument("--timesteps", type=int, default=50000,
                         help="Total training timesteps. 50k is a reasonable starting point; "
                              "watch the episode reward mean in the training log and increase "
                              "if it hasn't visibly plateaued.")
    args = parser.parse_args()

    print(f"Building IEEE-{args.grid} network for {args.action} training...")
    base_net = nw.case39() if args.grid == "39" else nw.case118()

    env = GridOptimizeEnv(base_net, action_type=args.action)
    print(f"Environment ready. Observation dim: {env.observation_space.shape[0]}, "
          f"valid target buses: {len(env.valid_target_buses)}")

    model = SAC("MlpPolicy", env, verbose=1)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    reward_log_path = models_dir / f"rl_optimizer_{args.action.lower()}_{args.grid}_rewards.csv"
    callback = RewardCSVLogger(reward_log_path)

    print(f"\nTraining for {args.timesteps} timesteps...")
    print(f"Reward log will be written to: {reward_log_path}\n")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    model_path = models_dir / f"rl_optimizer_{args.action.lower()}_{args.grid}.zip"
    model.save(str(model_path))

    print(f"\nSaved trained policy to: {model_path}")
    print(f"Saved reward log to: {reward_log_path}")
    print("Restart api.py to pick up the new policy (RLOptimizer loads it fresh on each /optimize_action call).")

# ==================================================
#              TESTING & EXECUTION
# ==================================================

# Usage:
#        python scripts/train_rl_agent.py --grid 118 --action SHED_LOAD
#        python scripts/train_rl_agent.py --grid 118 --action REDISPATCH_GEN
#        python scripts/train_rl_agent.py --grid 39  --action SHED_LOAD
#        python scripts/train_rl_agent.py --grid 39  --action REDISPATCH_GEN

if __name__ == "__main__":
    main()