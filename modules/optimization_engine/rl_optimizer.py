# ==================================================
#              MODULE IMPORTS
# ==================================================

from pathlib import Path
import numpy as np
from stable_baselines3 import SAC

# ==================================================
#              FUNCTIONAL MODULE
# ==================================================

# Thin inference wrapper around a trained SAC policy - mirrors the GridForecaster/TFTForecaster pattern already used elsewhere in this 
# project: load a saved model once, expose one clean predict method. 
# This class NEVER trains; that's scripts/train_rl_agent.py's job, run offline.


class RLOptimizer:
    def __init__(self, grid_type: str, models_dir: str = None):
        if grid_type not in ("39", "118"):
            raise ValueError("grid_type must be '39' or '118' -- RL policies are grid-size-specific "
                              "and cannot be shared across different IEEE cases.")
        self.grid_type = grid_type
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).resolve().parent.parent.parent / "models"
        self.model = None
        self.action_type = None

    def _model_path(self, action_type: str) -> Path:
        return self.models_dir / f"rl_optimizer_{action_type.lower()}_{self.grid_type}.zip"

    def load_model(self, action_type: str):
        if action_type not in ("SHED_LOAD", "REDISPATCH_GEN"):
            raise ValueError(f"No RL policy exists for action_type '{action_type}' - "
                              f"only SHED_LOAD and REDISPATCH_GEN are RL-optimized. "
                              f"LOAD_REDISTRIBUTION always uses the bisection search.")

        path = self._model_path(action_type)
        if not path.exists():
            raise FileNotFoundError(f"No trained RL policy at {path}. Run scripts/train_rl_agent.py first.")

        self.model = SAC.load(str(path))
        self.action_type = action_type
        print(f"RL optimizer loaded: {action_type} / IEEE-{self.grid_type} from {path}")

    def predict_mw(self, observation: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("Call load_model(action_type) before predict_mw().")
        action, _ = self.model.predict(observation, deterministic=True)
        '''
        Clip defensively - a policy trained on one stress distribution can still occasionally extrapolate 
        outside its training range on a live grid state it's never seen. 
        The caller (api.py) re-checks this value against real physics via _test_action() regardless,
        but there's no reason to hand back a nonsensical negative value.
        '''
        return float(max(0.0, action[0]))