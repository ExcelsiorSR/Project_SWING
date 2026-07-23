# =============================================
#              MODULE IMPORTS
# =============================================

from pathlib import Path
import pandas as pd
import torch
import pickle
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# =================================================================
#                           ARCHITECTURE
# =================================================================

class TFTForecaster:
    """
    CPU-only inference wrapper around a TFT checkpoint TRAINED ON COLAB.
    This class never trains - it only loads a .ckpt file and runs .predict() on a short recent-history window, 
    which is cheap enough to run on a laptop/desktop CPU. 
    Training remains Colab's job; see the 01_TFT_Multi_Horizon_Cloud_Training.ipynb notebook for that half.
    """

    def __init__(self, checkpoint_path: str = None):
        current_dir = Path(__file__).resolve().parent
        self.project_root = current_dir.parent.parent
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path
            else self.project_root / "models" / "tft_best_model.ckpt"
        )
        self.model = None
        self.training_dataset_params = None

    def load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No TFT checkpoint found at {self.checkpoint_path}. "
                f"Train on Colab first, download best_model.ckpt, and place it there."
            )
        # map_location="cpu" is what makes this safe to run without a GPU
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            str(self.checkpoint_path), map_location="cpu"
        )
        self.model.eval()
        print(f"TFT model loaded for CPU inference from {self.checkpoint_path}")

    def load_training_dataset_config(self, pkl_path: str = None):
        """
        Loads the pickled TimeSeriesDataSet config saved alongside the checkpoint 
        This is what TimeSeriesDataSet.from_dataset() needs to reconstruct the exact encoder/decoder structure the model was trained with.
        """

        path = Path(pkl_path) if pkl_path else self.project_root / "models" / "training_dataset.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No training_dataset.pkl found at {path}. "
                f"Export it from Colab (pickle.dump(training_dataset, ...)) and place it there."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def build_recent_history_window(self, window: int = 168, csv_path: str = None) -> pd.DataFrame:
        """
        PRAGMATIC data source, not a true live buffer: loads the tail of the same processed historical CSV used for training
        (data/processed_data/tft_ready_data.csv) as the encoder window for a live forecast. 
        This is honest about its limitation - the "recent history" is recent HISTORICAL data, not truly live telemetry, 
        since nothing in this project currently logs a continuous live feature stream. 
        It's good enough to exercise the model end-to-end and get real multi-horizon predictions; swap this for a genuine live buffer
        (e.g. appending each hour's real telemetry as it happens) once that exists.

        window: encoder length the model expects (168 = max_encoder_length used in training). 
        A few extra rows are included since TimeSeriesDataSet.from_dataset(..., predict=True) also wants
        prediction_length rows of "future" known covariates available.
        """
        path = Path(csv_path) if csv_path else self.project_root / "data" / "processed_data" / "tft_ready_data.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No historical data found at {path}. Run grid_forecaster.py's "
                f"__main__ block first to generate tft_ready_data.csv."
            )

        df = pd.read_csv(path)

        for col in ["month", "dayofweek", "hour", "region"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # +24 gives TimeSeriesDataSet room for prediction_length lookahead rows it expects to see when predict=True.
        return df.tail(window + 24).reset_index(drop=True)

    def forecast_multi_horizon(self, recent_history_df: pd.DataFrame, training_dataset: TimeSeriesDataSet):
        """
        recent_history_df: last max_encoder_length (168) hourly rows, same columns/dtypes as used in training (categoricals as strings).
        training_dataset: the TimeSeriesDataSet object saved alongside the checkpoint, needed to reconstruct the exact encoder/decoder config.
        Returns quantile predictions for the next 24 hours (15-min/1h/6h/24h horizons are just different slices of this same output).
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before forecasting.")

        inference_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, recent_history_df, predict=True, stop_randomization=True
        )
        inference_dataloader = inference_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

        with torch.no_grad():
            raw_predictions = self.model.predict(inference_dataloader, mode="quantiles")

        # raw_predictions shape: [batch, prediction_length, n_quantiles]
        # Default QuantileLoss([0.1, 0.5, 0.9]) -> index 0/1/2 = p10/p50/p90
        return {
            "p10": raw_predictions[0, :, 0].tolist(),
            "p50": raw_predictions[0, :, 1].tolist(),
            "p90": raw_predictions[0, :, 2].tolist(),
        }

# ==================================================
#              TESTING & EXECUTION
# ==================================================


if __name__ == "__main__":
    
    forecaster = TFTForecaster()
    print(f"Checking for TFT checkpoint at: {forecaster.checkpoint_path}")
    print(f"  (that path exists: {forecaster.checkpoint_path.exists()})")

    try:
        forecaster.load_model()
        print("SUCCESS: checkpoint loaded and ready for CPU inference.")
    except FileNotFoundError as e:
        print(f"\nNOT FOUND: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"\nERROR while loading: {type(e).__name__}: {e}")
        raise SystemExit(1)

    print("\nTesting the full pipeline (history window + training config + forecast)...")
    try:
        training_dataset = forecaster.load_training_dataset_config()
        print(f"  training_dataset.pkl loaded OK")
        window_df = forecaster.build_recent_history_window()
        print(f"  history window built OK -- {len(window_df)} rows")
        result = forecaster.forecast_multi_horizon(window_df, training_dataset)
        print(f"  forecast OK -- p50 (normalized) next few hours: {result['p50'][:5]}")
        print("\nFULL PIPELINE SUCCESS.")
    except Exception as e:
        print(f"\nFULL PIPELINE FAILED at: {type(e).__name__}: {e}")
        print("(Checkpoint loading works fine -- this is a separate issue in the")
        print(" data/training-config path, not the model itself. See the error above.)")