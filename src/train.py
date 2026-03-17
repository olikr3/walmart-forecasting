"""
src/train.py
------------
Trains multiple models on the Walmart sales dataset, logging each
experiment run to MLflow (params, metrics, artifacts).

Usage:
    python3 -m src.train                        # runs all models
    python3 -m src.train --model lightgbm       # single model
    python3 -m src.train --no-register          # skip model registry

MLflow UI:
    mlflow ui --port 5000
"""

import argparse
import logging
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from src.features import build_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


DATA_DIR        = Path("data/raw")
MLFLOW_URI      = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "walmart-sales-forecast"
REGISTERED_NAME = "walmart-sales-model"
TARGET          = "Weekly_Sales"
N_CV_SPLITS     = 5

DROP_COLS = [
    "Date",           # dropped in build_features by default
    "Weekly_Sales",   # target
    # Raw MarkDown columns are summarised; drop originals to reduce noise
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
]


def get_models() -> dict:
    return {
        "ridge": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("model",   Ridge(alpha=1.0)),
            ]),
            {"alpha": 1.0, "imputer_strategy": "median"},
        ),
        "gbm": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model",   GradientBoostingRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )),
            ]),
            {
                "n_estimators": 300, "max_depth": 5,
                "learning_rate": 0.05, "subsample": 0.8,
            },
        ),
        "lightgbm": (
            lgb.LGBMRegressor(
                n_estimators=500,
                num_leaves=63,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_samples=20,
                random_state=42,
                verbose=-1,
            ),
            {
                "n_estimators": 500, "num_leaves": 63,
                "learning_rate": 0.05, "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
            },
        ),
    }


def prepare_data(drop_date: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Run the feature pipeline and return X, y."""
    log.info("Building features...")
    df = build_features(
        train_path   = DATA_DIR / "train.csv",
        store_path   = DATA_DIR / "stores.csv",
        feature_path = DATA_DIR / "features.csv",
        drop_date    = drop_date,
    )

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET]

    # Convert any remaining bool columns to int (LightGBM / sklearn safety)
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    log.info(f"Feature matrix: {X.shape}  |  Target: {y.shape}")
    return X, y



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # Weighted MAE
    wmae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "wmae": round(wmae, 6)}



def time_series_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_SPLITS,
) -> dict:
    """
    Walk-forward TimeSeriesSplit CV.
    Returns mean ± std for each metric across folds.
    """
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    results = {"rmse": [], "mae": [], "wmae": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        fold_metrics = compute_metrics(y_val.values, preds)
        for k, v in fold_metrics.items():
            results[k].append(v)

        log.info(f"  Fold {fold+1}/{n_splits} — RMSE: {fold_metrics['rmse']:,.0f}  "
                 f"MAE: {fold_metrics['mae']:,.0f}  WMAE: {fold_metrics['wmae']:.4f}")

    summary = {}
    for k, vals in results.items():
        summary[f"cv_mean_{k}"] = round(float(np.mean(vals)), 4)
        summary[f"cv_std_{k}"]  = round(float(np.std(vals)),  4)
    return summary


# single run

def train_model(
    name:     str,
    model,
    params:   dict,
    X:        pd.DataFrame,
    y:        pd.Series,
    register: bool = True,
) -> str:
    """
    Run one experiment, log everything to MLflow, optionally register.
    Returns the MLflow run_id.
    """
    log.info(f"\n{'='*60}\nTraining: {name}\n{'='*60}")

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "model_type": name,
            "dataset":    "walmart-sales-forecast",
            "n_features": X.shape[1],
            "n_samples":  X.shape[0],
        })

        mlflow.log_params(params)
        mlflow.log_param("cv_splits", N_CV_SPLITS)
        mlflow.log_param("feature_count", X.shape[1])

        log.info("Running time-series CV...")
        cv_metrics = time_series_cv(model, X, y, n_splits=N_CV_SPLITS)
        mlflow.log_metrics(cv_metrics)
        log.info(f"CV results: {cv_metrics}")

        log.info("Fitting on full dataset...")
        model.fit(X, y)
        full_preds   = model.predict(X)
        train_metrics = {f"train_{k}": v for k, v in compute_metrics(y.values, full_preds).items()}
        mlflow.log_metrics(train_metrics)

        if name == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        _log_feature_importance(name, model, X.columns.tolist())

        if register:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, REGISTERED_NAME)
            log.info(f"Registered model version: {mv.version}")
            mlflow.log_param("registered_version", mv.version)

        log.info(f"Run complete. ID: {run_id}")

    return run_id


def _log_feature_importance(name: str, model, feature_names: list) -> None:
    """Log a feature importance CSV for tree-based models."""
    try:
        if name == "lightgbm":
            importances = model.feature_importances_
        elif name == "gbm":
            importances = model.named_steps["model"].feature_importances_
        else:
            return  # Ridge doesn't have feature_importances_

        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
              .sort_values("importance", ascending=False)
        )
        fi_path = Path("/tmp/feature_importance.csv")
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path), artifact_path="diagnostics")
        log.info(f"Top 5 features:\n{fi_df.head().to_string(index=False)}")

    except Exception as e:
        log.warning(f"Could not log feature importance: {e}")




def parse_args():
    parser = argparse.ArgumentParser(description="Train Walmart sales forecast models")
    parser.add_argument(
        "--model",
        choices=["ridge", "gbm", "lightgbm", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip MLflow model registry step",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to directory containing raw CSVs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global DATA_DIR
    DATA_DIR = args.data_dir

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info(f"MLflow experiment: '{EXPERIMENT_NAME}' at {MLFLOW_URI}")

    # Load data once — shared across all model runs
    X, y = prepare_data()

    models   = get_models()
    to_train = models if args.model == "all" else {args.model: models[args.model]}

    run_ids = {}
    for name, (estimator, params) in to_train.items():
        run_id = train_model(
            name     = name,
            model    = estimator,
            params   = params,
            X        = X,
            y        = y,
            register = not args.no_register,
        )
        run_ids[name] = run_id

    log.info(f"\nAll runs complete: {run_ids}")
    log.info(f"View results: mlflow ui --port 5000")


if __name__ == "__main__":
    main()