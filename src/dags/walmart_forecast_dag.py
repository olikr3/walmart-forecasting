
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from pathlib import Path
import mlflow

from src.features import build_features
from src.train import prepare_data, get_models, train_model, MLFLOW_URI, EXPERIMENT_NAME

DATA_DIR = Path("data/raw")

default_args = {"owner": "ml-team", "retries": 1}

with DAG(
    dag_id="walmart_sales_forecast",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "walmart"],
) as dag:

    def feature_task(**context):
        X, y = prepare_data()
        # for downstream task logging
        context["ti"].xcom_push(key="n_samples",  value=X.shape[0])
        context["ti"].xcom_push(key="n_features", value=X.shape[1])

    def train_task(model_name: str, **context):
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        X, y = prepare_data()
        models = get_models()
        estimator, params = models[model_name]

        run_id = train_model(
            name=model_name, model=estimator,
            params=params, X=X, y=y, register=False,
        )
        context["ti"].xcom_push(key=f"run_id_{model_name}", value=run_id)

    def register_task(**context):
        ti = context["ti"]
        # pick run with best CV WMAE across all three models
        client = mlflow.tracking.MlflowClient(MLFLOW_URI)
        run_ids = [
            ti.xcom_pull(key=f"run_id_{m}", task_ids=f"train_{m}")
            for m in ["ridge", "gbm", "lightgbm"]
        ]
        best_run = min(
            run_ids,
            key=lambda rid: client.get_run(rid).data.metrics.get("cv_mean_wmae", 999)
        )
        mlflow.register_model(f"runs:/{best_run}/model", "walmart-sales-model")

    build = PythonOperator(
        task_id="build_features",
        python_callable=feature_task,
    )

    train_ridge = PythonOperator(
        task_id="train_ridge",
        python_callable=train_task,
        op_kwargs={"model_name": "ridge"},
    )
    train_gbm = PythonOperator(
        task_id="train_gbm",
        python_callable=train_task,
        op_kwargs={"model_name": "gbm"},
    )
    train_lgbm = PythonOperator(
        task_id="train_lightgbm",
        python_callable=train_task,
        op_kwargs={"model_name": "lightgbm"},
    )

    register = PythonOperator(
        task_id="register_best_model",
        python_callable=register_task,
    )

    # Dependency chain
    build >> [train_ridge, train_gbm, train_lgbm] >> register