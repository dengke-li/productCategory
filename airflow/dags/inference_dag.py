from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from pipeline import inference_pipeline

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2023, 6, 29),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "mlflow_inference_workflow",
    default_args=default_args,
    schedule_interval="@daily",
)

inference_model_task = PythonOperator(
    task_id="inference_model_task", python_callable=inference_pipeline, dag=dag
)
inference_model_task
