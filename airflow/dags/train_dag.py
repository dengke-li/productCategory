from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline import train_pipeline

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2023, 6, 29),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="mlflow_training_workflow",
    default_args=default_args,
    schedule_interval="@monthly",
)

train_model_task = PythonOperator(
    task_id="train_model_task", python_callable=train_pipeline, dag=dag
)

train_model_task
