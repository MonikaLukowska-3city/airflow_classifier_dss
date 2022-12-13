from datetime import datetime
from airflow import DAG
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
import json


with DAG(
    dag_id='init_classifier',
    description='classification customer credit risk',
    schedule_interval='@once', 
    start_date=datetime(2022, 3, 1),
    catchup=False) as dag:

    task_is_api_active = HttpSensor(
            task_id='is_api_active',
            http_conn_id='rest_api',
            endpoint='/healthcheck'
        )

    taks_initial_train = SimpleHttpOperator(
            task_id='initial_train',
            http_conn_id='rest_api',
            endpoint='/initial_train',
            method='GET',
            response_filter=lambda response: json.loads(response.text),
            log_response=True
        )

task_is_api_active >> taks_initial_train 
