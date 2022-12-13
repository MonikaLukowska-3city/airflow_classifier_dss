from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

import requests
import json


dag = DAG(
    dag_id='classifier',
    description='classification customer credit risk',
    schedule_interval='0 0 * * *',
    start_date=datetime(2022, 3, 1),
    catchup=False)

C_APP_ENDPOINT = "http://app:5000"
C_MAX_MODEL_FAIL_COUNT = 2


#===============================================================================================================================
server_is_active_task = HttpSensor(task_id='server_is_active', http_conn_id='rest_api', endpoint='/healthcheck', dag=dag)


#===============================================================================================================================
end_task = DummyOperator(task_id='end', trigger_rule=TriggerRule.ONE_SUCCESS,  dag=dag)


#===============================================================================================================================
def __champion_predict_newdata_task(**context):
    response = requests.get(f"{C_APP_ENDPOINT}/last_processed_part")
    result = json.loads(response.text)
    print(f"Make champion prediction for part_id: {result['next_part']}")

    response = requests.get(f"{C_APP_ENDPOINT}/predict", params={"part_id": result['next_part']})
    predict_result = json.loads(response.text)
    print(f"Champion predict result: {predict_result}")

    if predict_result["challenger_id"]:
        task_instance = context['task_instance']
        task_instance.xcom_push(key="challenger_id", value = predict_result["challenger_id"])
        task_instance.xcom_push(key="part_id", value = result['next_part'])
        return "challenger_predict_newdata"
    else:
        return "test_if_retrain"

champion_predict_newdata_task = BranchPythonOperator(task_id='champion_predict_newdata', python_callable=__champion_predict_newdata_task, provide_context=True, dag=dag)


#===============================================================================================================================
def __challenger_predict_newdata_task(ti):
    challenger_id = ti.xcom_pull(task_ids='champion_predict_newdata', key='challenger_id')
    part_id = ti.xcom_pull(task_ids='champion_predict_newdata', key='part_id')
    print(f"Make prediction for challenger_id: {challenger_id} and part_id: {part_id}")

    response = requests.get(f"{C_APP_ENDPOINT}/predict", params={"model_id": challenger_id, "part_id": part_id})
    predict_result = json.loads(response.text)
    print(f"Challenger predict result: {predict_result}")

 
challenger_predict_newdata_task = PythonOperator(task_id='challenger_predict_newdata', python_callable=__challenger_predict_newdata_task, provide_context=True, dag=dag)


#===============================================================================================================================
def __evaluate_models_task():
    response = requests.get(f"{C_APP_ENDPOINT}/evaluate_models")
    result = json.loads(response.text)
    print(f"Evaluate models result: {result}")

    if result["challenger_win"]:
        return "end"
    else:
        if result["champion_fail_count"] > C_MAX_MODEL_FAIL_COUNT:
            return "fail_count_exceeded"
        else:
            return "end" 

evaluate_models_task = BranchPythonOperator(task_id='evaluate_models', python_callable=__evaluate_models_task, provide_context=True, dag=dag)


#===============================================================================================================================
def __test_if_retrain_task():
    response = requests.get(f"{C_APP_ENDPOINT}/rate_champion")
    result = json.loads(response.text)

    print(f"Champion model rate: {result}")

    if  result['fail_count'] > C_MAX_MODEL_FAIL_COUNT:
        return "fail_count_exceeded"

    if result["need_to_train"]:
        return "examine_data"
    else:
        return "end"

test_if_retrain_task = BranchPythonOperator(task_id='test_if_retrain', python_callable=__test_if_retrain_task, provide_context=True, dag=dag)


#===============================================================================================================================
def __fail_count_exceeded_task():
    print("Fatal model error => business strategy need to be updated!!!")

fail_count_exceeded_task = PythonOperator(task_id='fail_count_exceeded', python_callable=__fail_count_exceeded_task, trigger_rule=TriggerRule.ONE_SUCCESS, provide_context=True, dag=dag)


#===============================================================================================================================
def __examine_data_task():
    response = requests.get(f"{C_APP_ENDPOINT}/valid_data")
    result = json.loads(response.text)
    print(f"Examine data result: {result}")

examine_data_task = PythonOperator(task_id='examine_data', python_callable=__examine_data_task, provide_context=True, dag=dag)


#===============================================================================================================================
def __train_task():
    response = requests.get(f"{C_APP_ENDPOINT}/train")
    result = json.loads(response.text)
    print(f"Training result: {result}")

train_task = PythonOperator(task_id='train', python_callable=__train_task, provide_context=True, dag=dag)



#===============================================================================================================================
# Flow Graph
#===============================================================================================================================
examine_data_task >> train_task
test_if_retrain_task >> [examine_data_task, end_task, fail_count_exceeded_task]
challenger_predict_newdata_task >> evaluate_models_task >> [end_task, fail_count_exceeded_task]
server_is_active_task >> champion_predict_newdata_task >> [challenger_predict_newdata_task, test_if_retrain_task]
