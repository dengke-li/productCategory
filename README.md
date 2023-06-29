## Description
This project is to do classification on product based on product features

## Install library
```
(virtual_env)$ cd productCategory
(virtual_env)$ pip install -r requirements.txt
```


## How to use
Assure you are in productCategory
```

(virtual_env)$ export AIRFLOW_HOME=$PWD/airflow
(virtual_env)$ export PYTHONPATH=$PWD

(virtual_env)$ airflow standalone

```
Check in localhost:8080, use username and password shown by previous command `airflow standalone`

After some while, check in web ui, find DAG mlflow_training_workflow and inference_model_task in DAGs dashboard, if don't appear, 
you can do following:
```
open another terminal, run following command
(virtual_env)$ cd productCategory
(virtual_env)$ python airflow/dags/train_dag.py
(virtual_env)$ python airflow/dags/inference_dag.py
```
then normally you can find them in all DAGs dashboard, you can enable them,
by default mlflow_training_workflow DAG is scheduled monthly, and inference_model_task DAG is scheduled daily. 
But you can also trigger it manually by click triangle button, run mlflow_training_workflow firstly, it will run
training for all clients in data/train folder, once success, you can run inference_model_task, it will run inference
for all clients in data/prediction, and generate result in results folder.

Reference: https://airflow.apache.org/docs/apache-airflow/stable/start.html

## Unitest
```
(virtual_env)$ pip install pytest
(virtual_env)$ pytest test
```


## Techno choice
1. Use airflow to scheduler training and inference dag
2. Use mlflow to do log model metrics, tag, labels, even trained model to tracking server to allow muliple experiments 
   and model versioning, here I use local `productCategory/model_tracking` as tracking server storage, unfortunately, 
   after multiple test, I find mlflow.log_model to save model to tracking server is not stable, sometimes some run is 
   not generated or generated after job finish. don't know why, need to investigate. So currently in order to assure 
   model and category labels are well saved, i choose traditional way to save them in another folder(`model`)
   instead of using `mlflow.spark.log_model(model, "model")`
3. use spark to pipeline data loading, encoding, transforming, dataset split, training, evaluation, model saving

## Improvement
1. I choose spark mllib to do training instead of other framework like pytorch/tensorflow, reason is those deeplearn framework output
   model need to be transformed to format pmml in order to be used by inference(spark mlib), seems currently don't have mature reliable solution.

2. Can use dvc to versionize input dataset, and then use mlflow.set_tag to associate the dvc commit hash with this experiment/run
    then we can have a better control on history differentversions for dataset, model(versionized by mlflow), and code(versionized by git),
    and their associations.
3. Currently training, inference are done sequentially, we can parallelize them by launching parallel airflow jobs, and also put spark job in cluster mode using multiple worker resource,
    to assure low-latency and high-throughput
4. for multi-tenancy training, mlflow tracking server should be put in a remote server, allowing multiple datascientist 
   launch their experiments in same time.
5. for inference in production, need to assure new model deployment as blue-green deployment 
   or canary deployment(depends on strategy), can also do A/B testing to compare different model performance in real case
6. for inference in production, we can also monitor some performance metrics, input data quality/drift, resource utilisation, system health by using tool like prometheus
   for future improvement, establish alert mecanism for detecting anomalies on grafana etc
7. Need to integrate the whole pipeline into CICD to automate model test, deployment

