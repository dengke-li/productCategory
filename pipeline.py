from typing import List

import datetime
from pathlib import Path

import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import DenseVector
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, udf, explode, col
from pyspark.sql.types import (
    ArrayType,
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)

from utils import Utils


class pipelineManager:
    def __init__(self):
        self.root_path = Path(__file__).parent
        self.model_repo = f"{self.root_path}/model_tracking"
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(f"file://{self.model_repo}")
        self.mlflow.start_run()
        self.spark = SparkSession.builder.appName(
            "MLflow workflow"
        ).getOrCreate()
        self.utils = Utils()


    def close(self):
        self.mlflow.end_run()

    def train(self, client_id):
        csv_file_path = f"{self.root_path}/data/train/client_id={client_id}/data_train.csv.gz"
        df = self.utils.load_data(self.spark, csv_file_path)

        # Encode string category into numeric in order to do backpropagation/error calculation in training
        target_column = "category_id"
        encode_target_column = "numeric_encoded_category"
        data_encoded, labels = self.utils.encoding(df, target_column, encode_target_column)
        train_data, test_data = data_encoded.randomSplit([0.8, 0.2], seed=42)

        # Create a VectorAssembler to assemble the feature columns into a feature vector
        feature_columns = ["f0", "f1", "f2", "f3", "f4"]
        assembler = VectorAssembler(
            inputCols=feature_columns, outputCol="features"
        )

        classifier = RandomForestClassifier(
            labelCol=encode_target_column, featuresCol="features", numTrees=10
        )
        pipeline = Pipeline(stages=[assembler, classifier])

        # Train the model
        model = pipeline.fit(train_data)

        # Evaluate the model
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(
            labelCol=encode_target_column,
            predictionCol="prediction",
            metricName="f1",
        )
        f1_score = evaluator.evaluate(predictions)
        evaluator = MulticlassClassificationEvaluator(
            labelCol=encode_target_column,
            predictionCol="prediction",
            metricName="accuracy",
        )
        accuracy = evaluator.evaluate(predictions)

        # Log metrics to MLflow
        self.mlflow.log_metrics(
            metrics={
                "f1": f1_score,
                "accuracy": accuracy,
            }
        )
        # self.mlflow.spark.log_model(model, "model")
        self.mlflow.set_tag("client", client_id)
        # self.mlflow.set_tag("labels", json.dumps(labels))

        # Save trained model
        model_path = f"model/client_id={client_id}/model"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        mlflow.spark.save_model(model, model_path)
        label_path = f"model/client_id={client_id}"
        self.utils.save_labels(labels, label_path)

    def predict(self, client_id, experiment_id=None, selected_run_id=None) -> DataFrame:
        """
        Load trained model to apply prediction for each product, and get top K predicted category and its probability,
        :param client_id:
        :param experiment_id:
        :param selected_run_id:
        :return: dataframe containing predict label and it probabilty and rank
        """
        client_path = f"{self.root_path}/model/client_id={client_id}"
        # model = self.mlflow.spark.load_model(f"runs:/{selected_run_id}/model")
        model = mlflow.spark.load_model(f"{client_path}/model")

        # label_path = f"{self.model_repo}/{experiment_id}/{selected_run_id}/tags/labels"
        label_path = f"{client_path}/label.json"
        labels = self.utils.load_labels(label_path)

        csv_file_path = f"{self.root_path}/data/prediction/client_id={client_id}/data_predict.csv.gz"
        df = self.utils.load_data(self.spark, csv_file_path)

        # Use model to predict for each product
        predictions = model.transform(df)

        # Broadcast labels to all worker node in order to be used by task more efficiently
        broadcast_labels = self.spark.sparkContext.broadcast(labels)

        def pick_top_k_udf(probabilities_vector: DenseVector, k: int):
            labels: List[str] = broadcast_labels.value
            probabilities: List[
                float
            ] = probabilities_vector.toArray().tolist()
            top_k = sorted(
                zip(labels, probabilities), key=lambda x: x[1], reverse=True
            )[:k]
            return [
                (rank, item[0], item[1]) for rank, item in enumerate(top_k)
            ]

        pick_top_k = udf(
            pick_top_k_udf,
            ArrayType(
                StructType(
                    [
                        StructField("pred_rank", IntegerType()),
                        StructField("pred_label", StringType()),
                        StructField("pred_score", DoubleType()),
                    ]
                )
            ),
        )
        # Get top 5 predicted category and its probability for each product
        d = predictions.withColumn(
            "top_5_values", pick_top_k(predictions["probability"], lit(5))
        )
        exploded_df = d.select(
            col("product_id"),
            explode(col("top_5_values")).alias("pred_categrory"),
        )
        # explode each row to 5 rows
        split_df = exploded_df.select(
            col("product_id"),
            col("pred_categrory.pred_rank"),
            col("pred_categrory.pred_label"),
            col("pred_categrory.pred_score"),
        )
        date = datetime.datetime.today().strftime("%Y-%m-%d")
        result_df = split_df.withColumn(
            "client_id", lit(client_id)
        ).withColumn("date", lit(date))
        return result_df

    def get_experiment_run_id(self, client_id):
        """
        Get latest model run's experiment_id and run_id for one client
        :param client_id: string
        :return: experiment_id, latest_run_id
        """
        runs = self.mlflow.search_runs(
            filter_string=f"tags.client = '{client_id}'",
            order_by=["start_time desc"],
        )
        run = runs.iloc[0]
        latest_run_id = run["run_id"]
        experiment_id = run["experiment_id"]
        print(
            f"run inference for {client_id} using model from experiment_id {experiment_id}, run_id {latest_run_id}"
        )
        return experiment_id, latest_run_id

    def inference_for_clients(self):
        """
        Inference for each client, then union all result, save to disk by partition
        :return:
        """
        prediction_folder = f"{self.root_path}/data/prediction"
        union_df = None
        for client_id in self.utils.get_clients(prediction_folder):
            print(f"run inference for {client_id}")
            # experiment_id, latest_run_id = self.get_experiment_run_id(client_id)
            # self.predict(client_id, experiment_id, latest_run_id)
            client_df = self.predict(client_id)
            if not union_df:
                union_df = client_df
            else:
                union_df = union_df.union(client_df)
            union_df.write.partitionBy("client_id", "date").csv(
                "results", mode="overwrite"
            )

    def train_for_clients(self):
        train_folder = f"{self.root_path}/data/train"
        for client_id in self.utils.get_clients(train_folder):
            print(f"run training for {client_id}")
            self.train(client_id)


def train_pipeline():
    pipeline = pipelineManager()
    pipeline.train_for_clients()
    pipeline.close()


def inference_pipeline():
    pipeline = pipelineManager()
    pipeline.inference_for_clients()
    pipeline.close()
