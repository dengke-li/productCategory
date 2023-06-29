import json
from pathlib import Path

from pyspark.ml.feature import StringIndexer, IndexToString


class Utils:
    def load_labels(self, path):
        with open(path) as fp:
            labels = json.load(fp)
        return labels

    def save_labels(self, labels, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f"{path}/label.json", "w") as fp:
            json.dump(labels, fp)

    def encoding(self, df, target_column, encode_target_column):
        indexer = StringIndexer(
            inputCol=target_column, outputCol=encode_target_column
        )
        indexer_model = indexer.fit(df)

        data_encoded = indexer_model.transform(df)
        return data_encoded, indexer_model.labels

    def decode(self, df_encoded, encode_target_column, target_column, labels):
        # Decode the numeric values back to the original string format
        index_to_string = IndexToString(inputCol=encode_target_column, outputCol=target_column, labels=labels)

        # Transform the encoded data to include the decoded column
        df_decoded = index_to_string.transform(df_encoded)
        return df_decoded

    def load_data(self, spark, csv_file_path):
        return (
            spark.read.format("csv")
            .option("header", True)
            .option("inferSchema", True)
            .option("compression", "gzip")
            .load(csv_file_path)
        )

    def get_clients(self, folder_path):
        for path in Path(folder_path).iterdir():
            text_path = path.as_posix()
            if "client_id=" in text_path:
                client_id = text_path.split("/")[-1].split("=")[-1]
                yield client_id