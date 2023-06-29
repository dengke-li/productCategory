import unittest

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
)

from utils import Utils


class TestUtils(unittest.TestCase):

    def test_encoding(self):
        spark = SparkSession.builder.appName(
            "test"
        ).getOrCreate()
        utils = Utils()

        categories = [("cat10", 34), ("cat2", 22), ("cat24", 25), ("cat_8s", 24), ("y_abc", 45), ("cat24", 22)]
        field = [
            StructField("NAME", StringType(), True),
            StructField("AGE", IntegerType(), True),
        ]
        schema = StructType(field)
        df = spark.createDataFrame(categories, schema)
        encoded_df, labels = utils.encoding(df, 'NAME', 'encoded_name')
        df_decoded = utils.decode(encoded_df, 'encoded_name', 'name_decoded', labels)
        assert df_decoded.collect() == [
            Row(NAME='cat10', AGE=34, encoded_name=1.0, name_decoded='cat10'),
            Row(NAME='cat2', AGE=22, encoded_name=2.0, name_decoded='cat2'),
            Row(NAME='cat24', AGE=25, encoded_name=0.0, name_decoded='cat24'),
            Row(NAME='cat_8s', AGE=24, encoded_name=3.0, name_decoded='cat_8s'),
            Row(NAME='y_abc', AGE=45, encoded_name=4.0, name_decoded='y_abc'),
            Row(NAME='cat24', AGE=22, encoded_name=0.0, name_decoded='cat24')
        ]
        assert set(labels) == set(['cat10', 'cat2', 'cat24', 'cat_8s', 'y_abc'])