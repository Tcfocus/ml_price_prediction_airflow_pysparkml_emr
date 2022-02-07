# pyspark

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window
import pandas as pd
import ta

# function for calculating RSI from price
def getRsi(x):
    ta_rsi = ta.momentum.RSIIndicator(close=x, window=14)
    return ta_rsi.rsi()


def linear_regression_prediction(input_location, output_location):
    """
    Create a Linear Regression model for predicting price based off of current price, volume, market cap, and RSI values
    """
    df_input = spark.read.csv(input_location, header=True, inferSchema=True)

    # transform to pandas df and calculate RSI column then revert back to spark df
    df_input_ta = df_input.toPandas()
    df_input_ta['rsi'] = df_input_ta.price.transform(getRsi)
    df_input_ta = spark.createDataFrame(df_input_ta)

    # calculate 'next_price' column
    window = Window.orderBy("date")
    a = lead(col("price")).over(window)
    final_df = df_input_ta.withColumn("next_price", a).dropna(how="any")

    # Prepare data for Machine Learning by establishing features and label ("new_step") columns
    feature = VectorAssembler(inputCols=['price', 'volume', 'marketCap', 'rsi'], outputCol='features')
    final_ml_data = feature.transform(final_df)

    # select only 'features' and 'next_price' for the model
    final_ml_data = final_ml_data.select('features', 'next_price')

    # split data into test and train data
    splits = final_ml_data.randomSplit([0.7, 0.3])
    train_df, test_df = final_ml_data.randomSplit([0.7, 0.3])

    # Linear regression
    lr = LinearRegression(featuresCol='features', labelCol='next_price')
    model = lr.fit(train_df)

    # Run model on test set to make predictions
    predictions = model.transform(test_df)

    # Clean up final predictions data by extracting the values from the vector column
    split1_udf = udf(lambda value: value[0].item(), FloatType())
    split2_udf = udf(lambda value: value[1].item(), FloatType())
    split1_udf = udf(lambda value: value[2].item(), FloatType())
    split2_udf = udf(lambda value: value[3].item(), FloatType())

    predictions = predictions.withColumn('price', split1_udf('features')) \
        .withColumn('volume', split2_udf('features')) \
        .withColumn('marketCap', split2_udf('features')) \
        .withColumn('rsi', split2_udf('features'))

    predictions = predictions.select("price", "volume", "marketCap", "rsi", "next_price", "prediction")

    # Output data as a parquet = a columnar storage format
    predictions.write.mode("overwrite").parquet(output_location)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="HDFS input", default="/source")
    parser.add_argument("--output", type=str, help="HDFS output", default="/output")
    args = parser.parse_args()
    spark = SparkSession.builder.appName("Linear Regression Prediction").getOrCreate()
    linear_regression_prediction(input_location=args.input, output_location=args.output)


