# Data Pipeline for Linear Regression Model using Airflow, AWS EMR + S3, and PySpark ML

# Overview

This project demonstrates a Data Pipeline utilizing AWS components and Apache Airflow to supply data to a data lake(S3) for use by data analysts and data scientists. This displays a workflow for migrating to the Cloud (AWS in this example), using ETL methods to prepare the data and make it accessible to groups such as data science teams. A machine learning model with linear regression is applied to the data and then pushed to its final destination into a data lake.

The data of interest is historical price data of a crypto asset called Algorand, which is extracted seperately and stored locally in CSV format. This file contains historical daily data for price, market cap, and volume. The goal is to be able to output the results of a linear regression model, and test the addition of different technical analysis indicators. A common indicator called RSI is calculcated in this example and supplied to the model.

# Project Components
* Utilize Airflow to manage the data pipeline
* Create an EMR cluster for the ETL, which that terminates upon completition
* PySpark to perform the processing and transformation of data
* Spark ML to create a linear regression model for predicting future price
* Use Amazon S3 as the the primary data lake storage platform

# Architecture Design

![Alt_text](https://github.com/Tcfocus/ml_price_prediction_airflow_pysparkml_emr/blob/main/assets/images/architecture.jpg)

# Airflow ETL

![Alt text](https://github.com/Tcfocus/ml_price_prediction_airflow_pysparkml_emr/blob/main/assets/images/AirflowDesign.jpg)

### ETL Steps
   1. Access the source data and the main script for the linear regression model, and move it to S3.
   2. Create an EMR cluster that is supplied with a bootstrap action (.sh script) for installing the necessary modules.
   3. Add the EMR steps for running the linea regression script on the data.
   4. Terminate the cluster once the steps are complete.

# Creating the EMR Cluster and defining the Spark Steps

Often times, data and scripts are already existing in the Cloud in locations such as an S3 bucket, but this workflow will be adding those items into S3 itself and accessing them. The EMR cluster is created using Airflows EMRCreateJobFlowOperator, where the cluster configurations are defined. EMR version 5.34.0 is used along with M5.xlarge instance types are used for the master and the core, as well as a bootstrap action script.

Three spark steps are added to the cluster, which copy the S3 data into the clusters HDFS location, run the linear regression script, and copies the output from the HDFS location to the final S3 location.


# Data transformation and machine learning model

The algorand_price_lin_regression.py script contains the code that is applied to the input CSV data. The data is read from the CSV, and a RSI value is calculated using the TA module. A column is calculated containing the next day's price, which is used as the label for the model. 

```python
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
```

The data is then prepared for the machine learning model with a 'features' column containing the price, market cap, volume, and RSI values, along with the before mentioned 'next price' label.

```python

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

```

The predictions of the linear regression model are then cleaned up into a more usable format for further analysis. The 'features' column is split into it's respective columns, so the final output contains all of the relevants columns as well as the prediction results from the model.

```python

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
```





