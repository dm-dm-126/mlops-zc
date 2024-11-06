#!/usr/bin/env python
# coding: utf-8

import os
import pickle

import boto3
import click
import pandas as pd

from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def read_data(filename):
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': endpoint_url
            }
        }

        df = pd.read_parquet('s3://nyc-duration/yellow_tripdata_2023-03.parquet', storage_options=options)

    else:
        df = pd.read_parquet(filename)

    return df

def save_data(df, year, month):

    output_file = get_output_path(year=year, month=month)

    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if endpoint_url:
        options = {
            'client_kwargs': {
                'endpoint_url': endpoint_url
            }
        }

        df.to_parquet(output_file, engine="pyarrow", compression=None, index=False, storage_options=options)

    else:
        df.to_parquet(output_file, engine="pyarrow", compression=None, index=False)


def prepare_data(df, categorical):
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration/yellow_tripdata_{year:04d}-{month:02d}-processed.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


@click.command()
@click.option("--year", default=2023, prompt="Enter a year", help="Year of the data")
@click.option("--month", default=3, prompt="Enter a month (1-12)", help="Month of the data")
def main(year, month):

    print(f"Starting data processing for {year}-{month:02d}")

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    #df = read_data("https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_" + f"{year:04d}-{month:02d}.parquet")
    df = read_data(get_input_path(year, month))

    categorical = ["PULocationID", "DOLocationID"]

    df = prepare_data(df, categorical)

    print("Data loaded successfully")

    with open("models/model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print('Predicted mean duration:', y_pred.mean())

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    print("Preparing results for export")
    df_result = pd.DataFrame({"ride_id": df["ride_id"], "y_pred": y_pred})
    #output_file = f"taxi_type=yellow_year={year:04d}_month={month:02d}.parquet"
    #df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    save_data(df_result, year, month)

    print("Data processing and export completed successfully.")



if __name__ == "__main__":
    main()
