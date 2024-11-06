import os
import pandas as pd

from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():

    # Preparing mock data
    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    df = pd.DataFrame(data, columns=columns)

    df.to_parquet(
        "s3://nyc-duration/yellow_tripdata_2023-01.parquet", # input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options={"client_kwargs": {"endpoint_url": os.getenv("S3_ENDPOINT_URL")}}
    )

    os.system(f"S3_ENDPOINT_URL={os.getenv("S3_ENDPOINT_URL")} python3 batch.py --year 2023 --month 1")
    df_check = pd.read_parquet('s3://nyc-duration/yellow_tripdata_2023-01-processed.parquet', storage_options={"client_kwargs": {"endpoint_url": os.getenv("S3_ENDPOINT_URL")}})

    # Verify the structure of the processed file
    assert "ride_id" in df_check.columns, "'ride_id' column is missing from the output."
    assert "y_pred" in df_check.columns, "'y_pred' column is missing from the output."

    print(f'Sum of predictions: {df_check["y_pred"].sum()}')
