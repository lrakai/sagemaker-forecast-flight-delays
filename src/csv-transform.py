import math
import os
import pandas as pd
import numpy as np
import matplotlib as plt


def csv_to_df(csv_path, skiprows):
    dtypes = {
        "YEAR":  np.int64,
        "QUARTER":  "category",
        "MONTH":  "category",
        "DAY_OF_MONTH":  "category",
        "DAY_OF_WEEK":  "category",
        "UNIQUE_CARRIER":  "category",
        "TAIL_NUM":  "category",
        "FL_NUM":  "category",
        "ORIGIN":  "category",
        "DEST":  "category",
        "CRS_DEP_TIME":  np.int64,
        "DEP_TIME":  np.int64,
        "DEP_DELAY":  np.float64,
        "DEP_DELAY_NEW":  np.float64,
        "DEP_DEL15":   np.int64,
        "DEP_DELAY_GROUP":  np.int64,
        "CRS_ARR_TIME":  np.int64,
        "ARR_DELAY":  np.float64,
        "CRS_ELAPSED_TIME":  np.float64,
        "DISTANCE":  np.float64,
        "DISTANCE_GROUP":  "category"
    }
    df = pd.read_csv(csv_path, dtype=dtypes, skiprows=skiprows)
    return df


def onehot_encode(df):
    return df


def write_csv(df, csv_path, transform_name):
    out_path = "{0}_{1}.csv".format(
        os.path.splitext(csv_path)[0], transform_name)
    df.to_csv(path_or_buf=out_path, index=False)
    return


def csv_transform(csv_path, drop_columns, data_config):
    df = csv_to_df(csv_path, data_config["skiprows"])
    df = df.drop(columns=drop_columns)
    df = onehot_encode(df)
    write_csv(df, csv_path, data_config["transform_name"])
    return


if __name__ == "__main__":
    drop_columns = ["YEAR", "TAIL_NUM", "FL_NUM"]
    numrows = 1048576 - 1  # minus 1 for header
    training_ratio = 0.7
    data_config = {
        "train": {
            "skiprows": range(math.floor(training_ratio * numrows) + 1, numrows + 2),
            "skipfooter": numrows - math.floor(training_ratio * numrows),
            "transform_name": "training"
        },
        "test": {
            "skiprows": range(1, math.floor(training_ratio * numrows) + 1),
            "transform_name": "test"
        }
    }
    csv_transform("data/Flights.csv", drop_columns, data_config["train"])
    csv_transform("data/Flights.csv", drop_columns, data_config["test"])
