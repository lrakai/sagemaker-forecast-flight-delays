import math
import os
import pandas as pd
import numpy as np
import matplotlib as plt


def csv_to_df(csv_path, dtypes, skiprows):
    df = pd.read_csv(csv_path, dtype=dtypes, skiprows=skiprows)
    return df


def onehot_encode(df, dtypes):
    categoricals = [column for (column, dtype) in dtypes.items() if dtype == "category"]
    for column in categoricals:
        df = pd.get_dummies(data=df, columns=[column])
    return df


def write_csv(df, csv_path, transform_name):
    out_path = "{0}_{1}.csv".format(
        os.path.splitext(csv_path)[0], transform_name)
    columns = df.columns.tolist()
    if "ARR_DELAY" in columns:
        # place target in first column for SageMaker
        columns.remove("ARR_DELAY")
        columns.insert(0, "ARR_DELAY")
    df = df[columns]
    df.to_csv(path_or_buf=out_path, index=False)
    return


def csv_transform(csv_path, dtypes, drop_columns, data_config):
    df = csv_to_df(csv_path, dtypes, data_config["skiprows"])
    df = df.drop(columns=drop_columns)
    dtypes = {key: dtypes[key] for key in dtypes if key not in drop_columns}
    df = onehot_encode(df, dtypes)
    write_csv(df, csv_path, data_config["transform_name"])
    return


def memory_efficient():
    drop_columns = ["YEAR", "TAIL_NUM", "FL_NUM", "DEST"] # skip DEST to avoid memory issues
    dtypes = {
        "YEAR": np.int64,
        "QUARTER": "category",
        "MONTH": "category",
        "DAY_OF_MONTH": "category",
        "DAY_OF_WEEK": "category",
        "UNIQUE_CARRIER": "category",
        "TAIL_NUM": "category",
        "FL_NUM": "category",
        "ORIGIN": "category",
        "DEST": "category",
        "CRS_DEP_TIME": np.int64,
        "DEP_TIME": np.int64,
        "DEP_DELAY": np.float64,
        "DEP_DELAY_NEW": np.float64,
        "DEP_DEL15": np.int64,
        "DEP_DELAY_GROUP": np.int64,
        "CRS_ARR_TIME": np.int64,
        "ARR_DELAY": np.float64,
        "CRS_ELAPSED_TIME": np.float64,
        "DISTANCE": np.float64,
        "DISTANCE_GROUP": "category"
    }
    numrows = 1114751 - 1  # minus 1 for header
    training_ratio = 0.8

    data_config = {
        "train": {
            "skiprows": range(math.floor(training_ratio * numrows) + 1, numrows + 2),
            "transform_name": "training"
        },
        "test": {
            "skiprows": range(1, math.floor(training_ratio * numrows) + 1),
            "transform_name": "test"
        }
    }
    csv_transform("data/Flights.csv", dtypes, drop_columns, data_config["train"])
    csv_transform("data/Flights.csv", dtypes, drop_columns, data_config["test"])

    # create dest files to be joined with other columns
    drop_columns = [column for column in dtypes if column not in ["DEST"]]
    data_config["train"]["transform_name"] = "training_dest"
    data_config["test"]["transform_name"] = "test_dest"
    csv_transform("data/Flights.csv", dtypes, drop_columns, data_config["train"])
    csv_transform("data/Flights.csv", dtypes, drop_columns, data_config["test"])


def naive():
    drop_columns = [["YEAR", "TAIL_NUM", "FL_NUM"],
                    ["YEAR", "TAIL_NUM", "FL_NUM"]]
    dtypes = {
        "YEAR": np.int64,
        "QUARTER": "category",
        "MONTH": "category",
        "DAY_OF_MONTH": "category",
        "DAY_OF_WEEK": "category",
        "UNIQUE_CARRIER": "category",
        "TAIL_NUM": "category",
        "FL_NUM": "category",
        "ORIGIN": "category",
        "DEST": "category",
        "CRS_DEP_TIME": np.int64,
        "DEP_TIME": np.int64,
        "DEP_DELAY": np.float64,
        "DEP_DELAY_NEW": np.float64,
        "DEP_DEL15": np.int64,
        "DEP_DELAY_GROUP": np.int64,
        "CRS_ARR_TIME": np.int64,
        "ARR_DELAY": np.float64,
        "CRS_ELAPSED_TIME": np.float64,
        "DISTANCE": np.float64,
        "DISTANCE_GROUP": "category"
    }
    numrows = 1114751 - 1  # minus 1 for header
    training_ratio = 0.8

## Better to process all at once and split at the end (if memory wasn't a concern)
    data_config = {
        "train": {
            "skiprows": range(math.floor(training_ratio * numrows) + 1, numrows + 2),
            "transform_name": "training"
        },
        "test": {
            "skiprows": range(1, math.floor(training_ratio * numrows) + 1),
            "transform_name": "test"
        }
    }
    csv_transform("data/Flights.csv", dtypes, drop_columns, data_config["train"])
    csv_transform("data/Flights.csv", dtypes, drop_columns, data_config["test"])

if __name__ == "__main__":
    #naive()
    memory_efficient()
