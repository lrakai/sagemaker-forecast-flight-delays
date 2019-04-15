import pandas as pd
import numpy as np
import matplotlib as plt


def csv_to_df(csv_path):
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
        "DEP_DEL15":  "category",
        "DEP_DELAY_GROUP":  np.int64,
        "CRS_ARR_TIME":  np.int64,
        "ARR_DELAY":  np.float64,
        "CRS_ELAPSED_TIME":  np.float64,
        "DISTANCE":  np.float64,
        "DISTANCE_GROUP":  "category"
    }
    df = pd.read_csv(csv_path, dtype=dtypes)
    return df


def onehot_encode(df):
    return


def csv_transform(csv_path, transform_name):
    df = csv_to_df(csv_path)
    df = df.drop(columns=['YEAR'])
    return


if __name__ == '__main__':
    csv_transform("data/Flights_Dec2016-Nov2017_sampled.csv", "simple")
