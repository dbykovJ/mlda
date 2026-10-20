import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_z(df: pd.DataFrame, z_threshold: int):
    z_scores = np.abs(stats.zscore(df["e_distance"]))
    df = df[z_scores < z_threshold]
    return df