import numpy as np
from scipy import stats

def remove_outliers_z(df, z_threshold: int):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    print(f"Z-scores:\n{z_scores}")