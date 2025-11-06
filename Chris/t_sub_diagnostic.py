import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

T_SUB_DATA_PATH = "../datasets/t_sub.nc"

t_sub_ds = load_and_prepare_dataset(T_SUB_DATA_PATH)

t_sub_ds['T_sub_ANOMALY'].sel(TIME=121.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r')
plt.show()
