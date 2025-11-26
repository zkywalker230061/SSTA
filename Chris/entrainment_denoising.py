import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, get_eof, get_eof_with_nan_consideration, make_movie

TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"
T_SUB_DATA_PATH = "../datasets/t_sub.nc"

temp_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)

# get eof
map_mask = temp_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")
n_modes = 70
t_sub_anomaly_eof, explained_variance = get_eof_with_nan_consideration(t_sub_ds["T_sub_ANOMALY"], map_mask, modes=n_modes)

print("Variance explained:", explained_variance[:n_modes].sum().item())
make_movie(t_sub_anomaly_eof, -3, 3)
