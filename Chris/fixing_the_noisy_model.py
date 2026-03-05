"""The mathematical model when including exponential data can really get big, since the exponential is not necessarily
negative. Here I try a few attempts to salvage it. Will it work? well, does a fish know to fear the land?"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, make_movie, get_eof, get_eof_with_nan_consideration
import xeofs as xe

NOISY_DATASET_PATH = "../datasets/model_anomaly_exponential_damping_implicit.nc"
TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"

noisy_ds = xr.open_dataset(NOISY_DATASET_PATH, decode_times=False)
noisy_ds = noisy_ds.drop_vars("PRESSURE")
print(noisy_ds)
temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH, time_standard=True)
map_mask = temperature_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5)

# eof with predictive replacement of NaN; first change unphysical values to NaN
noisy_ds["ARGO_TEMPERATURE_ANOMALY"] = noisy_ds["ARGO_TEMPERATURE_ANOMALY"].where((noisy_ds["ARGO_TEMPERATURE_ANOMALY"] > -10) & (noisy_ds["ARGO_TEMPERATURE_ANOMALY"] < 10))
n_modes = 20
monthly_mean = get_monthly_mean(noisy_ds["ARGO_TEMPERATURE_ANOMALY"])
eof_ds, variance = get_eof_with_nan_consideration(noisy_ds["ARGO_TEMPERATURE_ANOMALY"], map_mask, modes=n_modes, monthly_mean_ds=None)
print("Variance explained:", variance[:n_modes].sum().item())
make_movie(eof_ds, -4, 4)
noisy_ds["EMEOF_DENOISED_ANOMALY"] = eof_ds
noisy_ds.to_netcdf("../datasets/cur_prev_denoised.nc")


# simple eof
# n_modes_simple = 5
# simple_eof_ds, variance = get_eof(noisy_ds["ARGO_TEMPERATURE_ANOMALY"], mask=map_mask, modes=n_modes_simple, clean_nan=True)
# print("Variance explained:", variance[:n_modes_simple].sum().item())
# make_movie(simple_eof_ds)
