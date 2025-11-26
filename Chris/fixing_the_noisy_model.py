"""The mathematical model when including exponential data can really get big, since the exponential is not necessarily
negative. Here I try a few attempts to salvage it. Will it work? well, does a fish know to fear the land?"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, make_movie, get_eof, get_simple_eof
import xeofs as xe

NOISY_DATASET_PATH = "../datasets/model_anomaly_exponential_damping_implicit.nc"
TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"

noisy_ds = xr.open_dataset(NOISY_DATASET_PATH, decode_times=False)
noisy_ds = noisy_ds.drop_vars("PRESSURE")
print(noisy_ds)
temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH, time_standard=True)
map_mask = temperature_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5)

# eof in the rather complicated way
# eof_ds, variance = get_eof(noisy_ds["ARGO_TEMPERATURE_ANOMALY"], map_mask, modes=1)
# print("Variance explained:", variance[:3].sum().item())

# simple eof
simple_eof_ds, variance = get_simple_eof(noisy_ds["ARGO_TEMPERATURE_ANOMALY"], mask=map_mask, modes=5, clean_nan=True)
print("Variance explained:", variance[:3].sum().item())
make_movie(simple_eof_ds)
