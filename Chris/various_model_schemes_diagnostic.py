import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie, get_eof, get_eof_with_nan_consideration
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

ALL_SCHEMES_DATA_PATH = "../datasets/all_anomalies.nc"
DENOISED_DATA_PATH = "../datasets/cur_prev_denoised.nc"

all_schemes_ds = xr.open_dataset(ALL_SCHEMES_DATA_PATH, decode_times=False)
denoised_scheme_ds = xr.open_dataset(DENOISED_DATA_PATH, decode_times=False)
all_schemes_ds["CHRIS_PREV_CUR_NAN"] = all_schemes_ds["CHRIS_PREV_CUR"].where((all_schemes_ds["CHRIS_PREV_CUR"] > -10) & (all_schemes_ds["CHRIS_PREV_CUR"] < 10))


# make_movie(all_schemes_ds["CHRIS_PREV_CUR"], -10, 10, colorbar_label="Chris Prev-Cur Scheme")
# make_movie(all_schemes_ds["CHRIS_MEAN_K"], -10, 10, colorbar_label="Chris Mean-k Scheme")
# make_movie(all_schemes_ds["CHRIS_PREV_K"], -10, 10, colorbar_label="Chris Prev-k Scheme")
# make_movie(all_schemes_ds["CHRIS_CAPPED_EXPONENT"], -10, 10, colorbar_label="Chris Capped Exponent Scheme")
# make_movie(all_schemes_ds["EXPLICIT"], -10, 10, colorbar_label="Explicit Scheme")
# make_movie(all_schemes_ds["IMPLICIT"], -10, 10, colorbar_label="Implicit Scheme")
# make_movie(all_schemes_ds["SEMI_IMPLICIT"], -10, 10, colorbar_label="Semi-Implicit Scheme")
# make_movie(denoised_scheme_ds["EMEOF_DENOISED_ANOMALY"], -10, 10, colorbar_label="Chris Prev-Cur Scheme Denoised")

TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"

temp_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
map_mask = temp_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")

temp_n_modes = 20
monthly_mean = get_monthly_mean(all_schemes_ds["CHRIS_PREV_CUR_NAN"])
temp_anomaly_eof, temp_explained_variance = get_eof_with_nan_consideration(all_schemes_ds["CHRIS_PREV_CUR_NAN"], modes=temp_n_modes, mask=map_mask, tolerance=1e-15, monthly_mean_ds=monthly_mean)

# temp_anomaly_eof, temp_explained_variance = get_eof(all_schemes_ds["CHRIS_MEAN_K"], modes=temp_n_modes, mask=map_mask, clean_nan=True)


print("Variance explained:", temp_explained_variance[:temp_n_modes].sum().item())
make_movie(temp_anomaly_eof, -4, 4)

# temp_anomaly_eof.to_netcdf("../datasets/chris_prev_cur_scheme_denoised.nc")
