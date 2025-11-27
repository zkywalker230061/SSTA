import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, get_eof, get_eof_with_nan_consideration, make_movie

TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"
T_SUB_DATA_PATH = "../datasets/t_sub.nc"
ENTRAINMENT_VEL_DATA_PATH = "../datasets/Entrainment_Velocity-(2004-2018).nc"

temp_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
map_mask = temp_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")
t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)

# # get eof of t_sub
# t_sub_n_modes = 60
# t_sub_anomaly_eof, t_sub_explained_variance = get_eof_with_nan_consideration(t_sub_ds["T_sub_ANOMALY"], map_mask, modes=t_sub_n_modes)
#
# print("Variance explained:", t_sub_explained_variance[:t_sub_n_modes].sum().item())
# make_movie(t_sub_anomaly_eof, -3, 3)
#
# t_sub_ds["T_sub_ANOMALY_DENOISED"] = t_sub_anomaly_eof
# t_sub_ds.to_netcdf("../datasets/t_sub_denoised.nc")

# get eof of entrainment_vel
entrainment_vel_n_modes = 30
entrainment_vel_eof, entrainment_vel_explained_variance = get_eof_with_nan_consideration(entrainment_vel_ds['ENTRAINMENT_VELOCITY'], map_mask, modes=entrainment_vel_n_modes)

print("Variance explained:", entrainment_vel_explained_variance[:entrainment_vel_n_modes].sum().item())
make_movie(entrainment_vel_eof, -5e-5, 5e-5)

entrainment_vel_ds["ENTRAINMENT_VELOCITY_DENOISED"] = entrainment_vel_eof
entrainment_vel_ds.to_netcdf("../datasets/entrainment_vel_denoised.nc")

