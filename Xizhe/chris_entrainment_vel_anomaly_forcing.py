import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from chris_utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, coriolis_parameter, \
    get_month_from_time
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, compute_gradient_lon, compute_gradient_lat
import matplotlib

DOWNLOADED_SSH = False

H_DATA_PATH = "datasets/New MLD & T_sub/h.nc"
ENTRAINMENT_VEL_DATA_PATH = "datasets/files for simulate_implicit/Entrainment_Velocity-(2004-2018).nc"
T_M_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
H_BAR_DATA_PATH = "datasets/New MLD & T_sub/hbar.nc"
T_SUB_DATA_PATH = "datasets/New MLD & T_sub/t_sub.nc"

h_ds = xr.open_dataset(H_DATA_PATH, decode_times=False)
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
tm_ds = xr.open_dataset(T_M_DATA_PATH, decode_times=False)
tsub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)

h = h_ds["MLD"]
hbar = hbar_ds["MONTHLY_MEAN_MLD"]
entrainment_vel_monthly_mean = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_ds = get_anomaly(entrainment_vel_ds, "ENTRAINMENT_VELOCITY", entrainment_vel_monthly_mean)
entrainment_vel_anomaly = entrainment_vel_ds["ENTRAINMENT_VELOCITY_ANOMALY"]
tm = tm_ds["MIXED_LAYER_TEMP"]
tm_monthly_mean = get_monthly_mean(tm)
tsub = tsub_ds["SUB_TEMPERATURE"]
tsub_monthly_mean = get_monthly_mean(tsub)

rho_0 = 1025.0
c_0 = 4100.0


def get_entrainment_vel_forcing(time):
    month = get_month_from_time(time)
    return rho_0 * c_0 * entrainment_vel_anomaly.sel(TIME=time) * (tsub_monthly_mean.sel(MONTH=month) - tm_monthly_mean.sel(MONTH=month))

entrainment_vel_anomaly_forcings = []
for time in tm.TIME.values:
    entrainment_vel_anomaly_forcing = get_entrainment_vel_forcing(time)
    entrainment_vel_anomaly_forcing = entrainment_vel_anomaly_forcing.reset_coords("MONTH", drop=True)
    entrainment_vel_anomaly_forcings.append(entrainment_vel_anomaly_forcing)
entrainment_vel_anomaly_forcings = xr.concat(entrainment_vel_anomaly_forcings, "TIME")
entrainment_vel_anomaly_forcings = entrainment_vel_anomaly_forcings.rename("ENTRAINMENT_VEL_ANOMALY_FORCING")
# entrainment_vel_anomaly_forcings.to_netcdf("datasets/files for simulate_implicit/entrainment_velocity_anomaly_forcing.nc.nc")

# existing contribution:
# cur_entrainment_vel * cur_tsub_anom * rho_0 * c_0