import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, coriolis_parameter
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC = True
INCLUDE_GEOSTROPHIC_DISPLACEMENT = True
USE_DOWNLOADED_SSH = False
gamma_0 = 30.0

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC, USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_DISPLACEMENT)
ALL_SCHEMES_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name + ".nc"
MLD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
OBSERVATIONS_JJ_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/observed_anomaly_JJ.nc"

all_models_ds = xr.open_dataset(ALL_SCHEMES_DATA_PATH, decode_times=False)
mld_ds = xr.open_dataset(MLD_DATA_PATH, decode_times=False)
tsub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
obs_ds = xr.open_dataset(OBSERVATIONS_JJ_DATA_PATH, decode_times=False)

tm = obs_ds['__xarray_dataarray_variable__']
tm_monthly_mean = get_monthly_mean(tm)
obs_ds = get_anomaly(obs_ds, '__xarray_dataarray_variable__', tm_monthly_mean)
tm_anomaly = obs_ds['__xarray_dataarray_variable___ANOMALY']

tsub_anomaly = tsub_ds['T_sub_ANOMALY']

implicit_model = all_models_ds["IMPLICIT"]

# make_movie(tm_anomaly, -3, 3, savepath="../results/videos/obs_tm_anom.mp4")
# make_movie(tsub_anomaly, -3, 3, savepath="../results/videos/obs_tsub_anom.mp4")


# abs(xr.corr(tm_anomaly, tsub_anomaly, dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=1)
# plt.show()

abs(xr.corr(tm_anomaly, implicit_model, dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=1)
plt.show()


