import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter, get_month_from_time
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib

INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True


EKMAN_MEAN_ADVECTION_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/ekman_mean_advection.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"
SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_monthly_mean_calculated_grad.nc"

ekman_mean_advection = xr.open_dataset(EKMAN_MEAN_ADVECTION_DATA_PATH, decode_times=False)
sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)
sea_surface_monthlymean_ds = xr.open_dataset(SEA_SURFACE_MONTHLY_MEAN_DATA_PATH, decode_times=False)

delta_t = 30.4375 * 24 * 60 * 60

alpha = xr.zeros_like(sea_surface_monthlymean_ds['alpha'])
beta = xr.zeros_like(sea_surface_monthlymean_ds['beta'])

if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
    alpha += sea_surface_monthlymean_ds['alpha']
    beta += sea_surface_monthlymean_ds['beta']

if INCLUDE_EKMAN_MEAN_ADVECTION:
    alpha = alpha + ekman_mean_advection["ekman_alpha"]
    beta = beta + ekman_mean_advection["ekman_beta"]

earth_radius = 6371000
latitudes = np.deg2rad(sea_surface_monthlymean_ds['LATITUDE'])  # any ds to get latitude
dx = (2 * np.pi * earth_radius / 360) * np.cos(latitudes)
dy = (2 * np.pi * earth_radius / 360) * np.ones_like(latitudes)
dx = xr.DataArray(dx, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values}, dims=['LATITUDE'])  # convert dx, dy to xarray for use below
dy = xr.DataArray(dy, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values}, dims=['LATITUDE'])
CFL_x = (abs(alpha) * delta_t / dx)
CFL_y = (abs(beta) * delta_t / dy)

print(CFL_x.sel(MONTH=6).mean().values)
print(CFL_x.mean().values)
print(CFL_x.max().values)

CFL_x.sel(MONTH=1).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=25)
plt.show()

CFL_y.sel(MONTH=1).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=25)
plt.show()

CFL_x.sel(MONTH=7).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=25)
plt.show()

CFL_y.sel(MONTH=7).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=25)
plt.show()
