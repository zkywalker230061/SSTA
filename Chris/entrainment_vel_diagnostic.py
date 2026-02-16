import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

ENTRAINMENT_VEL_DATA_PATH = "../datasets/Entrainment_Velocity-(2004-2018).nc"

entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])

vmin = 0
vmax = 6e-5

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=1).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=2).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=3).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=4).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=5).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=6).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=7).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=8).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=9).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=10).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=11).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()

entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'].sel(MONTH=12).plot(x='LONGITUDE', y='LATITUDE', cmap='berlin', vmin=vmin, vmax=vmax)
plt.show()
