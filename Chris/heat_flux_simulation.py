import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import esmpy as ESMF
from utils import get_monthly_mean, get_anomaly

HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"

heat_flux_ds = xr.open_dataset(HEAT_FLUX_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['slhf'] + heat_flux_ds['sshf']
temperature_ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
print(heat_flux_anomaly_ds)


