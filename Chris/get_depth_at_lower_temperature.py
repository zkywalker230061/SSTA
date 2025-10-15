import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

TEMP_DATA_PATH = "RG_ArgoClim_Temperature_2019.nc"

ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
print(ds)
print(type(ds))

sst = ds['ARGO_TEMPERATURE_ANOMALY'].isel(PRESSURE=0)
print(sst)
mld_t = sst - 1
print(mld_t)
mld_pressure = ds.PRESSURE.interp(ARGO_TEMPERATURE_ANOMALY=mld_t)
