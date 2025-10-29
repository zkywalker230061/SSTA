import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import esmpy as ESMF

HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"

heat_flux_ds = xr.open_dataset(HEAT_FLUX_DATA_PATH, decode_times=False)
temperature_ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)

print(temperature_ds)


