"""
Jilia Xie
"""

from calculate_Tm_Sm import depth_from_pressure, _full_field, fix_rg_time
# from calculate_Tm_Sm import mean_temperature_Tm, mean_salinity_Ts
from calculate_Tm_Jason import z_to_xarray, vertical_integral
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import re
from tqdm import tqdm
from typing import Optional
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


#---1. -------------------------------------------
ds_temp = xr.open_dataset(
    "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc",
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

ds_sal = xr.open_dataset(
    "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

ds_temp = fix_rg_time(ds_temp)
ds_sal = fix_rg_time(ds_sal)

T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]          # (P, Y, X)
T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]
T_full = _full_field(T_mean, T_anom)

#----2. gsw--------------------------------------------
p = ds_temp['PRESSURE']
lat = ds_temp['LATITUDE']

depth = depth_from_pressure(p,lat)


#-----3. z to x array-----------------------------------------
h_meters = 100.0
ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
TDIM = "TIME"
T_VAR = "ARGO_TEMPERATURE_MEAN"
S_VAR = "ARGO_SALINITY_MEAN"
T_VAR_ANOMALY = "ARGO_TEMPERATURE_ANOMALY"
  
z_new = z_to_xarray(depth, T_full)
#print(z_new)

#----4. Vertical Integration -------------------------------
vertical = vertical_integral(T_full,-z_new)          #??????i changed here to -z_new

# with ProgressBar():
#     vertical = vertical.compute()




if __name__ == "__main__":

    print(vertical)
    print(vertical.shape)


    #----Plot Map----------------------------------------------------
    t0 = vertical.sel(TIME="2010-07-01")

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], t0,
        cmap="RdYlBu_r", shading="auto", vmin=-2, vmax=30
    )
    plt.colorbar(pc, label="Mean Temperature (°C, 0–100 m)")
    plt.title("Upper 100 m Mean Temperature – June 2010")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()