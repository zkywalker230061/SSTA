"""
Jilia Xie
"""

#%%
from read_nc import fix_rg_time
from calculate_Tm_Sm import depth_from_pressure, _full_field
from calculate_Tm_Sm import z_to_xarray
from calculte_Tm import vertical_integral
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

temp_file_path = "C:\Msci Project\RG_ArgoClim_Temperature_2019.nc"
salinity_file_path = "C:\Msci Project\RG_ArgoClim_Salinity_2019.nc"
updated_h_file_path = "C:\Msci Project\Mixed_Layer_Depth_Pressure (2004-2018).nc"


ds_temp = xr.open_dataset(
    temp_file_path,
    engine="netcdf4",
    decode_times=False,   # disable decoding to avoid error
    mask_and_scale=True,
)

ds_sal = xr.open_dataset(
    salinity_file_path,
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

height_grid = xr.open_dataset(
    updated_h_file_path,
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True
)

height_grid = height_grid["MLD_PRESSURE"]

ds_temp = fix_rg_time(ds_temp)
h_normal = fix_rg_time(height_grid)

# ds_sal = fix_rg_time(ds_sal)

T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]          # (P, Y, X)
T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]
T_full = _full_field(T_mean, T_anom)

print(h_normal)
#%%
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

print(z_new)

#%%

#----4. Vertical Integration -------------------------------
vertical = vertical_integral(T_full,z_new, h_normal)          #??????i changed here to -z_new

# with ProgressBar():
#     vertical = vertical.compute()




if __name__ == "__main__":

    print(vertical)
    print(vertical.sizes)

    #----Plot Map----------------------------------------------------
    t0 = vertical.sel(TIME="2006-01-01")

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], t0,
        cmap="RdYlBu_r", shading="auto", vmin=-2, vmax=30
    )
    plt.colorbar(pc, label="Mean Temperature (°C, 0–100 m)")
    plt.title("Upper 100 m Mean Temperature - Jan 2006")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    # vertical.to_netcdf("Mean Temperature Dataset (2004-2018)")


# %%
