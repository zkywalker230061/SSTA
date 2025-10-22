"""
Julia Xie, JY
"""

#%%
from read_nc import fix_rg_time, fix_longitude_coord
from calculate_Tm_Sm import depth_dbar_to_meter, _full_field, z_to_xarray, vertical_integral
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import re
from tqdm import tqdm
from typing import Optional
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

#%%
#---1. ---Read File----------------------------------------------------------------

# temp_file_path = "C:\Msci Project\RG_ArgoClim_Temperature_2019.nc"
# salinity_file_path = "C:\Msci Project\RG_ArgoClim_Salinity_2019.nc"
# updated_h_file_path = "C:\Msci Project\Mixed_Layer_Depth_Pressure (2004-2018).nc"
temp_file_path = "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
salinity_file_path = "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc"
updated_h_file_path = "/Users/xxz/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure (2004-2018).nc"

# ds_temp = xr.open_dataset(
#     temp_file_path,
#     engine="netcdf4",
#     decode_times=False,   # disable decoding to avoid error
#     mask_and_scale=True,
# )

ds_sal = xr.open_dataset(
    "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
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
print(height_grid)

#%%
#---2. Fixing Time Coordinate---------------------------------------------------
ds_sal = fix_rg_time(ds_sal)
h_normal = fix_rg_time(height_grid)
# ds_temp = fix_rg_time(ds_temp)

S_mean = ds_sal["ARGO_SALINITY_MEAN"]
S_anom = ds_sal["ARGO_SALINITY_ANOMALY"]
S_full = _full_field(S_mean, S_anom)
S_full = fix_longitude_coord(S_full)

print('S_full:\n',S_full)
print(S_full.shape)
print('h_normal:\n',h_normal)

#%%
#----3. GSW---------------------------------------------------------------------
lat = ds_sal['LATITUDE']
p = ds_sal['PRESSURE']
depth = depth_dbar_to_meter(p,lat)

ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
TDIM = "TIME"
T_VAR = "ARGO_TEMPERATURE_MEAN"
S_VAR = "ARGO_SALINITY_MEAN"
T_VAR_ANOMALY = "ARGO_TEMPERATURE_ANOMALY"
z_new = z_to_xarray(depth, S_full)

# print('z_new:\n',z_new)

#%%
#----4. Vertical Integration ---------------------------------------------------
vertical = vertical_integral(S_full,z_new, h_normal)          #??????i changed here to -z_new
print('vertical_integral:\n',vertical_integral)


#%%
#----5. Main  ------------------------------------------------------------------
if __name__ == "__main__":
    #----Plot Map---------------------------------------------------------------
    test_data = vertical.sel(TIME="2006-01-01")

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        test_data["LONGITUDE"], test_data["LATITUDE"], np.ma.masked_invalid(test_data),
        cmap=cmap, shading="auto", vmin=30, vmax=40
    )
    plt.colorbar(pc, label="Mean Salinity (Scale 78)")
    plt.title("Upper 100 m Mean Salinity - Jan 2006")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    #vertical.to_netcdf("Mean Salinity Dataset (2004-2018)")