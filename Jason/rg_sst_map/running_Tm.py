"""
JY
"""

#%%
from read_nc import fix_rg_time, fix_longitude_coord
from calculate_Tm_Sm import depth_dbar_to_meter, _full_field
from calculate_Tm_Sm import z_to_xarray
from calculate_Tm import vertical_integral
from grad_field import compute_gradient_lat, compute_gradient_lon
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

#---1. ---Read File--------------------------------------

temp_file_path = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
salinity_file_path = r"C:\Users\jason\MSciProject\RG_ArgoClim_Salinity_2019.nc"
updated_h_file_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure (2004-2018).nc"



ds_temp = xr.open_dataset(
    temp_file_path,
    engine="netcdf4",
    decode_times=False,   # disable decoding to avoid error
    mask_and_scale=True,
)

# ds_sal = xr.open_dataset(
#     salinity_file_path,
#     engine="netcdf4",
#     decode_times=False,
#     mask_and_scale=True,
# )

height_grid = xr.open_dataset(
    updated_h_file_path,
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True
)

height_grid = height_grid["MLD_PRESSURE"]

#%%
#---2. Fixing Time Coordinate-----------
ds_temp = fix_rg_time(ds_temp)
h_normal = fix_rg_time(height_grid)

# ds_sal = fix_rg_time(ds_sal)

T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]          # (P, Y, X)
T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]
T_full = _full_field(T_mean, T_anom)
T_full = fix_longitude_coord(T_full)


print('T_full:\n',T_full)
print(T_full.shape)
print('h_normal:\n',h_normal)
print(T_full.coords)
#%%
#----3. gsw--------------------------------------------
p = ds_temp['PRESSURE']
lat = ds_temp['LATITUDE']
depth = depth_dbar_to_meter(p,lat)



ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
TDIM = "TIME"
T_VAR = "ARGO_TEMPERATURE_MEAN"
S_VAR = "ARGO_SALINITY_MEAN"
T_VAR_ANOMALY = "ARGO_TEMPERATURE_ANOMALY"
z_new = z_to_xarray(depth, T_full)

print('z_new:\n',z_new)

#%%
#----4. Vertical Integration -------------------------------
vertical = vertical_integral(T_full,z_new, h_normal)          #??????i changed here to -z_new

print('vertical_integral:\n',vertical)
#%% ----5. Gradient Test --------
gradient_lat = compute_gradient_lat(vertical)
print('gradient_lat:\n',gradient_lat)
gradient_lon = compute_gradient_lon(vertical)
print('gradient_lon:\n',gradient_lon)




#%%
if __name__ == "__main__":
    #----Plot Temperature Map----------------------------------------------------
    date = "2015-10-01"
    t0 = vertical.sel(TIME=f"{date}")

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto"
    )
    plt.colorbar(pc, label="Mean Temperature (°C)")
    plt.title(f"Mixed Layer Temperature - {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    #----Plot Gradient Map (Lon)----------------------------------------------------
    t0 = gradient_lat.sel(TIME=f"{date}")

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto", vmin= -1e-5, vmax=1e-5
    )
    plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    plt.title(f"Mixed Layer Temperature Gradient (Lat)- {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    #----Plot Gradient Map (Lon)----------------------------------------------------
    t0 = gradient_lon.sel(TIME=f"{date}")

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto", vmin = -1e-5, vmax=1e-5
    )
    plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    plt.title(f"Mixed Layer Temperature Gradient (Lon)- {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    # vertical.to_netcdf("Mean Temperature Dataset (2004-2018)")


# %%
