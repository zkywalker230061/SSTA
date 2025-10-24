"""
Julia Xie
"""

#%%
from read_nc import fix_rg_time, fix_longitude_coord
from calculate_Tm_Sm import depth_dbar_to_meter, _full_field, z_to_xarray, vertical_integral, mld_dbar_to_meter
from grad_field import compute_gradient_lat, compute_gradient_lon
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

#%%
#---1. ---Read File--------------------------------------

# temp_file_path = "C:\Msci Project\RG_ArgoClim_Temperature_2019.nc"
# salinity_file_path = "C:\Msci Project\RG_ArgoClim_Salinity_2019.nc"
# updated_h_file_path = "C:\Msci Project\Mixed_Layer_Depth_Pressure (2004-2018).nc"
temp_file_path = "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
salinity_file_path = "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc"
updated_h_file_path = "/Users/xxz/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure (2004-2018).nc"

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

ds_depth = xr.open_dataset(
    updated_h_file_path,
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True
)

#%%
#---2. Fixing Time Coordinate----------------------------------------------
ds_temp = fix_rg_time(ds_temp)
ds_depth = fix_rg_time(ds_depth)
# ds_sal = fix_rg_time(ds_sal)
pressure = ds_depth["MLD_PRESSURE"]
lat_3D = xr.broadcast(ds_depth["LATITUDE"], pressure)[0].transpose(*pressure.dims)
depth_m = mld_dbar_to_meter(pressure, lat_3D)
depth_m = fix_longitude_coord(depth_m)

T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]          # (P, Y, X)
T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]
T_full = _full_field(T_mean, T_anom)
T_full = fix_longitude_coord(T_full)

print('T_full:\n',T_full)
print(T_full.shape)
print('depth in meter:\n',depth_m)

#%%
#----3. GSW-----------------------------------------------------------------
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

# print('z_new:\n',z_new)

#%%
#----4. Vertical Integration ---------------------------------------------
vertical = vertical_integral(T_full,z_new, depth_m)          #??????i changed here to -z_new
print('vertical_integral:\n',vertical_integral)
gradient_lat = compute_gradient_lat(vertical)
print('gradient_lat:\n',gradient_lat)
gradient_lon = compute_gradient_lon(vertical)
print('gradient_lon:\n',gradient_lon)

#%%
if __name__ == "__main__":
    #----Plot Map----------------------------------------------------
    date = "2014-10-01"
    test_data = vertical.sel(TIME=f"{date}")

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        test_data["LONGITUDE"], test_data["LATITUDE"], np.ma.masked_invalid(test_data),
        cmap=cmap, shading="auto", vmin=-5, vmax=35
    )
    plt.colorbar(pc, label="Mean Temperature (°C)")
    plt.title(f"Mixed Layer Depth Mean Temperature {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    # vertical.to_netcdf("Mean Temperature Dataset (2004-2018)")

    # %%
    #----Gradient Latitude and Longtitude Plot-----------------------
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

    t0 = gradient_lon.sel(TIME=f"{date}")

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