"""
JY
"""

#%%
from read_nc import fix_rg_time, fix_longitude_coord
from calculate_Tm_Sm import depth_dbar_to_meter, _full_field, mld_dbar_to_meter
from calculate_Tm_Sm import z_to_xarray, vertical_integral
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
updated_h_bar_file_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"


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

h_bar = xr.open_dataset(
    updated_h_bar_file_path,
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True
)

height_grid = height_grid["MLD_PRESSURE"]

print("H:\n", height_grid)

#%%

h_bar_test = np.tile(h_bar["MONTH"], 15)

for i in range(len(h_bar_test)):
    h_bar_test[i] = h_bar_test[i] + (i//12)*12 

h_bar_test = h_bar_test -0.5
# print("h_bar_test:\n",h_bar_test)


y = h_bar["MONTHLY_MEAN_MLD_PRESSURE"]
h_bar_extended = np.tile(y, (15,1,1))
# print("h_bar_extended:\n",h_bar_extended)

data = xr.Dataset(
    {
        "MLD_PRESSURE": (("TIME","LATITUDE","LONGITUDE"), h_bar_extended)
    },
    coords={
        "TIME": h_bar_test,
        "LATITUDE": h_bar["LATITUDE"],
        "LONGITUDE": h_bar["LONGITUDE"]
    },
    attrs={
        "TIME": "months since 2004-01-01"
    }
)

data.TIME.attrs["units"] = "months since 2004-01-01"
data["MLD_PRESSURE"].attrs["units"] = "dbar"  # if that’s the correct unit
data = fix_rg_time(data)
print("data:\n",data["MLD_PRESSURE"])

# print("h_bar\n", h_bar)







#%%
#---2. Fixing Time Coordinate-----------
# ds_temp = fix_rg_time(ds_temp)
# h_normal = fix_rg_time(height_grid)

# ds_sal = fix_rg_time(ds_sal)

T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]          # (P, Y, X)
T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]
T_full = _full_field(T_mean, T_anom)
T_full = fix_longitude_coord(T_full)



# print('T_full:\n',T_full)
# print(T_full.shape)
# print('h_normal:\n',h_normal)
# print(T_full.coords)
#%%
#----3. gsw--------------------------------------------
p = ds_temp['PRESSURE']
lat = ds_temp['LATITUDE']
depth = depth_dbar_to_meter(p,lat)
height_grid = mld_dbar_to_meter(height_grid,height_grid['LATITUDE'])

print(height_grid)

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
vertical = vertical_integral(T_full,z_new, height_grid)          #??????i changed here to -z_new

print('vertical_integral:\n',vertical)
#%% ----5. Gradient Test --------
gradient_lat = compute_gradient_lat(vertical)
#print('gradient_lat:\n',gradient_lat)

gradient_lon = compute_gradient_lon(vertical)
#print('gradient_lon:\n',gradient_lon)

grad_mag = np.sqrt(gradient_lat**2 + gradient_lon**2)
#print('grad_mag:\n',grad_mag)


grad_Tfull_lat = compute_gradient_lat(T_full)
print('grad_Tfull_lat:\n',grad_Tfull_lat)
grad_Tfull_lon = compute_gradient_lon(T_full)
print('grad_Tfull_lon:\n',grad_Tfull_lon)


#%%


vertical.to_netcdf(path= r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc")
# print(gradient_lat)
# print(gradient_lon)

# Create copies to save 
# save_with_datetime(gradient_lat, r"C:\Users\jason\MSciProject\gradient_lat_test.nc")
# save_with_datetime(gradient_lon, r"C:\Users\jason\MSciProject\gradient_lon_test.nc")
# save_with_datetime(grad_mag,     r"C:\Users\jason\MSciProject\gradient_magnitude_test.nc")
#%%
if __name__ == "__main__":
    #----Plot Temperature Map----------------------------------------------------
    date = 0.5
    t0 = vertical.sel(TIME=f"{date}")

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto", vmin=-5, vmax=35
    )
    plt.colorbar(pc, label="Mean Temperature (°C)")
    plt.title(f"Mixed Layer Temperature - {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()
    # #----Plot Gradient Map (Lat)----------------------------------------------------
    # t0 = gradient_lat.sel(TIME=f"{date}")

    # # Copy the colormap and set NaN color
    # cmap = plt.get_cmap("RdYlBu_r").copy()
    # cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    # plt.figure(figsize=(10,5))
    # pc = plt.pcolormesh(
    #     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
    #     cmap=cmap, shading="auto", vmin= -1e-5, vmax=1e-5
    # )
    # plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    # plt.title(f"Mixed Layer Temperature Gradient (Lat)- {date}")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.tight_layout()
    # plt.show()

    # #----Plot Gradient Map (Lon)----------------------------------------------------
    # t0 = gradient_lon.sel(TIME=f"{date}")

    # # Copy the colormap and set NaN color
    # cmap = plt.get_cmap("RdYlBu_r").copy()
    # cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    # plt.figure(figsize=(10,5))
    # pc = plt.pcolormesh(
    #     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
    #     cmap=cmap, shading="auto", vmin = -1e-5, vmax=1e-5
    # )
    # plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    # plt.title(f"Mixed Layer Temperature Gradient (Lon)- {date}")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.tight_layout()
    # plt.show()

    # #----Plot Gradient Map (Test on T full Lat)----------------------------------------------------
    # t0 = grad_Tfull_lat.sel(TIME=f"{date}")

    # # Copy the colormap and set NaN color
    # cmap = plt.get_cmap("RdYlBu_r").copy()
    # cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    # plt.figure(figsize=(10,5))
    # pc = plt.pcolormesh(
    #     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
    #     cmap=cmap, shading="auto", vmin = -1e-5, vmax=1e-5
    # )
    # plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    # plt.title(f"Mixed Layer Temperature Gradient (T_FULL Lon)- {date}")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.tight_layout()
    # plt.show()

    # #----Plot Gradient Map (Test on T full Lon)----------------------------------------------------
    # t0 = grad_Tfull_lon.sel(TIME=f"{date}")

    # # Copy the colormap and set NaN color
    # cmap = plt.get_cmap("RdYlBu_r").copy()
    # cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    # plt.figure(figsize=(10,5))
    # pc = plt.pcolormesh(
    #     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
    #     cmap=cmap, shading="auto", vmin = -1e-5, vmax=1e-5
    # )
    # plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    # plt.title(f"Mixed Layer Temperature Gradient (T_FULL Lon)- {date}")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.tight_layout()
    # plt.show()
    #----Plot Gradient Map (Magnitude)----------------------------------------------------
    # t0 = grad_mag.sel(TIME=f"{date}")

    # # Copy the colormap and set NaN color
    # cmap = plt.get_cmap("RdYlBu_r").copy()
    # cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    # plt.figure(figsize=(10,5))
    # pc = plt.pcolormesh(
    #     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
    #     cmap=cmap, shading="auto", vmin = 0, vmax=1e-5
    # )
    # plt.colorbar(pc, label=f"Temperature Gradient (/m)")
    # plt.title(f"Mixed Layer Temperature Gradient Magnitude- {date}")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.tight_layout()
    # plt.show()


# %%
