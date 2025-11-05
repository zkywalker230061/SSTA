#%%
from read_nc import fix_rg_time, fix_longitude_coord, save_with_datetime
from calculate_Tm_Sm import depth_dbar_to_meter, _full_field, mld_dbar_to_meter, vertical_integral, z_to_xarray
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
import matplotlib

#%%

#---1. ---Read File--------------------------------------
#temp_file_path = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
salinity_file_path = r"C:\Users\jason\MSciProject\RG_ArgoClim_Salinity_2019.nc"
updated_h_bar_file_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"


# ds_temp = xr.open_dataset(
#     temp_file_path,
#     engine="netcdf4",
#     decode_times=False,   # disable decoding to avoid error
#     mask_and_scale=True,
# )

ds_sal = xr.open_dataset(
    salinity_file_path,
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

h_bar = xr.open_dataset(
    updated_h_bar_file_path,
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True
)


S_mean = ds_sal["ARGO_SALINITY_MEAN"]
S_anom = ds_sal["ARGO_SALINITY_ANOMALY"]
S_full = _full_field(S_mean, S_anom)
S_full = fix_longitude_coord(S_full)


def month_idx (time_da: xr.DataArray) -> xr.DataArray:
    n = time_da.sizes['TIME']
    # Repeat 1..12 along TIME; align coords to TIME
    month_idx = (xr.DataArray(np.arange(n) % 12 + 1, dims=['TIME'])
                 .assign_coords(TIME=time_da))
    month_idx.name = 'MONTH'
    return month_idx

def get_monthly_mean(da: xr.DataArray,) -> xr.DataArray:
    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    
    m = month_idx(da['TIME'])
    monthly_mean_da = da.groupby(m).mean('TIME', keep_attrs=True)
    # monthly_means = []
    # for _, month_num in MONTHS.items():
    #     monthly_means.append(
    #         da.sel(TIME=da['TIME'][month_num-1::12]).mean(dim='TIME')
    #     )
    # monthly_mean_da = xr.concat(monthly_means, dim='MONTH')
    # monthly_mean_da = monthly_mean_da.assign_coords(MONTH=list(MONTHS.values()))
    # monthly_mean_da['MONTH'].attrs['units'] = 'month'
    # monthly_mean_da['MONTH'].attrs['axis'] = 'M'
    # monthly_mean_da.attrs['units'] = da.attrs.get('units')
    # monthly_mean_da.attrs['long_name'] = f"Seasonal Cycle Mean of {da.attrs.get('long_name')}"
    # monthly_mean_da.name = f"MONTHLY_MEAN_{da.name}"
    return monthly_mean_da

def load_pressure_data(path: str, varname: str, *, compute_time_mode: str = "datetime",) -> xr.DataArray:
    """Load MLD in PRESSURE units, fix time, convert to meters (positive down)."""

    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=True)
    ds = fix_rg_time(ds, mode=compute_time_mode)

    pressure = ds[varname] # Coordinates = (TIME: 180, LATITUDE: 145, LONGITUDE: 360)
    lat_1D = ds["LATITUDE"]
    lat_3D = xr.broadcast(lat_1D, pressure)[0].transpose(*pressure.dims)
    depth_m   = mld_dbar_to_meter(pressure, lat_3D)
    depth_m   = fix_longitude_coord(depth_m)

    # print('depth_bar:\n',depth_bar, depth_bar.shape)
    # print(lat_3D)
    # print('depth_m:\n',depth_m)
    # print('depth_m after fix_longitude:\n', depth_m)
    return depth_m

ds_Sbar_monthly = get_monthly_mean(S_full)                                           # MONTH: 12, PRESSURE: 58, LATITUDE: 145, LONGITUDE: 360)
ds_h_bar_monthly = load_pressure_data(updated_h_bar_file_path,                       # MONTH: 12, LATITUDE: 145, LONGITUDE: 360
                                      'MONTHLY_MEAN_MLD_PRESSURE', 
                                      compute_time_mode="none")

print('ds_sbar_monthly:\n', ds_Sbar_monthly)
print('ds_h_bar_monthly:\n', ds_h_bar_monthly)

#----3. gsw--------------------------------------------
p = ds_sal['PRESSURE']
lat = ds_sal['LATITUDE']
depth = depth_dbar_to_meter(p,lat)

ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
TDIM = "TIME"
T_VAR = "ARGO_TEMPERATURE_MEAN"
S_VAR = "ARGO_SALINITY_MEAN"
T_VAR_ANOMALY = "ARGO_TEMPERATURE_ANOMALY"

dz = z_to_xarray(depth, ds_Sbar_monthly)                 #TIME: 180, PRESSURE: 58, LATITUDE: 145, LONGITUDE: 360
print(dz)

#%%
#----4. Vertical Integration -------------------------------
temp_mld_bar = vertical_integral(ds_Sbar_monthly, dz, ds_h_bar_monthly)          #??????i changed here to -z_new

print('vertical_integral:\n', temp_mld_bar)



#%%
print(matplotlib.__version__)
#%%
if __name__ == "__main__":
    #----Plot Salinity Map----------------------------------------------------
    date = 1
    t0 = temp_mld_bar.sel(MONTH=date)

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

    
# %%
