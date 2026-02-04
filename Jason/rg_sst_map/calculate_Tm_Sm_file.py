#%%
#---0. Importing Packages-----------------
from utils_read_nc import fix_rg_time, fix_longitude_coord, load_pressure_data
from utils_Tm_Sm import depth_dbar_to_meter, _full_field, mld_dbar_to_meter, vertical_integral, z_to_xarray
# from grad_field import compute_gradient_lat, compute_gradient_lon
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import re
from tqdm import tqdm
from typing import Optional
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter

#%%

#---1. File Paths for Temp, Salinity, Old h and New h--------------------------------------
temp_file_path = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
salinity_file_path = r"C:\Users\jason\MSciProject\RG_ArgoClim_Salinity_2019.nc"
h_file_path = r"C:\Users\jason\MSciProject\h.nc"
updated_h_file_path = r"C:\Users\jason\MSciProject\new_h.nc"

#%%
#---2. Read Temperature and Salinity Files -----------
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

#%%
#---3. Get Full Field Temperature and Salinity Dataset------------ 
T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]
T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]
T_full = _full_field(T_mean, T_anom)
T_full = fix_longitude_coord(T_full)


S_mean = ds_sal["ARGO_SALINITY_MEAN"]
S_anom = ds_sal["ARGO_SALINITY_ANOMALY"]
S_full = _full_field(S_mean, S_anom)
S_full = fix_longitude_coord(S_full)

print(T_full)
print(S_full)
#%%
#---4. Preparing old and new h datasets --------------

# Old h
h_da = load_pressure_data(h_file_path, 'MLD')

# New h
updated_h_da = load_pressure_data(updated_h_file_path, 'MLD')


#%%
#---5. Getting dz array from GSW functions-----------

# Temperature
p_temp = ds_temp['PRESSURE']
lat_temp = ds_temp['LATITUDE']
depth_temp = depth_dbar_to_meter(p_temp,lat_temp)

dz_temp = z_to_xarray(depth_temp, T_full) 

# Salinity
p_sal = ds_sal['PRESSURE']
lat_sal = ds_sal['LATITUDE']
depth_sal = depth_dbar_to_meter(p_sal, lat_sal)

dz_sal = z_to_xarray(depth_sal, S_full)


print(dz_temp)
print(dz_sal)
#%%
#---6. Computing Vertical Integral -------------------

# Mixed Layer Temperature Calculations 
Tm = vertical_integral(T_full, dz_temp, h_da)
updated_Tm = vertical_integral(T_full, dz_temp, updated_h_da)


# Mixed Layer Salinity Calculations 
Sm = vertical_integral(S_full, dz_sal, h_da)
updated_Sm = vertical_integral(S_full, dz_sal, updated_h_da)

# %%
#---7. Storing into a single dataset
Tm_renamed = Tm.rename("MIXED_LAYER_TEMP")
new_Tm_renamed = updated_Tm.rename("UPDATED_MIXED_LAYER_TEMP")
Sm_renamed = Sm.rename("MIXED_LAYER_SALINITY")
new_Sm_renamed = updated_Sm.rename("UPDATED_MIXED_LAYER_SALINITY")


Tm_Sm_ds = xr.merge([Tm_renamed, new_Tm_renamed, Sm_renamed, new_Sm_renamed])
print(Tm_Sm_ds)
output_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Datasets.nc"
Tm_Sm_ds.to_netcdf(output_path)
# %%
