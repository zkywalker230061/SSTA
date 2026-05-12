#%%
from utils_read_nc import get_monthly_mean
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
from scipy.ndimage import gaussian_filter

#%%

#---1. ---Read File--------------------------------------
MIXED_LAYER_DS_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Datasets.nc"
# OLD_MIXED_LAYER_DS_PATH = 

mixed_layer_ds = xr.open_dataset(
    MIXED_LAYER_DS_PATH,
    engine="netcdf4",
    decode_times=False,   # disable decoding to avoid error
    mask_and_scale=True,
)

#%%
#---2. Extract Different Datasets -----------------------

# Mixed Layer Temperature Datasets 
Tm = mixed_layer_ds["MIXED_LAYER_TEMP"]
updated_Tm = mixed_layer_ds["UPDATED_MIXED_LAYER_TEMP"]

# Mixed Layer Salinity Datasets 
Sm = mixed_layer_ds["MIXED_LAYER_SALINITY"]
updated_Sm = mixed_layer_ds["UPDATED_MIXED_LAYER_SALINITY"]

#%%
#---3. Calculate Climatology Means of Datasets ----------------

# Tm bar
Tm_bar = get_monthly_mean(Tm)
updated_Tm_bar = get_monthly_mean(updated_Tm)

# Sm bar

Sm_bar = get_monthly_mean(Sm)
updated_Sm_bar = get_monthly_mean(updated_Sm)

#%%
#---4. Compute Gradients --------------------------

# Temperature (Original Dataset)
dTm_dx = compute_gradient_lon(Tm_bar)
dTm_dy = compute_gradient_lat(Tm_bar)

# Temperature (New Dataset)
d_newTm_dx = compute_gradient_lon(updated_Tm_bar)
d_newTm_dy = compute_gradient_lat(updated_Tm_bar)

# Salinity (Original Dataset)
dSm_dx = compute_gradient_lon(Sm_bar)
dSm_dy = compute_gradient_lat(Sm_bar)

# Salinity (New Dataset)
d_newSm_dx = compute_gradient_lon(updated_Sm_bar)
d_newSm_dy = compute_gradient_lat(updated_Sm_bar)


#%%
#---5a. Test Plots: Temp Gradients (Latitude)
date = 1

t0a = dTm_dy.sel(MONTH=date)
t0b = d_newTm_dy.sel(MONTH=date)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

cmap = plt.get_cmap("RdYlBu_r").copy()
cmap.set_bad(color="black")

for ax, data, title in zip(
    axes,
    [t0a, t0b],
    ["Original Dataset", "New Dataset"]
):
    pc = ax.pcolormesh(
        data["LONGITUDE"],
        data["LATITUDE"],
        np.ma.masked_invalid(data),
        cmap=cmap,
        shading="auto",
        vmin = -1e-5,
        vmax = 1e-5 
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
fig.colorbar(pc, cax=cbar_ax, label="Temperature Latitude Gradient (°C/m)")
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

#---5b. Test Plots: Temp Gradients (Longitude)
date = 1

t0a = dTm_dx.sel(MONTH=date)
t0b = d_newTm_dx.sel(MONTH=date)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

cmap = plt.get_cmap("RdYlBu_r").copy()
cmap.set_bad(color="black")

for ax, data, title in zip(
    axes,
    [t0a, t0b],
    ["Original Dataset", "New Dataset"]
):
    pc = ax.pcolormesh(
        data["LONGITUDE"],
        data["LATITUDE"],
        np.ma.masked_invalid(data),
        cmap=cmap,
        shading="auto",
        vmin = -1e-5,
        vmax = 1e-5 
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
fig.colorbar(pc, cax=cbar_ax, label="Temperature Longitude Gradient (°C/m)")
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
#%%
#---5. Residual Plot: Temp Gradients (Lat)---------------
residual_grad_lat = d_newTm_dy - dTm_dy

date = 1
t0 = abs(residual_grad_lat.sel(MONTH=date))

# Copy the colormap and set NaN color
cmap = plt.get_cmap("RdYlBu_r").copy()
cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

plt.figure(figsize=(10,5))
pc = plt.pcolormesh(
    t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
    cmap=cmap, shading="auto", vmin=0, vmax=1e-7
)
plt.colorbar(pc, label="Temp Grad Lat Difference (degrees C/m)")
plt.title(f"Residual Temp Latitude Grad - {date}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
# plt.tight_layout()
plt.show()


#---6a. Test Plots: Salinity Gradients (Latitude)
date = 1

s0a = dSm_dy.sel(MONTH=date)
s0b = d_newSm_dy.sel(MONTH=date)

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

cmap = plt.get_cmap("RdYlBu_r").copy()
cmap.set_bad(color="black")

for ax, data, title in zip(
    axes,
    [s0a, s0b],
    ["Original Dataset", "New Dataset"]
):
    pc = ax.pcolormesh(
        data["LONGITUDE"],
        data["LATITUDE"],
        np.ma.masked_invalid(data),
        cmap=cmap,
        shading="auto",
        vmin = -1e-6,
        vmax = 1e-6
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
fig.colorbar(pc, cax=cbar_ax, label="Salinity Latitude Gradient")
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

#---6a. Test Plots: Salinity Gradients (Longitude)
date = 1

s0a = dSm_dx.sel(MONTH=date)
s0b = d_newSm_dx.sel(MONTH=date)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

cmap = plt.get_cmap("RdYlBu_r").copy()
cmap.set_bad(color="black")

for ax, data, title in zip(
    axes,
    [s0a, s0b],
    ["Original Dataset", "New Dataset"]
):
    pc = ax.pcolormesh(
        data["LONGITUDE"],
        data["LATITUDE"],
        np.ma.masked_invalid(data),
        cmap=cmap,
        shading="auto",
        vmin = -1e-6,
        vmax = 1e-6
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
fig.colorbar(pc, cax=cbar_ax, label="Salinity Longitude Gradient")
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

#%%
#---7. Saving into one dataset -------------------------------


# Temperature Renaming
dTm_dx = dTm_dx.rename("TEMP_LON_GRAD")
d_newTm_dx = d_newTm_dx.rename("UPDATED_TEMP_LON_GRAD")

dTm_dy = dTm_dy.rename("TEMP_LAT_GRAD")
d_newTm_dy = d_newTm_dy.rename("UPDATED_TEMP_LAT_GRAD")

# Salinity Renaming
dSm_dx = dSm_dx.rename("SALINITY_LON_GRAD")
d_newSm_dx = d_newSm_dx.rename("UPDATED_SALINITY_LON_GRAD")

dSm_dy = dSm_dy.rename("SALINITY_LAT_GRAD")
d_newSm_dy = d_newSm_dy.rename("UPDATED_SALINITY_LAT_GRAD")


derivative_ds = xr.merge([dTm_dx, dTm_dy,
                          d_newTm_dx, d_newTm_dy,
                          dSm_dx, dSm_dy,
                          d_newSm_dx, d_newSm_dy])

# output_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Climatology_Derivatives.nc"
# # print(derivative_ds)
# derivative_ds.to_netcdf(output_path)
# %%
