#%%
import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
import calendar
import pandas as pd
import seaborn as sns
import ffmpeg as ffmpeg

#%%
#---1. Read Files ----------------------------------------------------------
semi_implicit_path = r"C:\Users\jason\MSciProject\Semi_Implicit_Scheme_Test_ConstDamp(10)"
implicit_path = r"C:\Users\jason\MSciProject\Implicit_Scheme_Test_ConstDamp(10)"
explicit_path = r"C:\Users\jason\MSciProject\Explicit_Scheme_Test_ConstDamp(10)"
crank_path = r"C:\Users\jason\MSciProject\Crack_Scheme_Test_ConstDamp(10)"
observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc"
chris_path = r"C:\Users\jason\MSciProject\model_anomaly_exponential_damping_implicit.nc"




# --- Load and Prepare Data (assuming helper functions are correct) --------
observed_temp = xr.open_dataset(observed_path, decode_times=False)
implicit = load_and_prepare_dataset(implicit_path)
explicit = load_and_prepare_dataset(explicit_path)
crank = load_and_prepare_dataset(crank_path)
semi_implicit = load_and_prepare_dataset(semi_implicit_path)
chris = load_and_prepare_dataset(chris_path)


# --- Extracting the correct DataArray -------------------
temperature = observed_temp['__xarray_dataarray_variable__']
implicit = implicit["T_model_anom_implicit"]
explicit = explicit["T_model_anom_explicit"]
crank = crank["T_model_anom_crank_nicolson"]
semi_implicit = semi_implicit["T_model_anom_semi_implicit"]
chris = chris["ARGO_TEMPERATURE_ANOMALY"]

#---- Defining the Anomaly Temperature Dataset ----------------------
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)


#-----2. Defining Error Function Calculation-------------------
# def create_output_array(name, long_name):
#     """Pre-allocates an xr.DataArray with the correct coords."""
#     return xr.DataArray(
#         np.zeros,
#         coords=temperature_anomaly.coords,
#         dims=temperature_anomaly.dims,
#         name=name,
#         attrs={**temperature_anomaly.attrs,
#                'units': 'Wm^-2',
#                'long_name': long_name}
#     )


# times = implicit.TIME.values
# t0 = times[0]
# def calculate_error(test_data, observed_data):
#     error = test_data.isel(TIME=slice(1,None)) - observed_data.isel(TIME=slice(1,None))
#     error_mean = error.mean(dim=["LATITUDE", "LONGITUDE"], skipna=True)
#     error_std = error.std(dim=["LATITUDE", "LONGITUDE"], skipna=True)
#     return error, error_mean, error_std

# error_simp, error_mean_simp, error_std_simp = calculate_error(semi_implicit, temperature)

# plt.hist(error_mean_simp, bins=80)
# plt.title("Semi Implicit")
# plt.show()

# error_exp, error_mean_exp, error_std_exp = calculate_error(explicit, temperature)
# plt.hist(error_mean_simp, bins=80)
# plt.title("Explicit")
# plt.show()

# error_imp, error_mean_imp, error_std_imp = calculate_error(implicit, temperature)
# plt.hist(error_mean_imp, bins=80)
# plt.title("Implicit")
# plt.show()

# error_crank, error_mean_crank, error_std_crank = calculate_error(crank, temperature)
# plt.hist(error_mean_crank, bins=80)
# plt.title("Crank")
# plt.show()

# error_chris, error_mean_chris, error_std_chris = calculate_error(chris, temperature)
# plt.hist(error_mean_chris, bins=80)
# plt.title("Chris")
# plt.show()


# def calculate_spatial_error(test_data, observed_data):
#     spatial_error_list = []


#---New Approach-------------------------------
def consistent_mask(test_data, observed_data):
    valid = np.isfinite(test_data) & np.isfinite(observed_data)
    test_da = test_data.where(valid)
    observed_da = observed_data.where(valid)
    return observed_da

def calculate_error(test_data, observed_data):
    error = test_data.isel(TIME=slice(1,None)) - observed_data.isel(TIME=slice(1,None))
    # error_mean = error.mean(dim=["LATITUDE", "LONGITUDE"], skipna=True)
    # error_std = error.std(dim=["LATITUDE", "LONGITUDE"], skipna=True)
    # flattened_error = error.values.flatten()
    return error

#%%


temp_masked = consistent_mask(explicit, temperature_anomaly)
error_exp= calculate_error(explicit, temp_masked)
error_exp_flat = error_exp.values.flatten()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.histplot(error_exp_flat[~np.isnan(error_exp_flat)], bins=1000)
ax.set_xlabel("Error (K)")
ax.set_ylabel("Error Freq")
ax.set_title("Explicit Scheme Temporal Error Distribution")
# ax.set_xlim(-19e30,19e30)
ax.axvline(min(error_exp_flat), color='red', linestyle='--', label=f'Max Negative Error = {max(error_exp_flat):.4f})')
ax.axvline(max(error_exp_flat), color='red', linestyle='--', label=f'Max Positive Error = {max(error_exp_flat):.4f})')
ax.legend()

plt.show()

#%%

temp_masked = consistent_mask(implicit, temperature_anomaly)
error_imp = calculate_error(implicit, temp_masked)
error_imp_flat = error_imp.values.flatten()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.histplot(error_imp_flat[~np.isnan(error_imp_flat)], bins=1000)
ax.set_xlabel("Error (K)")
ax.set_ylabel("Error Freq")
ax.set_title("Implicit Scheme Temporal Error Distribution")
# ax.set_xlim(-19e30,19e30)
ax.axvline(min(error_imp_flat), color='red', linestyle='--', label=f'Max Negative Error = {max(error_imp_flat):.4f})')
ax.axvline(max(error_imp_flat), color='red', linestyle='--', label=f'Max Positive Error = {max(error_imp_flat):.4f})')
ax.legend()

plt.show()

temp_masked = consistent_mask(semi_implicit, temperature_anomaly)
error_simp = calculate_error(semi_implicit, temp_masked)
error_simp_flat = error_simp.values.flatten()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.histplot(error_simp_flat[~np.isnan(error_simp_flat)], bins=1000)
ax.set_xlabel("Error (K)")
ax.set_ylabel("Error Freq")
ax.set_title("Semi-Implicit Scheme Temporal Error Distribution")
# ax.set_xlim(-19e30,19e30)
ax.axvline(min(error_simp_flat), color='red', linestyle='--', label=f'Max Negative Error = {max(error_simp_flat):.4f})')
ax.axvline(max(error_simp_flat), color='red', linestyle='--', label=f'Max Positive Error = {max(error_simp_flat):.4f})')
ax.legend()

plt.show()

temp_masked = consistent_mask(crank, temperature_anomaly)
error_crank = calculate_error(crank, temp_masked)
error_crank_flat = error_crank.values.flatten()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.histplot(error_crank_flat[~np.isnan(error_crank_flat)], bins=1000)
ax.set_xlabel("Error (K)")
ax.set_ylabel("Error Freq")
ax.set_title("Crank Scheme Temporal Error Distribution")
# ax.set_xlim(-19e30,19e30)
ax.axvline(min(error_crank_flat), color='red', linestyle='--', label=f'Max Negative Error = {max(error_crank_flat):.4f})')
ax.axvline(max(error_crank_flat), color='red', linestyle='--', label=f'Max Positive Error = {max(error_crank_flat):.4f})')
ax.legend()

plt.show()


temp_masked = consistent_mask(chris, temperature_anomaly)
error_chris = calculate_error(chris, temp_masked)
error_chris_flat = error_chris.values.flatten()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.histplot(error_chris_flat[~np.isnan(error_chris_flat)], bins=1000)
ax.set_xlabel("Error (K)")
ax.set_ylabel("Error Freq")
ax.set_title("Chris Scheme Temporal Error Distribution")
# ax.set_xlim(-19e30,19e30)
ax.axvline(min(error_chris_flat), color='red', linestyle='--', label=f'Max Negative Error = {max(error_chris_flat):.4f})')
ax.axvline(max(error_chris_flat), color='red', linestyle='--', label=f'Max Positive Error = {max(error_chris_flat):.4f})')
ax.legend()

plt.show()


#%%
# x_array = np.linspace(1,179,179)

# fig, ax = plt.subplots(1,2, figsize=(16,12))
# ax[0].scatter(np.log(x_array), np.log(error_mean_exp))
# ax[0].set_ylabel("Log Mean Error (K)", fontsize = 15)
# ax[0].set_xlabel("Log No. of Time Step", fontsize=15)
# ax[0].set_title("Explicit Scheme Mean Error per Time Step (Log Scale)", fontsize=18)
# ax[0].grid()

# ax[1].scatter(np.log(x_array), np.log(error_std_exp))
# ax[1].set_ylabel("Log Mean Error (K)", fontsize = 15)
# ax[1].set_xlabel("Log No. of Time Step", fontsize=15)
# ax[1].set_title("Explicit Scheme Std of Error per Time Step (Log Scale)", fontsize=18)
# ax[1].grid()

# plt.tight_layout()
# plt.show()

#%%
# Spatial Error Datasets

ds_exp = error_exp
ds_imp = error_imp
ds_simp = error_simp
ds_crank = error_crank
ds_chris = error_chris

#%%

# Explicit Scheme Spatial Error
v = np.nanmax(np.abs(ds_exp))
vmin, vmax = -1e10, 1e10

# Coords
lats = ds_exp["LATITUDE"].values
lons = ds_exp["LONGITUDE"].values

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(
    ds_exp.isel(TIME=0).values,
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    origin="lower",
    vmin=vmin, vmax=vmax,
    cmap="RdBu_r",
    aspect="auto",
)
cb = plt.colorbar(im, ax=ax, shrink=0.8)
title = ax.set_title("Month 1 (January)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

def update(frame_idx):
    # frame_idx: 0..11
    arr = ds_exp.isel(TIME=frame_idx).values
    im.set_data(arr)
    month_num = int(ds_exp["TIME"].values[frame_idx])
    year = 2004 + (month_num - 1)//12
    moy = ((month_num -1) %12) + 1
    title.set_text(f"Explicit Scheme, Month {moy} in Year {year}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=179, interval=500, blit=False, repeat=True
)


_ = anim

HTML(anim.to_jshtml())
anim.save(r"C:\Users\jason\MSciProject\explicit_animation.gif", writer="pillow", fps=4)


#%%
#Implicit Scheme Spatial Mean Error

v = np.nanmax(np.abs(ds_imp))
vmin, vmax = -v,v

# Coords
lats = ds_imp["LATITUDE"].values
lons = ds_imp["LONGITUDE"].values

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(
    ds_imp.isel(TIME=0).values,
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    origin="lower",
    vmin=vmin, vmax=vmax,
    cmap="RdBu_r",
    aspect="auto",
)
cb = plt.colorbar(im, ax=ax, shrink=0.8, label="Error (K)")
title = ax.set_title("Month 1 (January)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

def update(frame_idx):
    arr = ds_imp.isel(TIME=frame_idx).values
    im.set_data(arr)
    month_num = int(ds_exp["TIME"].values[frame_idx])
    year = 2004 + (month_num - 1)//12
    moy = ((month_num -1) %12) + 1
    title.set_text(f"Implicit Scheme, Month {moy} in Year {year}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=179, interval=500, blit=False, repeat=True
)


_ = anim

HTML(anim.to_jshtml())
anim.save(r"C:\Users\jason\MSciProject\implicit_animation.gif", writer="pillow", fps=4)

#%%
#Semi-Implicit Scheme Spatial Mean Error

v = np.nanmax(np.abs(ds_simp))
vmin, vmax = -v,v

# Coords
lats = ds_simp["LATITUDE"].values
lons = ds_simp["LONGITUDE"].values

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(
    ds_simp.isel(TIME=0).values,
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    origin="lower",
    vmin=vmin, vmax=vmax,
    cmap="RdBu_r",
    aspect="auto",
)
cb = plt.colorbar(im, ax=ax, shrink=0.8, label="Error (K)")
title = ax.set_title("Month 1 (January)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

def update(frame_idx):
    arr = ds_simp.isel(TIME=frame_idx).values
    im.set_data(arr)
    month_num = int(ds_simp["TIME"].values[frame_idx])
    year = 2004 + (month_num - 1)//12
    moy = ((month_num -1) %12) + 1
    title.set_text(f"Implicit Scheme, Month {moy} in Year {year}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=179, interval=500, blit=False, repeat=True
)


_ = anim

HTML(anim.to_jshtml())
anim.save(r"C:\Users\jason\MSciProject\semi_implicit_animation.gif", writer="pillow", fps=4)
#%%
#Crank Scheme Spatial Mean Error

v = np.nanmax(np.abs(ds_crank))
vmin, vmax = -v,v

# Coords
lats = ds_crank["LATITUDE"].values
lons = ds_crank["LONGITUDE"].values

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(
    ds_crank.isel(TIME=0).values,
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    origin="lower",
    vmin=vmin, vmax=vmax,
    cmap="RdBu_r",
    aspect="auto",
)
cb = plt.colorbar(im, ax=ax, shrink=0.8, label="Error (K)")
title = ax.set_title("Month 1 (January)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

def update(frame_idx):
    arr = ds_crank.isel(TIME=frame_idx).values
    im.set_data(arr)
    month_num = int(ds_crank["TIME"].values[frame_idx])
    year = 2004 + (month_num - 1)//12
    moy = ((month_num -1) %12) + 1
    title.set_text(f"Implicit Scheme, Month {moy} in Year {year}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=179, interval=500, blit=False, repeat=True
)


_ = anim

HTML(anim.to_jshtml())
anim.save(r"C:\Users\jason\MSciProject\crank_animation.gif", writer="pillow", fps=4)

#%%
#Chris Integration Scheme Spatial Mean Error


v = np.nanmax(np.abs(ds_chris))
vmin, vmax = -v,v

# Coords
lats = ds_chris["LATITUDE"].values
lons = ds_chris["LONGITUDE"].values

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(
    ds_chris.isel(TIME=0).values,
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    origin="lower",
    vmin=vmin, vmax=vmax,
    cmap="RdBu_r",
    aspect="auto",
)
cb = plt.colorbar(im, ax=ax, shrink=0.8, label="Error (K)")
title = ax.set_title("Month 1 (January)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

def update(frame_idx):
    arr = ds_chris.isel(TIME=frame_idx).values
    im.set_data(arr)
    month_num = int(ds_chris["TIME"].values[frame_idx])
    year = 2004 + (month_num - 1)//12
    moy = ((month_num -1) %12) + 1
    title.set_text(f"Chris Scheme, Month {moy} in Year {year}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=179, interval=500, blit=False, repeat=True
)


_ = anim

HTML(anim.to_jshtml())
anim.save(r"C:\Users\jason\MSciProject\chris_animation.gif", writer="pillow", fps=4)
#%%



# fg = error_exp_monthly_clim.plot(
#     x="lon", y="lat",
#     col="month", col_wrap=3,
#     vmin=vmin, vmax=vmax,
#     cmap="RdBu_r",
#     robust=False,
#     figsize=(12, 12)
# )
# for ax in np.ravel(fg.axes):
#     ax.set_title(ax.get_title().replace("month = ", "Month "))





# %%
