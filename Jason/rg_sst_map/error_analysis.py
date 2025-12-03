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
from scipy.stats import gaussian_kde
from scipy.stats import zscore

#%%
#---1. Read Files ----------------------------------------------------------
semi_implicit_path = r"C:\Users\jason\MSciProject\Semi_Implicit_Scheme_Test_ConstDamp(10)"
implicit_path = r"C:\Users\jason\MSciProject\Implicit_Scheme_Test_ConstDamp(10)"
explicit_path = r"C:\Users\jason\MSciProject\Explicit_Scheme_Test_ConstDamp(10)"
crank_path = r"C:\Users\jason\MSciProject\Crack_Scheme_Test_ConstDamp(10)"
observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc"
chris_path = r"C:\Users\jason\MSciProject\model_anomaly_exponential_damping_implicit.nc"
all_anomalies_path = r"C:\Users\jason\MSciProject\all_anomalies.nc"




# --- Load and Prepare Data (assuming helper functions are correct) --------
observed_temp = xr.open_dataset(observed_path, decode_times=False)
# implicit = load_and_prepare_dataset(implicit_path)
# explicit = load_and_prepare_dataset(explicit_path)
# crank = load_and_prepare_dataset(crank_path)
# semi_implicit = load_and_prepare_dataset(semi_implicit_path)
# chris = load_and_prepare_dataset(chris_path)
all_anomalies = load_and_prepare_dataset(all_anomalies_path)



# --- Extracting the correct DataArray -------------------
# temperature = observed_temp['__xarray_dataarray_variable__']
# implicit = implicit["T_model_anom_implicit"]
# explicit = explicit["T_model_anom_explicit"]
# crank = crank["T_model_anom_crank_nicolson"]
# semi_implicit = semi_implicit["T_model_anom_semi_implicit"]
# chris = chris["ARGO_TEMPERATURE_ANOMALY"]

temperature = observed_temp['__xarray_dataarray_variable__']
implicit = all_anomalies["IMPLICIT"]
explicit = all_anomalies["EXPLICIT"]
semi_implicit = all_anomalies["SEMI_IMPLICIT"]
chris = all_anomalies["CHRIS_CAPPED_EXPONENT"]
chris_mean_k = all_anomalies["CHRIS_MEAN_K"]



#---- Defining the Anomaly Temperature Dataset ----------------------
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

#%%

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

temp_masked = consistent_mask(chris_mean_k, temperature_anomaly)
error_chris_mean = calculate_error(chris_mean_k, temp_masked)
error_chris_mean_flat = error_chris_mean.values.flatten()
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.histplot(error_chris_mean_flat[~np.isnan(error_chris_mean_flat)], bins=1000)
ax.set_xlabel("Error (K)")
ax.set_ylabel("Error Freq")
ax.set_title("Chris Mean Scheme Temporal Error Distribution")
# ax.set_xlim(-19e30,19e30)
ax.axvline(min(error_chris_mean_flat), color='red', linestyle='--', label=f'Max Negative Error = {max(error_chris_mean_flat):.4f})')
ax.axvline(max(error_chris_mean_flat), color='red', linestyle='--', label=f'Max Positive Error = {max(error_chris_mean_flat):.4f})')
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
fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.kdeplot(error_imp_flat[~np.isnan(error_imp_flat)], label="Implicit Scheme")
sns.kdeplot(error_simp_flat[~np.isnan(error_simp_flat)], label="Semi-Implicit Scheme")
sns.kdeplot(error_chris_mean_flat[~np.isnan(error_chris_mean_flat)], label="Chris Mean K Scheme")
sns.kdeplot(error_chris_flat[~np.isnan(error_chris_flat)], label="Chris Scheme")
ax.set_xlabel("Error (K)")
ax.set_ylabel("Error Freq")
ax.set_title("Temporal Error Distribution")
# ax.set_xlim(-19e30,19e30)
ax.legend()


plt.show()
#%%


# Previous graph difficult to interpret
# Consulted ChatGPT to help with log scale plotting
plt.figure(figsize=(10, 6))

for arr, label in [
    (error_imp_flat[~np.isnan(error_imp_flat)], "Implicit Scheme"),
    (error_simp_flat[~np.isnan(error_simp_flat)], "Semi-Implicit Scheme"),
    (error_chris_mean_flat[~np.isnan(error_chris_mean_flat)], "Chris Mean K Scheme"),
    (error_chris_flat[~np.isnan(error_chris_flat)], "Chris Scheme")
]:
    sns.kdeplot(arr, bw_adjust=1.0, label=label)

plt.xlim(-10, 10)        
plt.yscale("log")      
plt.ylim(1e-4, 1e2)
plt.title("Temporal Error Distribution (zoomed, log-scale)")
plt.xlabel("Error (K)")
plt.ylabel("PDF (log scale)")
plt.legend()
plt.show()
#%%
# 1. Standardise (z-score) each dataset
implicit_z      = zscore(error_imp_flat, nan_policy='omit')
semi_implicit_z = zscore(error_simp_flat, nan_policy='omit')
chris_meanK_z   = zscore(error_chris_mean_flat, nan_policy='omit')
chris_z         = zscore(error_chris_flat, nan_policy='omit')

plt.figure(figsize=(10, 5))

sns.kdeplot(implicit_z,      label='Implicit Scheme',        bw_adjust=3)
sns.kdeplot(semi_implicit_z, label='Semi-Implicit Scheme',   bw_adjust=3)
sns.kdeplot(chris_meanK_z,   label='Chris Mean K Scheme',    bw_adjust=3)
sns.kdeplot(chris_z,         label='Chris Scheme',           bw_adjust=3)

plt.xlim(-4, 4)                    
plt.xlabel('Standardised Error (z-score)')
plt.ylabel('Density')
plt.title('Standardised Temporal Error Distributions')
plt.legend()
plt.tight_layout()
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
ds_chris_mean = error_chris_mean
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
#anim.save(r"C:\Users\jason\MSciProject\explicit_animation.gif", writer="pillow", fps=4)


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
#anim.save(r"C:\Users\jason\MSciProject\implicit_animation.gif", writer="pillow", fps=4)

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
    title.set_text(f"Semi-Implicit Scheme, Month {moy} in Year {year}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=179, interval=500, blit=False, repeat=True
)


_ = anim

HTML(anim.to_jshtml())
#anim.save(r"C:\Users\jason\MSciProject\semi_implicit_animation.gif", writer="pillow", fps=4)
#%%
#Chris Mean Scheme Spatial Mean Error

v = np.nanmax(np.abs(ds_chris_mean))
vmin, vmax = -v,v

# Coords
lats = ds_chris_mean["LATITUDE"].values
lons = ds_chris_mean["LONGITUDE"].values

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(
    ds_chris_mean.isel(TIME=0).values,
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
    arr = ds_chris_mean.isel(TIME=frame_idx).values
    im.set_data(arr)
    month_num = int(ds_chris_mean["TIME"].values[frame_idx])
    year = 2004 + (month_num - 1)//12
    moy = ((month_num -1) %12) + 1
    title.set_text(f"Chris Mean Scheme, Month {moy} in Year {year}")
    return [im, title]

anim = FuncAnimation(
    fig, update, frames=179, interval=500, blit=False, repeat=True
)


_ = anim

HTML(anim.to_jshtml())
#anim.save(r"C:\Users\jason\MSciProject\crank_animation.gif", writer="pillow", fps=4)

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
#anim.save(r"C:\Users\jason\MSciProject\chris_animation.gif", writer="pillow", fps=4)
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
