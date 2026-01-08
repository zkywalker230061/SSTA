#%%
# --- 1. Running Implicit Scheme ---------------------------------- 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from chris_utils import make_movie, get_eof_with_nan_consideration
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.stats import kurtosis, skew

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = False
CLEAN_CHRIS_PREV_CUR = False        # only really useful when entrainment is turned on

observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = r"C:\Users\jason\MSciProject\heat_flux_interpolated_all_contributions.nc"
# HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = r"C:\Users\jason\MSciProject\Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-(2004-2018).nc"
ENTRAINMENT_VEL_DATA_PATH = r"C:\Users\jason\MSciProject\Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"
H_BAR_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\t_sub.nc"
rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 10

temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
observed_temp_ds = xr.open_dataset(observed_path, decode_times=False)

heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + \
                                heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
ekman_anomaly_da = ekman_anomaly_ds['Q_Ek_anom']
ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD_PRESSURE"]

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
t_sub_da = t_sub_ds["T_sub_ANOMALY"]

entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']


def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60


delta_t = month_to_second(1)

# initialise lists for temperature anomalies for implicit scheme
implicit_model_anomalies = []

# initialise lists for entrainment fluxes for the implicit scheme
entrainment_fluxes_implicit = []

added_baseline = False
for month in heat_flux_anomaly_ds.TIME.values:
    # find the previous and current month from 1 to 12 to access the monthly-averaged data (hbar, entrainment vel.)
    prev_month = month - 1
    month_in_year = int((month + 0.5) % 12)
    if month_in_year == 0:
        month_in_year = 12
    prev_month_in_year = month_in_year - 1
    if prev_month_in_year == 0:
        prev_month_in_year = 12

    if not added_baseline:  # just adds the baseline of a whole bunch of zero
        base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY'] - \
               temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
        base = base.expand_dims(TIME=[month])
        implicit_model_anomalies.append(base)
        added_baseline = True

    else:
        # store previous readings Tm(n-1)
        prev_implicit_k_tm_anom = implicit_model_anomalies[-1].isel(TIME=-1)

        # get previous data
        prev_tsub_anom = t_sub_da.sel(TIME=prev_month)
        prev_heat_flux_anom = surface_flux_da.sel(TIME=prev_month)
        prev_ekman_anom = ekman_anomaly_da.sel(TIME=prev_month)
        prev_entrainment_vel = entrainment_vel_da.sel(MONTH=prev_month_in_year)
        prev_hbar = hbar_da.sel(MONTH=prev_month_in_year)

        # get current data
        cur_tsub_anom = t_sub_da.sel(TIME=month)
        cur_heat_flux_anom = surface_flux_da.sel(TIME=month)
        cur_ekman_anom = ekman_anomaly_da.sel(TIME=month)
        cur_entrainment_vel = entrainment_vel_da.sel(MONTH=month_in_year)
        cur_hbar = hbar_da.sel(MONTH=month_in_year)

        # generate the right dataset depending on whether surface flux and/or Ekman terms are desired
        if INCLUDE_SURFACE and INCLUDE_EKMAN:
            cur_surf_ek = cur_heat_flux_anom + cur_ekman_anom
            prev_surf_ek = prev_heat_flux_anom + prev_ekman_anom

        elif INCLUDE_SURFACE:
            cur_surf_ek = cur_heat_flux_anom
            prev_surf_ek = prev_heat_flux_anom

        elif INCLUDE_EKMAN:
            cur_surf_ek = cur_ekman_anom
            prev_surf_ek = prev_ekman_anom

        else:       # just a way to get a zero dataset
            cur_surf_ek = cur_ekman_anom - cur_ekman_anom
            prev_surf_ek = prev_ekman_anom - prev_ekman_anom

        if INCLUDE_ENTRAINMENT:
            cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar) + cur_entrainment_vel / cur_hbar * cur_tsub_anom
            cur_a = cur_entrainment_vel / cur_hbar + gamma_0 / (rho_0 * c_0 * cur_hbar)
            cur_k = (gamma_0 / (rho_0 * c_0) + cur_entrainment_vel) / cur_hbar

            prev_b = prev_surf_ek / (rho_0 * c_0 * prev_hbar) + prev_entrainment_vel / prev_hbar * prev_tsub_anom
            prev_a = prev_entrainment_vel / prev_hbar + gamma_0 / (rho_0 * c_0 * prev_hbar)
            prev_k = (gamma_0 / (rho_0 * c_0) + prev_entrainment_vel) / prev_hbar
        else:
            cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar)
            cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)
            cur_k = cur_a

            prev_b = prev_surf_ek / (rho_0 * c_0 * prev_hbar)
            prev_a = gamma_0 / (rho_0 * c_0 * prev_hbar)
            prev_k = prev_a

        

        # update anomalies
        cur_implicit_k_tm_anom = (prev_implicit_k_tm_anom + delta_t * cur_b) / (1 + delta_t * cur_a)

        # reformat and save model
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.expand_dims(TIME=[month])
        implicit_model_anomalies.append(cur_implicit_k_tm_anom)

        # get entrainment flux components; for categorising each component
        if INCLUDE_ENTRAINMENT:
            entrainment_flux_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_implicit_k_tm_anom)
            entrainment_fluxes_implicit.append(entrainment_flux_implicit)


# concatenate anomalies into a ds
implicit_model_anomaly_ds = xr.concat(implicit_model_anomalies, 'TIME')

# rename all models
implicit_model_anomaly_ds = implicit_model_anomaly_ds.rename("IMPLICIT")

# # clean up prev_cur model
# if CLEAN_CHRIS_PREV_CUR:
#     all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = all_anomalies_ds["CHRIS_PREV_CUR"].where((all_anomalies_ds["CHRIS_PREV_CUR"] > -10) & (all_anomalies_ds["CHRIS_PREV_CUR"] < 10))
#     n_modes = 20
#     monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
#     map_mask = temperature_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5)
#     eof_ds, variance, PCs = get_eof_with_nan_consideration(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"], map_mask, modes=n_modes, monthly_mean_ds=None, tolerance=1e-4)
#     all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = eof_ds.rename("CHRIS_PREV_CUR_CLEAN")

# save
# all_anomalies_ds.to_netcdf("../datasets/all_anomalies_10.nc")

# format entrainment flux datasets
if INCLUDE_ENTRAINMENT:
    entrainment_flux_implicit_ds = xr.concat(entrainment_fluxes_implicit, 'TIME')
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.rename("ENTRAINMENT_FLUX_IMPLICIT_ANOMALY")


# merge the relevant fluxes into a single dataset
flux_components_to_merge = []
if INCLUDE_SURFACE:
    surface_flux_da = surface_flux_da.rename("SURFACE_FLUX_ANOMALY")
    flux_components_to_merge.append(surface_flux_da)
if INCLUDE_EKMAN:
    ekman_anomaly_da = ekman_anomaly_da.rename("EKMAN_FLUX_ANOMALY")
    flux_components_to_merge.append(ekman_anomaly_da)
if INCLUDE_ENTRAINMENT:
    flux_components_to_merge.append(entrainment_flux_implicit_ds)

flux_components_ds = xr.merge(flux_components_to_merge)

#%%
#--- 2. Prepare Observed Temperature Anomaly --------------------

# Extract Variables
observed_temperature_monthly_average = get_monthly_mean(observed_temp_ds['__xarray_dataarray_variable__'])
observed_temperature_anomaly = get_anomaly(observed_temp_ds, '__xarray_dataarray_variable__', observed_temperature_monthly_average)
observed_temperature_anomaly = observed_temperature_anomaly['__xarray_dataarray_variable___ANOMALY']

# Prepare Implicit Scheme Monthly Average At Each Grid Point
imp_temp_monthly_average = get_monthly_mean(implicit_model_anomaly_ds)
#%%
#--- 3. Helper Functions --------------------

def get_phase_map_seasonal_maximum(data_monthly_mean, obs_monthly_mean):
    # Creating Mask
    mask_data = data_monthly_mean.notnull().all(dim="MONTH")
    mask_obs = obs_monthly_mean.notnull().all(dim="MONTH")

    # Replacing NaNs with 0
    data_monthly_mean = data_monthly_mean.fillna(-999)
    obs_monthly_mean = obs_monthly_mean.fillna(-999)

    data_monthly_mean_max = data_monthly_mean.argmax(dim="MONTH")
    obs_monthly_mean_max = obs_monthly_mean.argmax(dim="MONTH")

    phase_map = data_monthly_mean_max - obs_monthly_mean_max
    
    corrected_phase_map = ((phase_map + 6) % 12) - 6

    corrected_phase_map = corrected_phase_map.where(mask_obs)

    return corrected_phase_map

def get_phase_map_seasonal_maximum(data_monthly_mean, obs_monthly_mean):
    # Creating Mask
    mask_data = data_monthly_mean.notnull().all(dim="MONTH")
    mask_obs = obs_monthly_mean.notnull().all(dim="MONTH")

    # Replacing NaNs with 0
    data_monthly_mean = data_monthly_mean.fillna(-999)
    obs_monthly_mean = obs_monthly_mean.fillna(-999)

    data_monthly_mean_max = data_monthly_mean.argmax(dim="MONTH")
    obs_monthly_mean_max = obs_monthly_mean.argmax(dim="MONTH")

    phase_map = data_monthly_mean_max - obs_monthly_mean_max
    
    corrected_phase_map = ((phase_map + 6) % 12) - 6

    corrected_phase_map = corrected_phase_map.where(mask_obs)

    return corrected_phase_map

def get_phase_map_seasonal_minimum(data_monthly_mean, obs_monthly_mean):
    # Creating Mask
    mask_data = data_monthly_mean.notnull().all(dim="MONTH")
    mask_obs = obs_monthly_mean.notnull().all(dim="MONTH")

    # Replacing NaNs with 0
    data_monthly_mean = data_monthly_mean.fillna(999)
    obs_monthly_mean = obs_monthly_mean.fillna(999)

    data_monthly_mean_max = data_monthly_mean.argmin(dim="MONTH")
    obs_monthly_mean_max = obs_monthly_mean.argmin(dim="MONTH")

    phase_map = data_monthly_mean_max - obs_monthly_mean_max
    
    corrected_phase_map = ((phase_map + 6) % 12) - 6

    corrected_phase_map = corrected_phase_map.where(mask_obs)

    return corrected_phase_map

def get_amplitude(data_monthly_mean, obs_monthly_mean):
    # Creating Mask
    mask_obs = obs_monthly_mean.notnull().all(dim="MONTH")

    data_monthly_mean_max = data_monthly_mean.max(dim="MONTH")
    data_monthly_mean_min = data_monthly_mean.min(dim="MONTH")

    obs_monthly_mean_max = obs_monthly_mean.max(dim="MONTH")
    obs_monthly_mean_min = obs_monthly_mean.min(dim="MONTH")

    # Getting Amplitudes
    amplitude_data = (data_monthly_mean_max - data_monthly_mean_min) / 2
    amplitude_obs = (obs_monthly_mean_max - obs_monthly_mean_min) / 2

    amplitude_map = np.log10(amplitude_data / amplitude_obs)

    amplitude_map = amplitude_map.where(mask_obs)

    return amplitude_map

#--- 4. Plot Phase Map --------------------
phase_map_maximum = get_phase_map_seasonal_maximum(imp_temp_monthly_average, observed_temperature_monthly_average)
print(phase_map_maximum)
phase_map_minimum = get_phase_map_seasonal_minimum(imp_temp_monthly_average, observed_temperature_monthly_average)
print(phase_map_minimum)
amplitudes = get_amplitude(imp_temp_monthly_average, observed_temperature_monthly_average)
print(amplitudes)

#%%

# Plotting Seasonal Maximum
fig, axes = plt.subplots(1, 1, figsize=(8,5))
scheme_name = "Implicit"
# Plotting
# ax = plt.subplot(3, 2, i + 1)
phase_map_maximum.plot(ax=axes, cmap='RdBu_r', cbar_kwargs={'label': 'Phase'}, vmin = -6, vmax = 6)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Phase Map Seasonal Maximum')
max_phase = phase_map_maximum.max().item()
print(scheme_name, max_phase)
plt.tight_layout()
# fig.text(
#     0.99, 0.01,
#     f"Gamma = {gamma_0}\n"
#     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
#     f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
#     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
#     ha='right', va='bottom', fontsize=18
# )
plt.show()


# Plotting Seasonal Minimum
fig, axes = plt.subplots(1, 1, figsize=(8,5))
scheme_name = "Implicit"
# Plotting
# ax = plt.subplot(3, 2, i + 1)
phase_map_minimum.plot(ax=axes, cmap='RdBu_r', cbar_kwargs={'label': 'Phase'}, vmin = -6, vmax = 6)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Phase Map Seasonal Minimum')
min_phase = phase_map_minimum.max().item()
print(scheme_name, min_phase)
plt.tight_layout()
# fig.text(
#     0.99, 0.01,
#     f"Gamma = {gamma_0}\n"
#     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
#     f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
#     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
#     ha='right', va='bottom', fontsize=18
# )
plt.show()

# Plotting Amplitudes
fig, axes = plt.subplots(1, 1, figsize=(8,5))
scheme_name = "Implicit"
# Plotting
# ax = plt.subplot(3, 2, i + 1)
amplitudes.plot(ax=axes, cmap='viridis', cbar_kwargs={'label': 'Amplitude'}, vmin = -7, vmax = -1)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Amplitude Plot')
max_amp = amplitudes.max().item()
print(scheme_name, max_amp)
min_amp = amplitudes.min().item()
print(scheme_name, min_amp)
plt.tight_layout()
# fig.text(
#     0.99, 0.01,
#     f"Gamma = {gamma_0}\n"
#     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
#     f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
#     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
#     ha='right', va='bottom', fontsize=18
# )
plt.show()
# %%
