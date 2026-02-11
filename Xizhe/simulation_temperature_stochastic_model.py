"""
Simulate Sea Surface Temperature Anomalies (SSTA).

Chengyun Zhu
2026-1-18
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils_read_nc import load_and_prepare_dataset
from chris_utils import get_monthly_mean, get_anomaly

# matplotlib.use('TkAgg')

# --- CONFIGURATION -----------------------------------------------------------
SURFACE = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = True
LAMBDA_A = 15

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # Average seconds in a month

# --- FILE PATHS --------------------------------------------------------------
observed_path_argo = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
observed_path_Reynold = "/Users/julia/Desktop/SSTA/datasets/Reynold_sst_anomalies-(2004-2018).nc"

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Anomaly_Full_Datasets.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Entrainment_Velocity-(2004-2018).nc"

H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/hbar.nc"
NEW_H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_hbar.nc"

T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/t_sub.nc"
NEW_T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_T_sub_prime.nc"

GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated_2.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"

USE_DOWNLOADED_SSH = False
USE_NEW_H_BAR_NEW_T_SUB = True 

# --- DATA PREPARATION --------------------------------------------------------

if USE_NEW_H_BAR_NEW_T_SUB:
    # New h bar
    hbar_ds = xr.open_dataset(NEW_H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    # New t sub
    t_sub_ds = xr.open_dataset(NEW_T_SUB_DATA_PATH, decode_times=False)
    t_sub_da = t_sub_ds["ANOMALY_SUB_TEMPERATURE"]
        
    # Observed Data (Tm) using new h
    observed_temp_ds_full = xr.open_dataset(observed_path_argo, decode_times=False)
    observed_temp_ds = observed_temp_ds_full["UPDATED_MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    obs_temp_anom = get_anomaly(observed_temp_ds_full, "UPDATED_MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly = obs_temp_anom["UPDATED_MIXED_LAYER_TEMP_ANOMALY"]

    # Ekman Anomaly using new h
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds['UPDATED_TEMP_EKMAN_ANOM']
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

else:
    hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
    t_sub_da = t_sub_ds["SUB_TEMPERATURE"]

    t_sub_mean = get_monthly_mean(t_sub_da)
    t_sub_danom = get_anomaly(t_sub_ds, "SUB_TEMPERATURE", t_sub_mean)
    t_sub_danom = t_sub_danom["SUB_TEMPERATURE_ANOMALY"]
    t_sub_da = t_sub_danom

    # Observed Data (Tm) using old h
    observed_temp_ds_full = xr.open_dataset(observed_path_argo, decode_times=False)
    observed_temp_ds = observed_temp_ds_full["MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_full, "MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly = observed_temperature_anomaly["MIXED_LAYER_TEMP_ANOMALY"]

    # Ekman Anomaly using old h
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds["TEMP_EKMAN_ANOM"]
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)


# Unchanged Parameters for the simulation 
temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)

observed_temp_ds_reynold = xr.open_dataset(observed_path_Reynold, decode_times=False)
observed_temperature_anomaly_reynold = observed_temp_ds_reynold['anom']

# Surface Heat Flux 
heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + \
                                heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

# Entrainment Velocity
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

# Geostrophic Term
geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)

# --- SWITCHES AND PRE-PROCESSING ---------------------------------------------

if not SURFACE:
    surface_flux_da = surface_flux_da - surface_flux_da

ekman_anomaly_da = ekman_anomaly_da.where((ekman_anomaly_da['LATITUDE'] > 5) | (ekman_anomaly_da['LATITUDE'] < -5), 0)
if not EKMAN:
    ekman_anomaly_da = ekman_anomaly_da - ekman_anomaly_da

geostrophic_anomaly_da = geostrophic_anomaly_da.where((geostrophic_anomaly_da['LATITUDE'] > 5) | (geostrophic_anomaly_da['LATITUDE'] < -5), 0)
if not GEOSTROPHIC:
    geostrophic_anomaly_da = geostrophic_anomaly_da - geostrophic_anomaly_da

if not ENTRAINMENT:
    entrainment_vel_da = entrainment_vel_da - entrainment_vel_da

# --- EXPANDING TIME DIMENSIONS -----------------------------------------------
# 1. Expand Entrainment Velocity
# Assuming 15 years of data (12 months * 15 = 180 time steps)
entrainment_vel_da = xr.concat([entrainment_vel_da] * 15, dim='MONTH').reset_coords(drop=True)
entrainment_vel_da = entrainment_vel_da.rename({'MONTH': 'TIME'})
entrainment_vel_da['TIME'] = observed_temperature_anomaly.TIME

# 2. Expand Mixed Layer Depth (hbar)
hbar_da = xr.concat([hbar_da] * 15, dim='MONTH').reset_coords(drop=True)
hbar_da = hbar_da.rename({'MONTH': 'TIME'})
hbar_da['TIME'] = observed_temperature_anomaly.TIME

# --- SIMULATION EQUATION TERMS -----------------------------------------------

# Now that hbar is expanded, we can calculate the forcing term
dobserved_temperature_anomaly_dt = (
    surface_flux_da
    + ekman_anomaly_da
    + geostrophic_anomaly_da
) / (RHO_O * C_O * hbar_da)

_lambda = LAMBDA_A / (RHO_O * C_O * hbar_da) + entrainment_vel_da / hbar_da

observed_temperature_anomaly_simulated_list = []

# --- RUN SIMULATION ----------------------------------------------------------
# Analytical Exponential Scheme
print("Running Simulation...")

for month_num in observed_temperature_anomaly['TIME'].values:
    if month_num == 0.5:
        # Initial condition
        observed_temperature_anomaly_simulated_da = observed_temperature_anomaly.sel(TIME=month_num)
        temp = observed_temperature_anomaly_simulated_da
    else:
        # Recursive step
        # Term 1: Decay of previous state
        decay_term = temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
        
        # Term 2: Forcing (Entrainment deepening + Heat Fluxes)
        forcing_term = (
             t_sub_da.sel(TIME=month_num-1) * np.log(hbar_da.sel(TIME=month_num)/hbar_da.sel(TIME=month_num-1)) / SECONDS_MONTH
             + dobserved_temperature_anomaly_dt.sel(TIME=month_num-1)
        )
        
        # Apply solution
        observed_temperature_anomaly_simulated_da = (
            decay_term 
            + (forcing_term / _lambda.sel(TIME=month_num-1)) * (1 - np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH))
        )
        
        temp = observed_temperature_anomaly_simulated_da
        
    # --- FIX IS HERE: Drop 'MONTH' before appending ---
    observed_temperature_anomaly_simulated_da = observed_temperature_anomaly_simulated_da.drop_vars('MONTH', errors='ignore')
    
    observed_temperature_anomaly_simulated_da = observed_temperature_anomaly_simulated_da.expand_dims(TIME=[month_num])
    observed_temperature_anomaly_simulated_list.append(observed_temperature_anomaly_simulated_da)

# Now concat should work without errors
observed_temperature_anomaly_simulated = xr.concat(
    observed_temperature_anomaly_simulated_list,
    dim="TIME",
    coords="minimal"
)

observed_temperature_anomaly_simulated.name = 'UPDATED_MIXED_LAYER_TEMP_ANOMALY'
observed_temperature_anomaly_simulated_ds = observed_temperature_anomaly_simulated.to_dataset()
observed_temperature_anomaly_simulated_monthly_mean = get_monthly_mean(
    observed_temperature_anomaly_simulated_ds['UPDATED_MIXED_LAYER_TEMP_ANOMALY']
)
anomaly_result_ds = get_anomaly(
    observed_temperature_anomaly_simulated_ds, 
    'UPDATED_MIXED_LAYER_TEMP_ANOMALY', 
    observed_temperature_anomaly_simulated_monthly_mean
)

observed_temperature_anomaly_simulated = anomaly_result_ds['UPDATED_MIXED_LAYER_TEMP_ANOMALY_ANOMALY']
observed_temperature_anomaly_simulated = observed_temperature_anomaly_simulated.drop_vars('MONTH', errors='ignore')

# --- CALCULATE MISSING Q_ENTRAINMENT -----------------------------------------
# We calculate this AFTER simulation because it depends on the simulated Temperature Anomaly
# Formula: Q_ent = rho * Cp * w_e * (T_sub - T_m_simulated)
q_entrainment = RHO_O * C_O * entrainment_vel_da * (t_sub_da - observed_temperature_anomaly_simulated)

if not ENTRAINMENT:
    q_entrainment = q_entrainment - q_entrainment

# --- STATISTICS & PLOTTING ---------------------------------------------------

print("simulated (max, min, mean, abs mean):")
print(observed_temperature_anomaly_simulated.max().item(), observed_temperature_anomaly_simulated.min().item())
print(observed_temperature_anomaly_simulated.mean().item())
print(abs(observed_temperature_anomaly_simulated).mean().item())

print("observed (max, min, mean, abs mean):")
print(observed_temperature_anomaly_reynold.max().item(), observed_temperature_anomaly_reynold.min().item())
print(observed_temperature_anomaly_reynold.mean().item())
print(abs(observed_temperature_anomaly_reynold).mean().item())
print('-----')

rms_difference = np.sqrt(((observed_temperature_anomaly_reynold - observed_temperature_anomaly_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((observed_temperature_anomaly_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((observed_temperature_anomaly_reynold ** 2).mean(dim=['TIME']))

print("rms simulated", rms_difference.mean().item())
rms_simulated.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.title("RMS Simulated")
plt.show()

print("rms observed", rms_observed.mean().item())
rms_observed.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.title("RMS Observed (Reynolds)")
plt.show()

rmse = rms_difference / rms_observed
print("normalised rmse", rmse.mean().item())
rmse.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.title("Normalized RMSE")
plt.show()

corr = xr.corr(observed_temperature_anomaly_reynold, observed_temperature_anomaly_simulated, dim='TIME')
print("corr", corr.mean().item())
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.title("Correlation")
plt.show()

# --- CONTRIBUTION PLOT -------------------------------------------------------

surface_fraction = []
entrainment_fraction = []
ekman_fraction = []
geo_fraction = []
surface_contribution = []
entrainment_contribution = []
ekman_contribution = []
geo_contribution = []

# Recalculate total with the newly computed q_entrainment
total = abs(surface_flux_da) + abs(q_entrainment) + abs(ekman_anomaly_da) + abs(geostrophic_anomaly_da)

for month_num in observed_temperature_anomaly['TIME'].values:
    # Fractions (Absolute contributions relative to total absolute flux)
    surface_fraction.append(
        (abs(surface_flux_da.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    entrainment_fraction.append(
        (abs(q_entrainment.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    ekman_fraction.append(
        (abs(ekman_anomaly_da.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    geo_fraction.append(
        (abs(geostrophic_anomaly_da.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    
    # Raw Contributions (Means)
    surface_contribution.append(surface_flux_da.sel(TIME=month_num).mean().item())
    entrainment_contribution.append(q_entrainment.sel(TIME=month_num).mean().item())
    ekman_contribution.append(ekman_anomaly_da.sel(TIME=month_num).mean().item())
    geo_contribution.append(geostrophic_anomaly_da.sel(TIME=month_num).mean().item())

plt.figure(figsize=(10, 6))
plt.plot(observed_temperature_anomaly['TIME'], surface_fraction, label='Surface Heat Flux')
plt.plot(observed_temperature_anomaly['TIME'], entrainment_fraction, label='Entrainment')
plt.plot(observed_temperature_anomaly['TIME'], ekman_fraction, label='Ekman Transport')
plt.plot(observed_temperature_anomaly['TIME'], geo_fraction, label='Geostrophic Advection')

plt.title("Fractional Contribution of Heat Flux Terms")
plt.xlabel("Time (Months)")
plt.ylabel("Fraction of Total Flux Magnitude")
plt.legend()
plt.grid(True)
plt.show()