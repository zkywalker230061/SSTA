#%%
# --- 1. Running Implicit Scheme ---------------------------------- 
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import pandas as pd
from chris_utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, coriolis_parameter
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.stats import kurtosis, skew, pearsonr, t

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC_MEAN = True
INCLUDE_GEOSTROPHIC_ANOM = True
CLEAN_CHRIS_PREV_CUR = False        # only really useful when entrainment is turned on

observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Anomaly_Full_Datasets.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"
# H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/hbar.nc"
NEW_H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_hbar.nc"

T_SUB_DATA_PATH = "datasets/New_Entrainment/Tsub_Max_Gradient_Method_h.nc" #######--------------check
NEW_T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_T_sub_prime.nc"

GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated_2.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"

USE_DOWNLOADED_SSH = False
USE_NEW_H_BAR_NEW_T_SUB = False 

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15
g = 9.81
f = 1 

if USE_NEW_H_BAR_NEW_T_SUB:
    # New h bar
    hbar_ds = xr.open_dataset(NEW_H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    # New t sub
    t_sub_ds = xr.open_dataset(NEW_T_SUB_DATA_PATH, decode_times=False)
    t_sub_da = t_sub_ds["ANOMALY_SUB_TEMPERATURE"]
        
    # Observed Data (Tm) using new h
    observed_temp_ds_full = xr.open_dataset(observed_path, decode_times=False)
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
    t_sub_anom = get_anomaly(t_sub_ds, "SUB_TEMPERATURE", t_sub_mean)
    t_sub_anom = t_sub_anom["SUB_TEMPERATURE_ANOMALY"]

    t_sub_da = t_sub_anom

    # Observed Data (Tm) using new "old" h
    observed_temp_ds_full = xr.open_dataset(observed_path, decode_times=False)
    observed_temp_ds = observed_temp_ds_full["MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_full, "MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly = observed_temperature_anomaly["MIXED_LAYER_TEMP_ANOMALY"]

    # Ekman Anomaly using new "old" h
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds["TEMP_EKMAN_ANOM"]
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

    # ekman_anomaly_da_centred_mean = get_monthly_mean(ekman_anomaly_ds["Q_Ek_anom"])
    # ekman_anomaly_da_final = get_anomaly(ekman_anomaly_ds, "Q_Ek_anom", ekman_anomaly_da_centred_mean)
    # ekman_anomaly_da_final = ekman_anomaly_da_final["Q_Ek_anom_ANOMALY"]

    # print(f"Original Mean Ekman: {ekman_anomaly_ds['Q_Ek_anom'].mean().values}")
    # print(f"Centered Mean Ekman: {ekman_anomaly_da_final.mean().values}")



# Unchanged Parameters for the simulation 
temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)

# Surface Heat Flux 
heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + \
                                heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

# surface_flux_da_centred_mean = get_monthly_mean(heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'])
# surface_flux_da_anomaly = get_anomaly(heat_flux_anomaly_ds, 'NET_HEAT_FLUX_ANOMALY', surface_flux_da_centred_mean)
# surface_flux_da_final = heat_flux_anomaly_ds["NET_HEAT_FLUX_ANOMALY_ANOMALY"]

# print(surface_flux_da)
# print(surface_flux_da_final)
# print(f"Original Net Heat Flux Mean: {heat_flux_ds['NET_HEAT_FLUX'].mean().values}")
# print(f"Double-Centered Mean: {surface_flux_da_final.mean().values}")

# Entrainment Velocity
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']
print(entrainment_vel_da)

# Overwrite off-centred anomalies to avoid changing variables in the simulation
# surface_flux_da = surface_flux_da_final
# ekman_anomaly_da = ekman_anomaly_da_final
# ekman_anomaly_da = ekman_anomaly_da.fillna(0)


# Edit Geostrophic terms once I receive files from Chris
geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]



if USE_DOWNLOADED_SSH:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"
    ssh_var_name = "sla"
else:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_calculated_grad.nc"
    ssh_var_name = "ssh"
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)


def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60

delta_t = month_to_second(1)



# initialise lists for temperature anomalies for each model
implicit_model_anomalies = []
# chris_prev_cur_model_anomalies = []
# chris_mean_k_model_anomalies = []
# chris_prev_k_model_anomalies = []
# chris_capped_exponent_model_anomalies = []
# explicit_model_anomalies = []
# semi_implicit_model_anomalies = []

# initialise lists for entrainment fluxes for each model; for categorising each component
entrainment_fluxes_implicit = []
# entrainment_fluxes_prev_cur = []
# entrainment_fluxes_mean_k = []
# entrainment_fluxes_prev_k = []
# entrainment_fluxes_capped_exponent = []
# entrainment_fluxes_explicit = []
# entrainment_fluxes_semi_implicit = []



added_baseline = False
testparam = False
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
        # chris_prev_cur_model_anomalies.append(base)
        # chris_mean_k_model_anomalies.append(base)
        # chris_prev_k_model_anomalies.append(base)
        # chris_capped_exponent_model_anomalies.append(base)
        # explicit_model_anomalies.append(base)
        # semi_implicit_model_anomalies.append(base)
        added_baseline = True

    else:
        # store previous readings Tm(n-1)
        if INCLUDE_GEOSTROPHIC_ANOM:    # then need to take the previous reading "back-propagated" based on current
            prev_implicit_k_tm_anom_at_cur_loc = implicit_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_prev_cur_tm_anom_at_cur_loc = chris_prev_cur_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_mean_k_tm_anom_at_cur_loc = chris_mean_k_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_prev_k_tm_anom_at_cur_loc = chris_prev_k_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_capped_exponent_k_tm_anom_at_cur_loc = chris_capped_exponent_model_anomalies[-1].isel(TIME=-1)
            # prev_explicit_k_tm_anom_at_cur_loc = explicit_model_anomalies[-1].isel(TIME=-1)
            # prev_semi_implicit_k_tm_anom_at_cur_loc = semi_implicit_model_anomalies[-1].isel(TIME=-1)

            f = coriolis_parameter(sea_surface_grad_ds['LATITUDE']).broadcast_like(sea_surface_grad_ds[ssh_var_name]).broadcast_like(sea_surface_grad_ds[ssh_var_name + '_anomaly_grad_long'])  # broadcasting based on Jason/Julia's usage
            alpha = g / f.sel(TIME=month) * sea_surface_grad_ds[ssh_var_name + '_anomaly_grad_long'].sel(TIME=month)
            beta = g / f.sel(TIME=month) * sea_surface_grad_ds[ssh_var_name + '_anomaly_grad_lat'].sel(TIME=month)
            back_x = sea_surface_grad_ds['LONGITUDE'] + alpha * month_to_second(1)      # just need a list of long/lat
            back_y = sea_surface_grad_ds['LATITUDE'] - beta * month_to_second(1)        # ss_grad is a useful dummy for that

            # interpolate to the "back-propagated" x and y position, but if that turns out nan (due to coastline), then
            # just use the temperature at current position. BC == "coast buffer"
            prev_implicit_k_tm_anom = prev_implicit_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_implicit_k_tm_anom_at_cur_loc)
            # prev_chris_prev_cur_tm_anom = prev_chris_prev_cur_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_prev_cur_tm_anom_at_cur_loc)
            # prev_chris_mean_k_tm_anom = prev_chris_mean_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_mean_k_tm_anom_at_cur_loc)
            # prev_chris_prev_k_tm_anom = prev_chris_prev_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_prev_k_tm_anom_at_cur_loc)
            # prev_chris_capped_exponent_k_tm_anom = prev_chris_capped_exponent_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_capped_exponent_k_tm_anom_at_cur_loc)
            # prev_explicit_k_tm_anom = prev_explicit_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_explicit_k_tm_anom_at_cur_loc)
            # prev_semi_implicit_k_tm_anom = prev_semi_implicit_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_semi_implicit_k_tm_anom_at_cur_loc)
        else:
            prev_implicit_k_tm_anom = implicit_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_prev_cur_tm_anom = chris_prev_cur_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_mean_k_tm_anom = chris_mean_k_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_prev_k_tm_anom = chris_prev_k_model_anomalies[-1].isel(TIME=-1)
            # prev_chris_capped_exponent_k_tm_anom = chris_capped_exponent_model_anomalies[-1].isel(TIME=-1)
            # prev_explicit_k_tm_anom = explicit_model_anomalies[-1].isel(TIME=-1)
            # prev_semi_implicit_k_tm_anom = semi_implicit_model_anomalies[-1].isel(TIME=-1)

        # get previous data
        prev_tsub_anom = t_sub_da.sel(TIME=prev_month)
        prev_heat_flux_anom = surface_flux_da.sel(TIME=prev_month)
        prev_ekman_anom = ekman_anomaly_da.sel(TIME=prev_month)
        prev_entrainment_vel = entrainment_vel_da.sel(MONTH=prev_month_in_year)
        prev_geo_anom = geostrophic_anomaly_da.sel(TIME=prev_month)
        prev_hbar = hbar_da.sel(MONTH=prev_month_in_year)

        # get current data
        cur_tsub_anom = t_sub_da.sel(TIME=month)
        cur_heat_flux_anom = surface_flux_da.sel(TIME=month)
        cur_ekman_anom = ekman_anomaly_da.sel(TIME=month)
        cur_entrainment_vel = entrainment_vel_da.sel(MONTH=month_in_year)
        cur_geo_anom = geostrophic_anomaly_da.sel(TIME=month)
        cur_hbar = hbar_da.sel(MONTH=month_in_year)

        # generate the right dataset depending on whether surface flux and/or Ekman and/or geostrophic terms are desired
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

        if INCLUDE_GEOSTROPHIC_MEAN:
            cur_surf_ek = cur_surf_ek + cur_geo_anom
            prev_surf_ek = prev_surf_ek + prev_geo_anom

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

        # exponent_prev_cur = prev_k * month_to_second(prev_month) - cur_k * month_to_second(month)
        # exponent_mean_k = -0.5 * (prev_k + cur_k) * delta_t
        # exponent_prev_k = prev_k * month_to_second(prev_month) - prev_k * month_to_second(month)
        # exponent_capped = exponent_prev_cur.where(exponent_prev_cur <= 0, 0)

        # update anomalies
        # if INCLUDE_ENTRAINMENT:
        #     cur_chris_prev_cur_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_prev_cur_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_prev_cur)
        #     cur_chris_mean_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_mean_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_mean_k)
        #     cur_chris_prev_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_prev_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_prev_k)
        #     cur_chris_capped_exponent_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_capped_exponent_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_capped)
        # else:
            # cur_chris_prev_cur_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_prev_cur_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_prev_cur)
            # cur_chris_mean_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_mean_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_mean_k)
            # cur_chris_prev_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_prev_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_prev_k)
            # cur_chris_capped_exponent_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_capped_exponent_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_capped)

        cur_implicit_k_tm_anom = (prev_implicit_k_tm_anom + delta_t * cur_b) / (1 + delta_t * cur_a)
        # cur_explicit_k_tm_anom = prev_explicit_k_tm_anom + delta_t * (prev_b - prev_a * prev_explicit_k_tm_anom)
        # cur_semi_implicit_k_tm_anom = (prev_semi_implicit_k_tm_anom + delta_t * prev_b) / (1 + delta_t * cur_a)

        # reformat and save each model
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.expand_dims(TIME=[month])
        implicit_model_anomalies.append(cur_implicit_k_tm_anom)
        # cur_chris_prev_cur_tm_anom = cur_chris_prev_cur_tm_anom.drop_vars('MONTH', errors='ignore')
        # cur_chris_prev_cur_tm_anom = cur_chris_prev_cur_tm_anom.expand_dims(TIME=[month])
        # chris_prev_cur_model_anomalies.append(cur_chris_prev_cur_tm_anom)
        # cur_chris_mean_k_tm_anom = cur_chris_mean_k_tm_anom.drop_vars('MONTH', errors='ignore')
        # cur_chris_mean_k_tm_anom = cur_chris_mean_k_tm_anom.expand_dims(TIME=[month])
        # chris_mean_k_model_anomalies.append(cur_chris_mean_k_tm_anom)
        # cur_chris_prev_k_tm_anom = cur_chris_prev_k_tm_anom.drop_vars('MONTH', errors='ignore')
        # cur_chris_prev_k_tm_anom = cur_chris_prev_k_tm_anom.expand_dims(TIME=[month])
        # chris_prev_k_model_anomalies.append(cur_chris_prev_k_tm_anom)
        # cur_chris_capped_exponent_k_tm_anom = cur_chris_capped_exponent_k_tm_anom.drop_vars('MONTH', errors='ignore')
        # cur_chris_capped_exponent_k_tm_anom = cur_chris_capped_exponent_k_tm_anom.expand_dims(TIME=[month])
        # chris_capped_exponent_model_anomalies.append(cur_chris_capped_exponent_k_tm_anom)
        # cur_explicit_k_tm_anom = cur_explicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        # cur_explicit_k_tm_anom = cur_explicit_k_tm_anom.expand_dims(TIME=[month])
        # explicit_model_anomalies.append(cur_explicit_k_tm_anom)
        # cur_semi_implicit_k_tm_anom = cur_semi_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        # cur_semi_implicit_k_tm_anom = cur_semi_implicit_k_tm_anom.expand_dims(TIME=[month])
        # semi_implicit_model_anomalies.append(cur_semi_implicit_k_tm_anom)

        # get entrainment flux components; for categorising each component
        if INCLUDE_ENTRAINMENT:
            entrainment_flux_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_implicit_k_tm_anom)
            entrainment_fluxes_implicit.append(entrainment_flux_implicit)
            # entrainment_flux_prev_cur = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_prev_cur_tm_anom)
            # entrainment_fluxes_prev_cur.append(entrainment_flux_prev_cur)
            # entrainment_flux_mean_k = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_mean_k_tm_anom)
            # entrainment_fluxes_mean_k.append(entrainment_flux_mean_k)
            # entrainment_flux_prev_k = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_prev_k_tm_anom)
            # entrainment_fluxes_prev_k.append(entrainment_flux_prev_k)
            # entrainment_flux_capped_exponent = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_capped_exponent_k_tm_anom)
            # entrainment_fluxes_capped_exponent.append(entrainment_flux_capped_exponent)
            # entrainment_flux_explicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_explicit_k_tm_anom)
            # entrainment_fluxes_explicit.append(entrainment_flux_explicit)
            # entrainment_flux_semi_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_semi_implicit_k_tm_anom)
            # entrainment_fluxes_semi_implicit.append(entrainment_flux_semi_implicit)


# concatenate anomalies into a ds
implicit_model_anomaly_ds = xr.concat(implicit_model_anomalies, 'TIME')
# chris_prev_cur_model_anomaly_ds = xr.concat(chris_prev_cur_model_anomalies, 'TIME')
# chris_mean_k_model_anomaly_ds = xr.concat(chris_mean_k_model_anomalies, 'TIME')
# chris_prev_k_model_anomaly_ds = xr.concat(chris_prev_k_model_anomalies, 'TIME')
# chris_capped_exponent_model_anomaly_ds = xr.concat(chris_capped_exponent_model_anomalies, 'TIME')
# explicit_model_anomaly_ds = xr.concat(explicit_model_anomalies, 'TIME')
# semi_implicit_model_anomaly_ds = xr.concat(semi_implicit_model_anomalies, 'TIME')

# rename all models
implicit_model_anomaly_ds = implicit_model_anomaly_ds.rename("IMPLICIT")
# chris_prev_cur_model_anomaly_ds = chris_prev_cur_model_anomaly_ds.rename("CHRIS_PREV_CUR")
# chris_mean_k_model_anomaly_ds = chris_mean_k_model_anomaly_ds.rename("CHRIS_MEAN_K")
# chris_prev_k_model_anomaly_ds = chris_prev_k_model_anomaly_ds.rename("CHRIS_PREV_K")
# chris_capped_exponent_model_anomaly_ds = chris_capped_exponent_model_anomaly_ds.rename("CHRIS_CAPPED_EXPONENT")
# explicit_model_anomaly_ds = explicit_model_anomaly_ds.rename("EXPLICIT")
# semi_implicit_model_anomaly_ds = semi_implicit_model_anomaly_ds.rename("SEMI_IMPLICIT")

# combine to a single ds
all_anomalies_ds = xr.merge([# chris_prev_cur_model_anomaly_ds, chris_mean_k_model_anomaly_ds, chris_prev_k_model_anomaly_ds, chris_capped_exponent_model_anomaly_ds, explicit_model_anomaly_ds, semi_implicit_model_anomaly_ds,
                             implicit_model_anomaly_ds])

# remove whatever seasonal cycle remains
model_names = [#"CHRIS_PREV_CUR", "CHRIS_MEAN_K", "CHRIS_PREV_K", "CHRIS_CAPPED_EXPONENT", "EXPLICIT", "SEMI_IMPLICIT"
               "IMPLICIT"]
for variable_name in model_names:
    monthly_mean = get_monthly_mean(all_anomalies_ds[variable_name])
    all_anomalies_ds[variable_name] = get_anomaly(all_anomalies_ds, variable_name, monthly_mean)[variable_name + "_ANOMALY"]
    all_anomalies_ds = all_anomalies_ds.drop_vars(variable_name + "_ANOMALY")

# clean up prev_cur model
# if CLEAN_CHRIS_PREV_CUR:
#     all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = all_anomalies_ds["CHRIS_PREV_CUR"].where((all_anomalies_ds["CHRIS_PREV_CUR"] > -10) & (all_anomalies_ds["CHRIS_PREV_CUR"] < 10))
#     n_modes = 20
#     monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
#     map_mask = temperature_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5)
#     eof_ds, variance, PCs, EOFs = get_eof_with_nan_consideration(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"], map_mask, modes=n_modes, monthly_mean_ds=None, tolerance=1e-2)
#     all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = eof_ds.rename("CHRIS_PREV_CUR_CLEAN")
#     chris_prev_cur_clean_monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
#     all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = get_anomaly(all_anomalies_ds, "CHRIS_PREV_CUR_CLEAN", chris_prev_cur_clean_monthly_mean)["CHRIS_PREV_CUR_CLEAN_ANOMALY"]
#     all_anomalies_ds = all_anomalies_ds.drop_vars("CHRIS_PREV_CUR_CLEAN_ANOMALY")

# save files
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

flux_components_ds = xr.merge(flux_components_to_merge, join='inner')
implicit_model_anomaly_ds = all_anomalies_ds["IMPLICIT"]





import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def plot_research_subplots(locations, model_ds, obs_ds, fluxes_ds, geo_da, hbar_da, dt_seconds):
    """
    Generates a 2x2 publication-ready multi-panel figure for 4 locations.
    """
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 12), dpi=600)
    axes = axes.flatten()  # Flatten to easily iterate over them
    
    # Placeholders for our unified legend
    legend_lines = []
    legend_labels = []

    for idx, loc in enumerate(locations):
        ax_ts = axes[idx]
        # ax_ts_twin = ax_ts.twinx()
        
        lon, lat, loc_name = loc["lon"], loc["lat"], loc["name"]

        try:
            # --- A. DATA RETRIEVAL ---
            model_ts = model_ds.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            obs_ts = obs_ds.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            
            found_lat = float(model_ts.LATITUDE)
            found_lon = float(model_ts.LONGITUDE)

            # Extract fluxes 
            AirSea = fluxes_ds.get("SURFACE_FLUX_ANOMALY")
            ekman = fluxes_ds.get("EKMAN_FLUX_ANOMALY")
            entrain = fluxes_ds.get("ENTRAINMENT_FLUX_IMPLICIT_ANOMALY")
            
            if AirSea is not None: AirSea = AirSea.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            if ekman is not None: ekman = ekman.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            if entrain is not None: entrain = entrain.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            
            geo = geo_da.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest') if geo_da is not None else None

            # Map TIME to MONTH (1-12) for hbar
            months = ((model_ts.TIME.values + 0.5) % 12).astype(int)
            months[months == 0] = 12
            month_da = xr.DataArray(months, dims="TIME", coords={"TIME": model_ts.TIME})
            
            loc_hbar = hbar_da.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest').sel(MONTH=month_da)
            
            # --- B. CALCULATE BUDGET TERMS (°C / month) ---
            rho_0, c_0 = 1025.0, 4100.0
            rho_c_h = rho_0 * c_0 * loc_hbar
            
            term_AirSea = (AirSea / rho_c_h) * dt_seconds if AirSea is not None else None
            term_ekman = (ekman / rho_c_h) * dt_seconds if ekman is not None else None
            term_ent = (entrain / rho_c_h) * dt_seconds if entrain is not None else None
            term_geo = (geo / rho_c_h) * dt_seconds if geo is not None else None
            
            # --- C. PLOTTING ---
            lines_twin = []
            # if term_AirSea is not None:
            #     l1, = ax_ts_twin.plot(term_AirSea.TIME, term_AirSea, color='goldenrod', alpha=0.5, linewidth=1.5, label='AirSea')
            #     lines_twin.append(l1)
            # if term_ekman is not None:
            #     l2, = ax_ts_twin.plot(term_ekman.TIME, term_ekman, color='green', alpha=0.3, linewidth=1.5, label='Ekman')
            #     lines_twin.append(l2)
            # if term_ent is not None:
            #     l3, = ax_ts_twin.plot(term_ent.TIME, term_ent, color='purple', alpha=0.3, linewidth=1.5, label='Entrain')
            #     lines_twin.append(l3)
            # if term_geo is not None:
            #     l4, = ax_ts_twin.plot(term_geo.TIME, term_geo, color='brown', alpha=0.3, linewidth=1.5, linestyle=':', label='Geo')
            #     lines_twin.append(l4)

            # Primary Axis (SSTA)
            l5, = ax_ts.plot(model_ts.TIME, model_ts, label='Implicit Model SSTA', color='tab:blue', linewidth=2.5)
            l6, = ax_ts.plot(obs_ts.TIME, obs_ts, label='Observed SSTA', color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # --- D. ALIGN ZERO LINES & Y-LIMITS ---
            y1_min, y1_max = ax_ts.get_ylim()
            range1 = max(abs(y1_min), abs(y1_max))
            ax_ts.set_ylim(-range1, range1) 
            
            # y2_min, y2_max = ax_ts_twin.get_ylim()
            # range2 = max(abs(y2_min), abs(y2_max))
            # ax_ts_twin.set_ylim(-range2 * 1.5, range2 * 1.5) # Buffer for forcing terms
            
            ax_ts.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

            # --- E. METRICS & CLEAN UP ---
            model_aligned, obs_aligned = xr.align(model_ts, obs_ts)
            r_val = float(xr.corr(model_aligned, obs_aligned))
            rmse_val = float(np.sqrt(((model_aligned - obs_aligned)**2).mean()))
            
            # Subplot titles
            subplot_letter = chr(97 + idx) # a, b, c, d
            ax_ts.set_title(f"({subplot_letter}) {loc_name} | {found_lat:.1f}°N, {found_lon:.1f}°E\n$R$ = {r_val:.2f}, RMSE = {rmse_val:.2f} °C", 
                            fontsize=30, fontweight='bold', loc='left')
            ax_ts.tick_params(axis = 'both', which= 'major',labelsize=15)      
            # ax_ts_twin.tick_params(axis = 'both', which= 'major',labelsize=15)     
            # Clean up axes labels to prevent clutter in the grid
            # if idx % 2 == 0:  # Left column
            #     ax_ts.set_ylabel("SST Anomaly (°C)", fontsize=11, fontweight='medium')
            # if idx % 2 != 0:  # Right column
            #     ax_ts_twin.set_ylabel("Forcing Tendency (°C/month)", color='dimgray', fontsize=11, fontweight='medium')

            if idx == 0:
                ax_ts.set_ylabel("SST Anomaly (°C)", fontsize=20)
                # ax_ts_twin.set_ylabel("Forcing Tendency (°C/month)", fontsize=20, color='dimgray')

            
            # Despine top
            ax_ts.spines['top'].set_visible(True)
            # ax_ts_twin.spines['top'].set_visible(False)
            ax_ts.grid(True, alpha=0.2, color='gray')

            # Capture legend info from the first valid subplot
            if not legend_lines:
                legend_lines = [l5, l6] + lines_twin
                legend_labels = [l.get_label() for l in legend_lines]

        except Exception as e:
            ax_ts.set_title(f"Data missing for {loc_name}", fontsize=30)
            print(f"Failed to plot for {loc_name} at ({lon}, {lat}). Error: {e}")

    # --- F. UNIFIED LEGEND & FINAL LAYOUT ---
    # Add a single legend at the bottom center of the figure
    fig.legend(legend_lines, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
               fontsize=20, ncol=len(legend_lines), frameon=True, framealpha=0.9, edgecolor='lightgray')
    ax_ts.set_xlabel("Time", fontsize=20, fontweight='medium')
    # Adjust layout so subplots don't overlap the legend
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08) # Make room for the legend
    
    # Save output
    os.makedirs("plots", exist_ok=True)
    filename = "plots/Multi_Region_Timeseries_Subplots.png"
    plt.savefig(filename, bbox_inches='tight', dpi=500)
    print(f"Saved: {filename}")
    plt.show()

# --- Execute Subplot Plotting ---
locations_to_plot = [
    {"lon": 190.0, "lat": 0.0, "name": "Niño 3.4 (Equatorial Pacific)"},
    # {"lon": 150.0, "lat": 35.0, "name": "Kuroshio Extension"},
    # {"lon": 310.0, "lat": 40.0, "name": "Gulf Stream"},
    {"lon": 170.0, "lat": -50.0, "name": "Southern Ocean"}
]

dt_seconds = month_to_second(1)

plot_research_subplots(
    locations=locations_to_plot, 
    model_ds=implicit_model_anomaly_ds, 
    obs_ds=observed_temperature_anomaly, 
    fluxes_ds=flux_components_ds, 
    geo_da=geostrophic_anomaly_da, 
    hbar_da=hbar_da,
    dt_seconds=dt_seconds
)