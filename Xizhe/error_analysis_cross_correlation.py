#%%
# --- 1. Running Implicit Scheme ---------------------------------- 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from chris_utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, coriolis_parameter
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.stats import kurtosis, skew, pearsonr, t

matplotlib.use('TkAgg')

INCLUDE_SURFACE = False
INCLUDE_EKMAN = False
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC_MEAN = False
INCLUDE_GEOSTROPHIC_ANOM = False
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

T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/t_sub.nc"
NEW_T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_T_sub_prime.nc"

GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated_2.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"

USE_DOWNLOADED_SSH = False
USE_NEW_H_BAR_NEW_T_SUB = True 

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

flux_components_ds = xr.merge(flux_components_to_merge)

#--- 2.1 Prepare Observed Temperature Anomaly ------------------------------------------------------------


implicit_model_anomaly_ds = all_anomalies_ds["IMPLICIT"]


#--- 2.2 Helper Functions (Visualization) ------------------------------------------------------------
def make_lag_movie(data_array, vmin=-1, vmax=1, savepath=None, cmap='RdBu_r'):
    """Generates an animation of the correlation map across lags."""
    lags = data_array.lag.values
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Initial Plot
    mesh = data_array.isel(lag=0).plot(
        ax=ax, 
        cmap=cmap, 
        vmin=vmin, vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={'label': 'Correlation Coefficient'}
    )
    
    title = ax.set_title(f'Lag: {lags[0]} months')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    def update(frame):
        data_slice = data_array.isel(lag=frame).values
        mesh.set_array(data_slice.ravel())
        title.set_text(f'Lag: {lags[frame]} months')
        return [mesh, title]

    anim = FuncAnimation(fig, update, frames=len(lags), interval=600, blit=False)
    
    if savepath:
        anim.save(savepath, fps=5, dpi=150)
    plt.show()

def plot_map(data, title, label, cmap="nipy_spectral", vmin=None, vmax=None, levels=None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    
    # Handle levels if provided (for discrete colorbar)
    plot_kwargs = {'cmap': cmap, 'cbar_kwargs': {'label': label}}
    if vmin is not None and vmax is not None:
        plot_kwargs.update({'vmin': vmin, 'vmax': vmax})
    if levels is not None:
        plot_kwargs.update({'levels': levels})

    data.plot(ax=ax, **plot_kwargs)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.text(
        0.99, 0.01,
        f"Gamma = {gamma_0}\n"
        f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
        f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
        f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
        f"INCLUDE_GEOSTROPHIC = {INCLUDE_GEOSTROPHIC_MEAN}\n"
        f"INCLUDE_GEOSTROPHIC_DISPLACEMENT = {INCLUDE_GEOSTROPHIC_ANOM}",
        ha='right', va='bottom', fontsize=8
        )
    plt.tight_layout()
    plt.show()

#%%
#--- 3. Cross Correlation Calculation ----------------------------------------------------------------

# 3.1 Setup Parameters
lags = np.arange(-12, 13)
scheme_name = "Implicit"
lat = observed_temperature_anomaly['LATITUDE'].values
lon = observed_temperature_anomaly['LONGITUDE'].values

# 3.2 Pre-calculate Autocorrelation (Persistence)
# High autocorrelation reduces the Effective Degrees of Freedom (N_eff).
r_x = xr.corr(observed_temperature_anomaly, 
              observed_temperature_anomaly.shift(TIME=1), dim="TIME")
r_y = xr.corr(implicit_model_anomaly_ds, 
              implicit_model_anomaly_ds.shift(TIME=1), dim="TIME")

# 3.3 Core Calculation Function
def calculate_lag_stats(obs, model, lag, r_x_map, r_y_map):
    """
    Calculates Correlation (r), T-statistic, and Effective N for a single lag.
    """
    # Shift model: Positive lag => Model lags Obs
    model_shifted = model.shift(TIME=lag)
    
    # 1. Correlation
    r = xr.corr(obs, model_shifted, dim="TIME")
    
    # 2. Effective Degrees of Freedom (Bretherton et al. 1999)
    # N_total = 180. We subtract abs(lag) because shifting loses data points.
    N_lagged = 180 - abs(lag)
    N_effective = N_lagged * (1 - r_x_map * r_y_map) / (1 + r_x_map * r_y_map)
    
    # 3. T-Statistic
    t_stat = r * np.sqrt((N_effective - 2) / (1 - r**2))
    
    return r, t_stat, N_effective

# 3.4 Execute Loop over Lags
print("Calculating cross-correlations...")
results = [calculate_lag_stats(observed_temperature_anomaly, 
                               implicit_model_anomaly_ds, 
                               k, r_x, r_y) for k in lags]

r_list, t_list, n_list = zip(*results)
lag_dim = xr.DataArray(lags, dims="lag", name="lag")

corr_by_lag = xr.concat(r_list, dim=lag_dim)
t_stat_by_lag = xr.concat(t_list, dim=lag_dim)
n_eff_by_lag = xr.concat(n_list, dim=lag_dim)

# 3.5 Calculate Significance (P-Value)
# Two-tailed test using Survival Function (sf = 1 - cdf)
p_values_da = 2 * t.sf(np.abs(t_stat_by_lag), df=n_eff_by_lag - 2)
p_values_da = xr.DataArray(p_values_da, coords=t_stat_by_lag.coords, name="p_value")

# Mask for significance (95% confidence)
significant_mask = p_values_da < 0.05
sig_correlations = corr_by_lag.where(significant_mask)

#%%
# --- 4. Visualization: Static Map (Lag 0) -----------------------------------------------------------
plot_map(
    data=corr_by_lag.sel(lag=0),
    title=f'{scheme_name} Scheme - Cross Correlation (Lag 0)',
    label='Correlation',
    cmap='nipy_spectral', 
    vmin=-1, vmax=1
)

#%%
# --- 5. Visualization: Time Series (Selected Locations) ---------------------------------------------
locations = [
    {'name': 'Southern Ocean', 'lat': -52.5, 'lon': -95.5, 'color': 'red'},
    {'name': 'North Atlantic', 'lat': 41.5, 'lon': -50.5, 'color': 'green'},
    {'name': 'North Atlantic 2', 'lat': 50, 'lon': -25, 'color': 'pink'},
    {'name': 'Indian', 'lat': -20, 'lon': 75, 'color': 'blue'},
    {'name': 'North Pacific', 'lat': 30, 'lon': -150, 'color': 'goldenrod'},
    {'name': 'Cape Agulhas', 'lat': -40, 'lon': 25, 'color': 'orange'},
]

plt.figure(figsize=(10, 6))

for loc in locations:
    # 'nearest' selects the closest grid cell to the specified lat/lon
    point_data = corr_by_lag.sel(LATITUDE=loc['lat'], LONGITUDE=loc['lon'], method='nearest')
    
    plt.plot(
        point_data['lag'], 
        point_data, 
        label=f"{loc['name']}", 
        color=loc['color'], 
        marker='o', markersize=4, alpha=0.8
    )

plt.axvline(0, color='k', linestyle='--', alpha=0.5, label='Zero Lag')
plt.axhline(0, color='k', linewidth=0.8)
plt.ylim(-1, 1)
plt.xlabel("Lag (months)\n(Positive: Model lags Obs)")
plt.ylabel("Cross-correlation")
plt.title(f"{scheme_name} Scheme: Lagged Cross-Correlation")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# --- 6. Visualization: Movie ------------------------------------------------------------------------
# make_lag_movie(corr_by_lag, vmin=-1, vmax=1, savepath=None)

#%%
# --- 7. Visualization: Maps of Best Lag -------------------------------------------------------------

# Strategy: Find the lag that produces the Strongest Magnitude Correlation (Positive OR Negative)
# A) best positive correlation:
# best_lag = corr_by_lag.idxmax(dim="lag")
# best_corr = corr_by_lag.max(dim="lag")

# B) strongest magnitude (ignoring sign):
mask_obs = corr_by_lag.notnull().all(dim="lag")
abs_run = np.abs(corr_by_lag)
abs_filled = abs_run.where(np.isfinite(abs_run), -np.inf)
best_idx = abs_filled.argmax(dim="lag") # dims: (LATITUDE, LONGITUDE)
best_lag = corr_by_lag["lag"].isel(lag=best_idx) # dims: (LATITUDE, LONGITUDE)
best_lag = best_lag.where(mask_obs)
best_corr = corr_by_lag.isel(lag=best_idx) # dims: (LATITUDE, LONGITUDE)

# # C) best (most negative) correlation:
# best_lag = corr_by_lag.idxmin(dim="lag")
# best_corr = corr_by_lag.min(dim="lag")

# --- Plot 7.1: Value of the Best Correlation ---
plot_map(
    data=best_corr,
    title=f"{scheme_name}: Max Correlation Value (at best lag)",
    label="Correlation",
    cmap="nipy_spectral",
    vmin=-1, vmax=1
)

# --- Plot 7.2: Which Lag was the Best? ---
# define discrete levels for the colorbar so each month is distinct
lag_levels = np.arange(lags.min() - 0.5, lags.max() + 1.5, 1)
plot_map(
    data=best_lag,
    title=f"{scheme_name}: Lag (months) of Max Correlation",
    label="Lag (months)",
    cmap="nipy_spectral",
    levels=lag_levels
)

#%%
# --- 8. Visualization: Maps of Best Lag (SIGNIFICAN ADDED) -------------------------------------------------------------

# B) strongest magnitude (ignoring sign):
mask_obs_sig = sig_correlations.notnull().all(dim="lag")
abs_run_sig = np.abs(sig_correlations)
abs_filled_sig = abs_run_sig.where(np.isfinite(abs_run_sig), -np.inf)
best_idx_sig = abs_filled_sig.argmax(dim="lag") # dims: (LATITUDE, LONGITUDE)
best_lag_sog =  sig_correlations["lag"].isel(lag=best_idx) # dims: (LATITUDE, LONGITUDE)
best_lag_sig = best_lag_sog.where(mask_obs_sig)
best_corr_sig = sig_correlations.isel(lag=best_idx_sig) # dims: (LATITUDE, LONGITUDE)


# --- Plot 7.1: Value of the Best Correlation ---
plot_map(
    data=best_corr_sig,
    title=f"{scheme_name}: Max Correlation Value (at best lag with significance level 95%)",
    label="Correlation",
    cmap="nipy_spectral",
    vmin=-1, vmax=1
)

# --- Plot 7.2: Which Lag was the Best? ---
# define discrete levels for the colorbar so each month is distinct
lag_levels = np.arange(lags.min() - 0.5, lags.max() + 1.5, 1)
plot_map(
    data=best_lag,
    title=f"{scheme_name}: Lag (months) of Max Correlation with significance level 95%",
    label="Lag (months)",
    cmap="nipy_spectral",
    levels=lag_levels
)