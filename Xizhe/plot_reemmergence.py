# ============================================================================
# Reemergence / Persistence diagnostics: Month-conditioned Auto Correlation Function
# Observed = Reynolds anomalies
# Model    = implicit_model_anomaly_ds
#
# Output:
# 1) For each region/location: month-conditioned Auto Correlation Functions curves (obs vs model)
# 2) Optional significance estimates using effective N (lag-dependent)
#
# Notes:
# - Month-conditioned Auto Correlation Function: for each start month m (1..12), compute corr(x(t), x(t+L))
#   using only t where month(t)=m.
# - Works for point locations (nearest grid) OR box-mean regions.
# ============================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import t as tdist
from chris_utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, coriolis_parameter
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset


# ----------------------------------------------------------------------------
# 0) Inputs (use your existing objects from the notebook)
# ----------------------------------------------------------------------------

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC_MEAN = True
INCLUDE_GEOSTROPHIC_ANOM = True
CLEAN_CHRIS_PREV_CUR = False 

USE_DOWNLOADED_SSH = False
USE_NEW_H_BAR_NEW_T_SUB = False 

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15
g = 9.81
f = 1 


observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
observed_path_Reynolds = "/Users/julia/Desktop/SSTA/datasets/Reynold_sst_anomalies-(2004-2018).nc"
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



# Unchanged Parameters for the simulation 
temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)

observed_temp_ds_reynold = xr.open_dataset(observed_path_Reynolds, decode_times=False)['anom']
observed_temperature_anomaly_reynold = observed_temp_ds_reynold

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


# Entrainment Velocity
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']


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

obs = observed_temperature_anomaly_reynold  # dims: TIME, LATITUDE, LONGITUDE
mod = implicit_model_anomaly_ds             # dims: TIME, LATITUDE, LONGITUDE

# Make sure TIME aligns
obs, mod = xr.align(obs, mod, join="inner")

# If your Reynolds "TIME" is numeric months (e.g., 0..179) and you don't have datetime,
# we build a synthetic "month-in-year" index as you did in the simulation loop.
# If you DO have datetime coords, you can use obs['TIME'].dt.month instead.
def month_in_year_from_time_index(time_coord):
    # time_coord expected 1D array-like (len = nt)
    # We follow your convention: int((t + 0.5) % 12); 0 -> 12
    t = xr.DataArray(time_coord, dims=["TIME"])
    m = ((t + 0.5) % 12).astype(int)
    m = xr.where(m == 0, 12, m)
    return m

month_index = month_in_year_from_time_index(obs["TIME"].values)  # dims: TIME

# ----------------------------------------------------------------------------
# 1) Helpers: select a region time series (point or box average)
# ----------------------------------------------------------------------------
def select_point_ts(da, lat, lon):
    """Nearest-gridpoint time series at (lat, lon)."""
    return da.sel(LATITUDE=lat, LONGITUDE=lon, method="nearest")

def select_boxmean_ts(da, lat1, lat2, lon1, lon2):
    """Area-mean time series over a lat/lon box."""
    sub = da.sel(
        LATITUDE=slice(min(lat1, lat2), max(lat1, lat2)),
        LONGITUDE=slice(min(lon1, lon2), max(lon1, lon2))
    )
    return sub.mean(dim=["LATITUDE", "LONGITUDE"], skipna=True)

# ----------------------------------------------------------------------------
# 2) Core: month-conditioned ACF for a 1D time series
# ----------------------------------------------------------------------------
def month_conditioned_acf_1d(ts, month_index, max_lag=24, min_pairs=20):
    """
    Month-conditioned autocorrelation:
    For each start month m (1..12), corr(ts(t), ts(t+lag)) computed using only
    times t where month_index(t) == m.

    Parameters
    ----------
    ts : xr.DataArray (TIME,)
    month_index : xr.DataArray (TIME,) values in 1..12
    max_lag : int
    min_pairs : int  (skip correlations with fewer valid pairs)

    Returns
    -------
    acf : xr.DataArray dims (start_month, lag)
    n_pairs : xr.DataArray dims (start_month, lag)
    """
    lags = np.arange(0, max_lag + 1)
    start_months = np.arange(1, 13)

    acf = xr.DataArray(
        np.full((12, len(lags)), np.nan, dtype=float),
        coords={"start_month": start_months, "lag": lags},
        dims=("start_month", "lag"),
        name="acf"
    )
    n_pairs = xr.DataArray(
        np.zeros((12, len(lags)), dtype=int),
        coords={"start_month": start_months, "lag": lags},
        dims=("start_month", "lag"),
        name="n_pairs"
    )

    # Ensure 1D
    ts = ts.squeeze()

    for mi, m in enumerate(start_months):
        mask_m = (month_index == m)

        # Indices (in TIME) that belong to this start month
        idx = np.where(mask_m.values)[0]

        for li, L in enumerate(lags):
            idx2 = idx + L
            idx2 = idx2[idx2 < ts.sizes["TIME"]]

            idx1 = idx[:len(idx2)]  # match length

            x = ts.isel(TIME=idx1)
            y = ts.isel(TIME=idx2)

            # Pairwise valid mask
            valid = np.isfinite(x.values) & np.isfinite(y.values)
            n = int(valid.sum())
            n_pairs.loc[dict(start_month=m, lag=L)] = n

            if n < min_pairs:
                continue

            xv = x.values[valid]
            yv = y.values[valid]

            # If variance is zero, corr undefined
            if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
                continue

            r = np.corrcoef(xv, yv)[0, 1]
            acf.loc[dict(start_month=m, lag=L)] = r

    return acf, n_pairs

# ----------------------------------------------------------------------------
# 3) Optional: significance for ACF using an effective N estimate
# ----------------------------------------------------------------------------
def acf_significance(acf, n_pairs, r1, alpha=0.05):
    """
    Two-sided p-values for correlation with an effective DOF.
    For autocorrelation, a simple approximation is:
      Neff = N * (1 - r1^2) / (1 + r1^2)
    where r1 is lag-1 autocorrelation (can be month-specific).

    Parameters
    ----------
    acf : (start_month, lag)
    n_pairs : (start_month, lag)
    r1 : xr.DataArray (start_month,)  lag-1 ACF by start month
    alpha : float

    Returns
    -------
    pvals : xr.DataArray (start_month, lag)
    sigmask : xr.DataArray (start_month, lag) True where p < alpha
    """
    # Broadcast r1 to (start_month, lag)
    r1b = r1.broadcast_like(acf)

    Neff = n_pairs * (1 - r1b**2) / (1 + r1b**2)
    Neff = xr.where(Neff < 3, np.nan, Neff)  # avoid df <= 1

    tstat = acf * np.sqrt((Neff - 2) / (1 - acf**2))
    pvals = 2 * xr.apply_ufunc(lambda z, df: tdist.sf(np.abs(z), df),
                               tstat, Neff - 2)
    sigmask = pvals < alpha
    return pvals, sigmask

# ----------------------------------------------------------------------------
# 4) Build the diagnostics for your "same regions"
#    You can interpret your list either as points OR boxes.
#    Below I implement POINTS (nearest grid), matching your existing locations list.
# ----------------------------------------------------------------------------
locations = [
    {'name': 'Southern Ocean', 'lat': -52.5, 'lon': -95.5},
    {'name': 'North Atlantic', 'lat': 41.5, 'lon': -50.5},
    {'name': 'North Atlantic 2', 'lat': 50, 'lon': -25},
    {'name': 'Indian', 'lat': -20, 'lon': 75},
    {'name': 'North Pacific', 'lat': 30, 'lon': -150},
    {'name': 'Cape Agulhas', 'lat': -40, 'lon': 25},
]

regions = [
    # {"name":"SPG", "lat1":45, "lat2":65, "lon1":-60, "lon2":-20},
    # {"name":"Gulf Stream", "lat1":30, "lat2":45, "lon1":-80, "lon2":-40},
    {"name":"North Atlantic", "lat1":23.5, "lat2":70, "lon1":-80, "lon2":0},
    {"name":"Southern Ocean", "lat1":-23.5, "lat2":-65, "lon1":-180, "lon2":180},
    {"name":"South Pacific Ocean", "lat1":-23.5, "lat2":-65, "lon1":-180, "lon2":-67},
]

MAX_LAG = 24        # commonly 24–60; 36 is a nice compromise
MIN_PAIRS = 10      # for monthly-conditioned, pairs per month can be small

results = {}  # dict: name -> dict with acf_obs, acf_mod, etc.

# for loc in locations:
#     name = loc["name"]
#     obs_ts = select_point_ts(obs, loc["lat"], loc["lon"])
#     mod_ts = select_point_ts(mod, loc["lat"], loc["lon"])

#     # Ensure both are 1D TIME
#     obs_ts = obs_ts.drop_vars([v for v in obs_ts.coords if v not in ["TIME"]], errors="ignore")
#     mod_ts = mod_ts.drop_vars([v for v in mod_ts.coords if v not in ["TIME"]], errors="ignore")

#     acf_obs, n_obs = month_conditioned_acf_1d(obs_ts, month_index, max_lag=MAX_LAG, min_pairs=MIN_PAIRS)
#     acf_mod, n_mod = month_conditioned_acf_1d(mod_ts, month_index, max_lag=MAX_LAG, min_pairs=MIN_PAIRS)

#     # Optional significance masks (month-specific r1 = ACF at lag=1)
#     # If lag=1 is NaN for some months (too few pairs), significance will be NaN there.
#     r1_obs = acf_obs.sel(lag=1)
#     r1_mod = acf_mod.sel(lag=1)

#     p_obs, sig_obs = acf_significance(acf_obs, n_obs, r1_obs, alpha=0.05)
#     p_mod, sig_mod = acf_significance(acf_mod, n_mod, r1_mod, alpha=0.05)

#     results[name] = dict(
#         acf_obs=acf_obs, n_obs=n_obs, p_obs=p_obs, sig_obs=sig_obs,
#         acf_mod=acf_mod, n_mod=n_mod, p_mod=p_mod, sig_mod=sig_mod
#     )

for loc in regions:
    name = loc["name"]
    # obs_ts = select_point_ts(obs, loc["lat"], loc["lon"])
    # mod_ts = select_point_ts(mod, loc["lat"], loc["lon"])
    obs_ts = select_boxmean_ts(obs, loc["lat1"], loc["lat2"], loc["lon1"], loc["lon2"])
    mod_ts = select_boxmean_ts(mod, loc["lat1"], loc["lat2"], loc["lon1"], loc["lon2"])

    # Ensure both are 1D TIME
    obs_ts = obs_ts.drop_vars([v for v in obs_ts.coords if v not in ["TIME"]], errors="ignore")
    mod_ts = mod_ts.drop_vars([v for v in mod_ts.coords if v not in ["TIME"]], errors="ignore")

    acf_obs, n_obs = month_conditioned_acf_1d(obs_ts, month_index, max_lag=MAX_LAG, min_pairs=MIN_PAIRS)
    acf_mod, n_mod = month_conditioned_acf_1d(mod_ts, month_index, max_lag=MAX_LAG, min_pairs=MIN_PAIRS)

    # Optional significance masks (month-specific r1 = ACF at lag=1)
    # If lag=1 is NaN for some months (too few pairs), significance will be NaN there.
    r1_obs = acf_obs.sel(lag=1)
    r1_mod = acf_mod.sel(lag=1)

    p_obs, sig_obs = acf_significance(acf_obs, n_obs, r1_obs, alpha=0.05)
    p_mod, sig_mod = acf_significance(acf_mod, n_mod, r1_mod, alpha=0.05)

    results[name] = dict(
        acf_obs=acf_obs, n_obs=n_obs, p_obs=p_obs, sig_obs=sig_obs,
        acf_mod=acf_mod, n_mod=n_mod, p_mod=p_mod, sig_mod=sig_mod
    )

# ----------------------------------------------------------------------------
# 5) Plotting: month-conditioned ACF, observed vs model (one figure per region)
# ----------------------------------------------------------------------------
def plot_month_conditioned_acf(region_name, pack, max_lag=MAX_LAG, show_sig=True):
    acf_obs = pack["acf_obs"]
    acf_mod = pack["acf_mod"]
    sig_obs = pack["sig_obs"]
    sig_mod = pack["sig_mod"]

    start_months = acf_obs["start_month"].values
    lags = acf_obs["lag"].values

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, m in enumerate(start_months):
        ax = axes[i]
        y_obs = acf_obs.sel(start_month=m)
        y_mod = acf_mod.sel(start_month=m)

        ax.plot(lags, y_obs, label="Obs (Reynolds)", linewidth=1.5)
        ax.plot(lags, y_mod, label="Model (Implicit)", linewidth=1.5)

        if show_sig:
            # Mark significant points with small dots (optional)
            s_obs = sig_obs.sel(start_month=m)
            s_mod = sig_mod.sel(start_month=m)

            ax.scatter(lags[s_obs.values.astype(bool)], y_obs.values[s_obs.values.astype(bool)],
                       s=10, marker="o")
            ax.scatter(lags[s_mod.values.astype(bool)], y_mod.values[s_mod.values.astype(bool)],
                       s=10, marker="x")

        ax.axhline(0, linewidth=0.8, alpha=0.5)
        ax.set_title(f"Start month = {m}", fontsize=10)

    # Common styling
    for ax in axes[-4:]:
        ax.set_xlabel("Lag (months)")
    for ax in axes[::4]:
        ax.set_ylabel("ACF")

    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Month-conditioned ACF (Reemergence/Persistence)\nRegion: {region_name}", y=1.02)
    plt.tight_layout()
    plt.show()

# Make plots
for name, pack in results.items():
    plot_month_conditioned_acf(name, pack, max_lag=MAX_LAG, show_sig=True)

# ----------------------------------------------------------------------------
# 6) OPTIONAL: If you prefer a "seasonal reemergence summary" plot
#    Example: compare start-month=Feb (2) through next 24 months (reemergence often shows winter->winter)
# ----------------------------------------------------------------------------
def plot_selected_start_month(region_name, pack, start_month=2):
    lags = pack["acf_obs"]["lag"].values
    y_obs = pack["acf_obs"].sel(start_month=start_month)
    y_mod = pack["acf_mod"].sel(start_month=start_month)

    plt.figure(figsize=(7,4))
    plt.plot(lags, y_obs, label="Obs (Reynolds)", linewidth=2)
    plt.plot(lags, y_mod, label="Model (Implicit)", linewidth=2)
    plt.axhline(0, linewidth=0.8, alpha=0.5)
    plt.xlabel("Lag (months)")
    plt.ylabel("ACF")
    plt.title(f"{region_name}: Month-conditioned ACF (start month = {start_month})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_selected_start_month("North Atlantic", results["North Atlantic"], start_month=2)