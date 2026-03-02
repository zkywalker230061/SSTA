# =============================================================================
# Residual (Budget-Closure) Figure for your existing IMPLICIT simulation setup
# -----------------------------------------------------------------------------
# Goal:
#   Compute and plot the residual:
#
#     R = dT_obs/dt  -  RHS_model_terms
#
#   where RHS_model_terms are computed consistently with your simulation choices:
#   INCLUDE_SURFACE, INCLUDE_EKMAN, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_MEAN
#
# Observations:
#   Use Reynolds SST anomaly (observed_temperature_anomaly_reynold) as "T_obs".
#
# Outputs (publication-style):
#   Fig A) Global map of std(residual) [K/month]
#   Fig B) Global map of mean(residual) [K/month]
#   Fig C) Fraction of tendency variance unexplained: Var(R)/Var(dT_obs/dt)
#   Fig D) Regional time series: tendency vs RHS vs residual (box or points)
#
# =============================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import t as tdist
from chris_utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, coriolis_parameter
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

def month_in_year_from_time_index(time_coord):
    t = xr.DataArray(time_coord, dims=["TIME"])
    m = ((t + 0.5) % 12).astype(int)
    m = xr.where(m == 0, 12, m)
    return m

def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60

dt = month_to_second(1)

# -------------------------------------------------------------------------
# 2) Align datasets in TIME/space
# -------------------------------------------------------------------------
Tobs = observed_temperature_anomaly_reynold.rename("Tobs")
Tmod = implicit_model_anomaly_ds.rename("Tmod")

Tobs, Tmod = xr.align(Tobs, Tmod, join="inner")

# Ensure fluxes aligned (if they exist)
# NOTE: if ekman_anomaly_da is in temperature units (K/month) rather than W/m^2,
# you MUST adjust below (see the "Unit sanity check" section).
flux_list = []
if "surface_flux_da" in globals():
    surface_flux_da = surface_flux_da.rename("Qsurf")
    flux_list.append(surface_flux_da)
if "ekman_anomaly_da" in globals():
    ekman_anomaly_da = ekman_anomaly_da.rename("Qek")
    flux_list.append(ekman_anomaly_da)
if "geostrophic_anomaly_da" in globals():
    geostrophic_anomaly_da = geostrophic_anomaly_da.rename("Qgeo")
    flux_list.append(geostrophic_anomaly_da)

if len(flux_list) > 0:
    flux_all = xr.merge(flux_list)
    flux_all, Tobs = xr.align(flux_all, Tobs, join="inner")
    flux_all, Tmod = xr.align(flux_all, Tmod, join="inner")

# Entrainment flux term from your simulation (already computed if INCLUDE_ENTRAINMENT)
if INCLUDE_ENTRAINMENT:
    Qent = entrainment_flux_implicit_ds.rename("Qent")  # W/m^2 anomaly by construction
    Qent, Tobs = xr.align(Qent, Tobs, join="inner")
    Qent, Tmod = xr.align(Qent, Tmod, join="inner")

# -------------------------------------------------------------------------
# 3) Compute observed tendency dTobs/dt (K/s or K/month)
#    We'll compute in K/month for interpretation.
# -------------------------------------------------------------------------
# Use centered difference (better than forward)
# dT/dt in K/month (since dt is seconds, multiply by seconds/month at end)
# dTobs_dt = (Tobs.shift(TIME=-1) - Tobs.shift(TIME=1)) / (2 * dt)   # K/s
dTobs_dt = (Tobs.shift(TIME=-1) - Tobs)/dt
dTobs_dt = dTobs_dt * dt                                          # K/month (per 1-month step)
dTobs_dt = dTobs_dt.rename("dTobs_dt")

# Optional: also compute model tendency (for an additional check)
dTmod_dt = (Tmod.shift(TIME=-1) - Tmod.shift(TIME=1)) / (2 * dt)
dTmod_dt = dTmod_dt * dt
dTmod_dt = dTmod_dt.rename("dTmod_dt")

# -------------------------------------------------------------------------
# 4) Build RHS tendency from terms, consistent with your model configuration
#    RHS should be in K/month.
#
#    If Q terms are W/m^2:
#      tendency contribution = (Q / (rho*c*h)) * dt   [K]
#
#    For entrainment:
#      you already built Qent = rho*c*w_e*(Tsub - Tm_model) [W/m^2]
#      so tendency contribution = (Qent / (rho*c*h)) * dt
#
#    Damping term:
#      model uses + gamma/(rho*c*h) * ( -T ) implicitly
#      Here we include as: -(gamma/(rho*c*h)) * Tobs * dt  OR Tmod * dt
#      For closure against observed tendency, use Tobs.
#
#    IMPORTANT:
#      Your heat budget in the simulation uses anomalies; residual should also
#      use anomalies consistently. We'll use Tobs anomaly for damping.
# -------------------------------------------------------------------------

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = gamma_0  # from your script

# Build h(TIME,...) from monthly climatology hbar_da(MONTH,...)
month_index = month_in_year_from_time_index(Tobs["TIME"].values)  # dims TIME, values 1..12
h_time = xr.concat([hbar_da.sel(MONTH=int(m)) for m in month_index.values], dim=Tobs["TIME"])
h_time = h_time.assign_coords(TIME=Tobs["TIME"]).rename("h")
h_time, Tobs = xr.align(h_time, Tobs, join="inner")

lat = 50.5
lon = -24.5

h_point = h_time.sel(LATITUDE=lat, LONGITUDE=lon, method="nearest")
print("Min h:", float(h_point.min().values))
print("Max h:", float(h_point.max().values))
print("Mean h:", float(h_point.mean().values))

# Surface + Ekman + Geostrophic flux sum in W/m^2
Qsum = 0
if INCLUDE_SURFACE:
    Qsum = Qsum + flux_all["Qsurf"]
if INCLUDE_EKMAN:
    Qsum = Qsum + flux_all["Qek"]
if INCLUDE_GEOSTROPHIC_MEAN:
    Qsum = Qsum + flux_all["Qgeo"]

# Convert to K/month tendency contribution
rhs_flux = (Qsum / (rho_0 * c_0 * h_time)) * dt
rhs_flux = rhs_flux.rename("rhs_flux")

# Entrainment contribution
if INCLUDE_ENTRAINMENT:
    rhs_ent = (Qent / (rho_0 * c_0 * h_time)) * dt
    rhs_ent = rhs_ent.rename("rhs_ent")
else:
    rhs_ent = 0

# Damping contribution (use observed anomaly for closure)
rhs_damp = -(gamma_0 / (rho_0 * c_0 * h_time)) * Tobs * dt
rhs_damp = rhs_damp.rename("rhs_damp")

# Total RHS
rhs_total = rhs_flux + rhs_ent + rhs_damp
rhs_total = rhs_total.rename("rhs_total")

# -------------------------------------------------------------------------
# 5) Residual: observed tendency minus RHS
# -------------------------------------------------------------------------
residual = (dTobs_dt - rhs_total).rename("residual")  # K/month

# Mask edges (since centered difference makes first/last NaN)
valid_time = slice(1, -1)
dTobs_dt_v = dTobs_dt.isel(TIME=valid_time)
rhs_total_v = rhs_total.isel(TIME=valid_time)
residual_v = residual.isel(TIME=valid_time)

# -------------------------------------------------------------------------
# 6) Metrics for maps
# -------------------------------------------------------------------------
res_mean = residual_v.mean(dim="TIME", skipna=True).rename("res_mean")    # K/month
res_std  = residual_v.std(dim="TIME", skipna=True).rename("res_std")     # K/month

tend_var = dTobs_dt_v.var(dim="TIME", skipna=True)
res_var  = residual_v.var(dim="TIME", skipna=True)
frac_unexpl = (res_var / tend_var).rename("frac_unexpl")  # dimensionless

# Clip frac_unexpl for plotting stability
frac_unexpl_plot = frac_unexpl.clip(0, 2)

# -------------------------------------------------------------------------
# 7) Plot helpers (Cartopy global)
# -------------------------------------------------------------------------
def plot_global_map(da, title, cbar_label, vmin=None, vmax=None, cmap="RdBu_r"):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none", zorder=1)

    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        add_colorbar=False,
        zorder=0
    )
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.85)
    cbar.set_label(cbar_label)
    ax.set_title(title, loc="left")
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# 8) FIGURE A/B/C: Residual summary maps
# -------------------------------------------------------------------------
# A) Mean residual (bias-like): should be near 0 if closure good
plot_global_map(
    res_mean,
    title="Residual Mean:  ⟨ dT_obs/dt  -  dT_mod/dt ⟩",
    cbar_label="K / month",
    cmap="RdBu_r",
    vmin=-0.5, vmax=0.5
)

# B) Std residual: where your terms fail to explain variability
plot_global_map(
    res_std,
    title="Residual Std Dev:  std( dT_obs/dt  -  dT_mod/dt )",
    cbar_label="K / month",
    cmap="RdBu_r",
    vmin=0, vmax=float(np.nanpercentile(res_std.values, 95))
)

# C) Fraction unexplained variance: Var(residual)/Var(tendency)
plot_global_map(
    frac_unexpl_plot,
    title="Fraction of tendency variance unexplained: Var(R) / Var(dT_obs/dt)",
    cbar_label="dimensionless",
    cmap="RdBu_r",
    vmin=0, vmax=2
)

# -------------------------------------------------------------------------
# 9) FIGURE D: Regional time series closure plot (points, matching your locations)
#    Shows: dTobs/dt, RHS_total, residual.
# -------------------------------------------------------------------------
locations = [
    {'name': 'Southern Ocean', 'lat': -52.5, 'lon': -95.5, 'color': 'red'},
    {'name': 'North Atlantic', 'lat': 41.5, 'lon': -50.5, 'color': 'green'},
    {'name': 'North Atlantic 2', 'lat': 50, 'lon': -25, 'color': 'pink'},
    {'name': 'Indian', 'lat': -20, 'lon': 75, 'color': 'blue'},
    {'name': 'North Pacific', 'lat': 30, 'lon': -150, 'color': 'goldenrod'},
    {'name': 'Cape Agulhas', 'lat': -40, 'lon': 25, 'color': 'orange'},
]

def plot_closure_timeseries_point(lat, lon, name, months=60):
    # nearest grid cell
    tend_p = dTobs_dt_v.sel(LATITUDE=lat, LONGITUDE=lon, method="nearest")
    rhs_p  = rhs_total_v.sel(LATITUDE=lat, LONGITUDE=lon, method="nearest")
    res_p  = residual_v.sel(LATITUDE=lat, LONGITUDE=lon, method="nearest")

    # Take a subset (e.g., first N months)
    tend_p = tend_p.isel(TIME=slice(0, months))
    rhs_p  = rhs_p.isel(TIME=slice(0, months))
    res_p  = res_p.isel(TIME=slice(0, months))

    plt.figure(figsize=(11, 4))
    plt.plot(tend_p["TIME"], tend_p, label="Observed tendency dT/dt (Reynolds)", linewidth=1.5)
    plt.plot(rhs_p["TIME"], rhs_p, label="Diagnosed RHS (model terms)", linewidth=1.5)
    plt.plot(res_p["TIME"], res_p, label="Residual (Obs - RHS)", linewidth=1.2, alpha=0.9)

    plt.axhline(0, color="k", linewidth=0.8, alpha=0.6)
    plt.title(f"Closure at point: {name} (lat={float(tend_p.LATITUDE.values):.2f}, lon={float(tend_p.LONGITUDE.values):.2f})")
    plt.ylabel("K / month")
    plt.xlabel("TIME index")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# for loc in locations:
#     plot_closure_timeseries_point(loc["lat"], loc["lon"], loc["name"], months=80)

# -------------------------------------------------------------------------
# 10) OPTIONAL: seasonal composite of residual (DJF/MAM/JJA/SON)
#     FIXED: build season labels from residual_v.TIME so groupby aligns.
# -------------------------------------------------------------------------

\

def season_labels_from_time(time_coord):
    """
    Make season labels (DJF/MAM/JJA/SON) from numeric monthly TIME coordinate.
    Returns a DataArray with the SAME TIME coordinate, so groupby won't crash.
    """
    month = ((time_coord + 0.5) % 12).astype(int)
    month = xr.where(month == 0, 12, month)  # make it 1..12

    season = xr.where((month == 12) | (month <= 2), "DJF",
              xr.where((month >= 3) & (month <= 5), "MAM",
              xr.where((month >= 6) & (month <= 8), "JJA", "SON")))

    # Ensure it's a DataArray with TIME coord
    season = xr.DataArray(season.data, coords={"TIME": time_coord}, dims=["TIME"], name="season")
    return season


# --- IMPORTANT: residual_v is what you want to composite ---
# If you used centered differencing and trimmed TIME, residual_v.TIME is shorter than Tobs.TIME.
# So ALWAYS derive labels from residual_v.TIME (not Tobs.TIME / month_index).

season_labels = season_labels_from_time(residual_v.TIME)

# Optional safety check (can comment out after it works)
# This will raise if TIME coords differ
residual_v, season_labels = xr.align(residual_v, season_labels, join="exact")

# Seasonal mean residual maps
residual_seasonal = residual_v.groupby(season_labels).mean(dim="TIME", skipna=True)

for s in ["DJF", "MAM", "JJA", "SON"]:
    plot_global_map(
        residual_seasonal.sel(season=s),
        title=f"Residual Mean by season: {s}",
        cbar_label="K / month",
        cmap="RdBu_r",
        vmin=-0.2, vmax=0.2
    )
# =============================================================================
# UNIT SANITY CHECK (IMPORTANT)
# =============================================================================
# This residual analysis assumes:
#   surface_flux_da, ekman_anomaly_da, geostrophic_anomaly_da, and Qent are in W/m^2.
#
# If your ekman_anomaly_da is ALREADY in temperature tendency units (K/month),
# then you should NOT divide by (rho*c*h). In that case, modify Qsum construction:
#
#   rhs_flux = (Qsurf/(rho*c*h))*dt  +  (Tekman)*1  + ...
#
# Quick check: print typical magnitudes:
#   surface_flux_da.std().values ~ O(10-100) W/m^2
#   temperature tendency ~ O(0.1) K/month
#
# If ekman_anomaly_da.std() is O(0.1), it's probably already K/month.
# =============================================================================