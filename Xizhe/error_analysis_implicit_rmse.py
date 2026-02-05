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
from scipy.stats import kurtosis, skew, pearsonr

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEO_MEAN = True
INCLUDE_GEO_ANOM = True
CLEAN_CHRIS_PREV_CUR = False        # only really useful when entrainment is turned on

observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
# HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
ENTRAINMENT_VEL_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"
# H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/t_sub.nc"
GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"
USE_DOWNLOADED_SSH = False

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15
g = 9.81
f = 1 

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
        if INCLUDE_GEO_ANOM:    # then need to take the previous reading "back-propagated" based on current
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

        if INCLUDE_GEO_MEAN:
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

#------------------------------------------------------------------------------------------------------------
#%%
def calculate_RMSE (obs, model, dim = 'TIME'):
    """
    Calculates Root Mean Square Error.
    Formula: sqrt( mean( (obs - model)^2 ) )
    """
    error = (model - obs)
    squared_error = error ** 2
    mean_squared_error = squared_error.mean(dim=dim)
    rmse = np.sqrt(mean_squared_error)
    return rmse

def calculate_RMSE_weighted(obs, model, dim= 'TIME'):
    error = (model - obs)
    squared_error = error ** 2
    mean_squared_error = squared_error.mean(dim=dim)
    rmse = np.sqrt(mean_squared_error)

    weights = np.cos(np.deg2rad(mean_squared_error.LATITUDE))
    weighted_rmse = mean_squared_error.weighted(weights).mean (dim = ("LATITUDE", "LONGITUDE"))
    rmse_weighted = float(np.sqrt(weighted_rmse))
    return rmse_weighted

def calculate_RMSE_norm(obs, model, dim = 'TIME'):
    error = (model - obs)
    squared_error = error ** 2
    mean_squared_error = squared_error.mean(dim=dim)
    rmse_error = np.sqrt(mean_squared_error)

    squared_obs = obs**2
    mean_sqaured_obs = squared_obs.mean(dim=dim)
    rmse_obs = np.sqrt(mean_sqaured_obs)

    rmse = rmse_error / rmse_obs
    return rmse


observed_temperature_monthly_average = get_monthly_mean(observed_temp_ds['__xarray_dataarray_variable__'])
observed_temperature_anomaly = get_anomaly(observed_temp_ds, '__xarray_dataarray_variable__', observed_temperature_monthly_average)
observed_temperature_anomaly = observed_temperature_anomaly['__xarray_dataarray_variable___ANOMALY']

implicit_model_anomaly_ds = all_anomalies_ds["IMPLICIT"]
# ----- To check if the observed temperature anomaly dataset is going wrong...
# observed_temperature_anomaly_mean = observed_temperature_anomaly.mean(dim=['TIME'])
# vmin, vmax = -0.1, 0.1
# observed_temperature_anomaly_mean.plot(cmap='RdBu_r', vmin=vmin, vmax=vmax)
# plt.show()

fig, axes = plt.subplots(1, 1, figsize=(8,5))


scheme_name = "Implicit"
rmse_map = calculate_RMSE(observed_temperature_anomaly, implicit_model_anomaly_ds, dim='TIME')
rmse_weighted = calculate_RMSE_weighted(observed_temperature_anomaly, implicit_model_anomaly_ds, dim='TIME')
rmse_map_norm = calculate_RMSE_norm(observed_temperature_anomaly, implicit_model_anomaly_ds, dim='TIME')
print("rmse_weighted", rmse_weighted)

# Plotting
# ax = plt.subplot(3, 2, i + 1)
rmse_map.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Overall RMSE')
max_rmse = rmse_map.max().item()
print(scheme_name, 'Overall RMSE max', max_rmse)
plt.tight_layout()
fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
    f"INCLUDE_GEO_MEAN = {INCLUDE_GEO_MEAN}\n"
    f"INCLUDE_GEO_ANOM = {INCLUDE_GEO_ANOM}",
    ha='right', va='bottom', fontsize=8
)
plt.show()


fig, axes = plt.subplots(1, 1, figsize=(8,5))

rmse_map_norm.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Normalized RMSE')
max_rmse = rmse_map_norm.max().item()
print(scheme_name, 'Normalized RMSE max', max_rmse)
plt.tight_layout()
fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
    f"INCLUDE_GEO_MEAN = {INCLUDE_GEO_MEAN}\n"
    f"INCLUDE_GEO_ANOM = {INCLUDE_GEO_ANOM}",
    ha='right', va='bottom', fontsize=8
)
plt.show()

#%%

# Seasonal Analysis (Summer for the Northern Hemisphere)
summer_months_north_index = []
for i in range(13):
    summer_months_north = (17.5 + i*12, 18.5 + i*12, 19.5 + i*12)
    summer_months_north_index.extend(summer_months_north)



obs_summer_north_ds = observed_temperature_anomaly.sel(TIME=summer_months_north_index, method="nearest")
obs_summer_north_ds = obs_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

imp_summer_north_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_north_index, method="nearest")
imp_summer_north_ds = imp_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

# Summer months for the Southern Hemisphere 

summer_months_south_index = []
for i in range(13):
    summer_months_south = (11.5 + i*12, 12.5 + i*12, 13.5 +i*12)
    summer_months_south_index.extend(summer_months_south)


obs_summer_south_ds = observed_temperature_anomaly.sel(TIME=summer_months_south_index, method="nearest")
obs_summer_south_ds = obs_summer_south_ds.sel(LATITUDE=slice(-64.5,0))

imp_summer_south_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_south_index, method="nearest")
imp_summer_south_ds = imp_summer_south_ds.sel(LATITUDE=slice(-64.5, 0))
# Calculating RMSE for Summer season

rmse_summer_north = calculate_RMSE(obs_summer_north_ds, imp_summer_north_ds, dim="TIME")
rmse_summer_south = calculate_RMSE(obs_summer_south_ds, imp_summer_south_ds, dim="TIME")

rmse_summer = xr.concat([rmse_summer_south, rmse_summer_north], dim="LATITUDE")

fig, axes = plt.subplots(1, 1, figsize=(8,5))
rmse_summer.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Summer RMSE')
max_rmse = rmse_summer.max().item()
print(scheme_name, 'Summer RMSE max:', max_rmse)
max_rmse_location_summer = rmse_summer.where(rmse_summer == rmse_summer.max(), drop=True).squeeze()
print('max_rmse_location_summer: \n', max_rmse_location_summer)
min_rmse = rmse_summer.min().item()
print(scheme_name, 'Summer RMSE min:', min_rmse)
min_rmse_location_summer = rmse_summer.where(rmse_summer == rmse_summer.min(), drop=True).squeeze()
print('min_rmse_location_summer: \n', min_rmse_location_summer)
plt.tight_layout()
fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
    f"INCLUDE_GEO_MEAN = {INCLUDE_GEO_MEAN}\n"
    f"INCLUDE_GEO_ANOM = {INCLUDE_GEO_ANOM}",
    ha='right', va='bottom', fontsize=10
)
plt.show()

#%%

# Winter Seasonal Analysis 

winter_months_north_index = summer_months_south_index

obs_winter_north_ds = observed_temperature_anomaly.sel(TIME=winter_months_north_index, method="nearest")
obs_winter_north_ds = obs_winter_north_ds.sel(LATITUDE=slice(0,79.5))

imp_winter_north_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_north_index, method="nearest")
imp_winter_north_ds = imp_winter_north_ds.sel(LATITUDE=slice(0,79.5))

winter_months_south_index = summer_months_north_index

obs_winter_south_ds = observed_temperature_anomaly.sel(TIME=winter_months_south_index, method="nearest")
obs_winter_south_ds = obs_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))

imp_winter_south_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_south_index, method="nearest")
imp_winter_south_ds = imp_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))

rmse_winter_north = calculate_RMSE(obs_winter_north_ds, imp_winter_north_ds)
rmse_winter_south = calculate_RMSE(obs_winter_south_ds, imp_winter_south_ds)

rmse_winter = xr.concat([rmse_winter_south, rmse_winter_north], dim="LATITUDE")

fig, axes = plt.subplots(1, 1, figsize=(8,5))
rmse_winter.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Winter RMSE')
max_rmse = rmse_winter.max().item()
print(scheme_name, 'Winter RMSE max' , max_rmse)
max_rmse_location_winter = rmse_winter.where(rmse_winter == rmse_winter.max(), drop=True).squeeze()
print('max_rmse_location_winter', max_rmse_location_winter)
min_rmse = rmse_winter.min().item()
print(scheme_name, 'Winter RMSE min', min_rmse)
min_rmse_location_winter = rmse_winter.where(rmse_winter == rmse_winter.min(), drop=True).squeeze()
print('min_rmse_location_winter', min_rmse_location_winter)
plt.tight_layout()
fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
    f"INCLUDE_GEO_MEAN = {INCLUDE_GEO_MEAN}\n"
    f"INCLUDE_GEO_ANOM = {INCLUDE_GEO_ANOM}",
    ha='right', va='bottom', fontsize=10
)
plt.show()

#%% Time Series-----------------------------------------------------------------

# Win_min = dict(LATITUDE=45.0, LONGITUDE=-30.0)   # Northern Hemisphere
# Win_max = dict(LATITUDE=-40.0, LONGITUDE=60.0)   # Southern Hemisphere

Win_min = dict(LATITUDE=-52.5, LONGITUDE=-95.5)
Win_max = dict(LATITUDE=41.5, LONGITUDE=-50.5)

def extract_point_timeseries(ds, lat, lon):
    return ds.sel(
        LATITUDE=lat,
        LONGITUDE=lon,
        method="nearest"
    )

obs_NH = extract_point_timeseries(
    observed_temperature_anomaly,
    Win_min["LATITUDE"],
    Win_min["LONGITUDE"]
)

mod_NH = extract_point_timeseries(
    implicit_model_anomaly_ds,
    Win_min["LATITUDE"],
    Win_min["LONGITUDE"]
)

# Southern Hemisphere point
obs_SH = extract_point_timeseries(
    observed_temperature_anomaly,
    Win_max["LATITUDE"],
    Win_max["LONGITUDE"]
)

mod_SH = extract_point_timeseries(
    implicit_model_anomaly_ds,
    Win_max["LATITUDE"],
    Win_max["LONGITUDE"]
)

rmse_ts_NH = np.sqrt((mod_NH - obs_NH) ** 2)
rmse_ts_SH = np.sqrt((mod_SH - obs_SH) ** 2)

fig, ax = plt.subplots(figsize=(10, 5))

rmse_ts_NH.plot(
    ax=ax,
    label=f"P1 Win_min({Win_min['LATITUDE']}°, {Win_min['LONGITUDE']}°)",
    linewidth=2
)

rmse_ts_SH.plot(
    ax=ax,
    label=f"P2 Win_max({Win_max['LATITUDE']}°, {Win_max['LONGITUDE']}°)",
    linewidth=2
)

ax.set_title(f"{scheme_name} Scheme – Time Series RMSE at Selected Grid Points")
ax.set_xlabel("Time (months since Jan 2004)")
ax.set_ylabel("RMSE (K)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()