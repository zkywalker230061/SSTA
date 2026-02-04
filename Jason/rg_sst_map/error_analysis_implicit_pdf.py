#%%
# --- 1. Running Implicit Scheme ---------------------------------- 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from chris_utils import get_anomaly, coriolis_parameter
from utils_read_nc import get_monthly_mean, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.stats import kurtosis, skew, pearsonr

matplotlib.use('TkAgg')

USE_DOWNLOADED_SSH = False


INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC = False
INCLUDE_GEOSTROPHIC_DISPLACEMENT = False
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
GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_calculated.nc"
SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_interpolated_grad.nc"

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 30
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

surface_flux_da_centred_mean = get_monthly_mean(heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'])
surface_flux_da_anomaly = get_anomaly(heat_flux_anomaly_ds, 'NET_HEAT_FLUX_ANOMALY', surface_flux_da_centred_mean)
surface_flux_da_final = heat_flux_anomaly_ds["NET_HEAT_FLUX_ANOMALY_ANOMALY"]


print(surface_flux_da)
print(surface_flux_da_final)
print(f"Original Net Heat Flux Mean: {heat_flux_ds['NET_HEAT_FLUX'].mean().values}")
print(f"Double-Centered Mean: {surface_flux_da_final.mean().values}")

ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
ekman_anomaly_da = ekman_anomaly_ds['Q_Ek_anom']
ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

ekman_anomaly_da_centred_mean = get_monthly_mean(ekman_anomaly_ds["Q_Ek_anom"])
ekman_anomaly_da_final = get_anomaly(ekman_anomaly_ds, "Q_Ek_anom", ekman_anomaly_da_centred_mean)
ekman_anomaly_da_final = ekman_anomaly_da_final["Q_Ek_anom_ANOMALY"]



print(f"Original Mean Ekman: {ekman_anomaly_ds['Q_Ek_anom'].mean().values}")
print(f"Centered Mean Ekman: {ekman_anomaly_da_final.mean().values}")

# Overwrite off-centred anomalies to avoid changing variables in the simulation
surface_flux_da = surface_flux_da_final
ekman_anomaly_da = ekman_anomaly_da_final
ekman_anomaly_da = ekman_anomaly_da.fillna(0)

#%%

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD_PRESSURE"]

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
t_sub_da = t_sub_ds["T_sub_ANOMALY"]

entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']
print(entrainment_vel_da)

geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

if USE_DOWNLOADED_SSH:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_interpolated_grad.nc"
    ssh_var_name = "sla"
else:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_calculated_grad.nc"
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



# Testing drift before sim

components = {
    "Heat Flux Anomaly": surface_flux_da,
    "Ekman Anomaly": ekman_anomaly_da,
    "T Sub Anomaly": t_sub_da  
}

print("---Mean Anomaly Diagnostic----")

for name, da in components.items():
    mean_da = da.fillna(0).mean().values
    print(f"{name} Mean: {mean_da}")



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
        if INCLUDE_GEOSTROPHIC_DISPLACEMENT:    # then need to take the previous reading "back-propagated" based on current
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


        # Testing if mean anomalies = 0

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

        if INCLUDE_GEOSTROPHIC:
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


print(entrainment_vel_da.min().values)

#%%
#--- 2. Prepare Observed Temperature Anomaly --------------------

# Extract Variables
observed_temperature_monthly_average = get_monthly_mean(observed_temp_ds['__xarray_dataarray_variable__'])
observed_temperature_anomaly = get_anomaly(observed_temp_ds, '__xarray_dataarray_variable__', observed_temperature_monthly_average)
observed_temperature_anomaly = observed_temperature_anomaly['__xarray_dataarray_variable___ANOMALY']


#%%
#--- 3. Helper Functions ---------------------------------------------------

def get_clean_error_distribution(test_da, obs_da):
    """
    Calculates error (Test - Obs), slices time, and returns 
    a flattened numpy array with NaNs removed.
    """
    # 1. Align and Slice (Time slice 1:end)
    test_sliced = test_da.isel(TIME=slice(1, None))
    obs_sliced = obs_da.isel(TIME=slice(1, None))
    
    # 2. Calculate Error (xarray handles alignment automatically)
    error_da = test_sliced - obs_sliced
    
    # 3. Flatten and drop NaNs
    flat_error = error_da.values.flatten()
    return flat_error[~np.isnan(flat_error)]


def plot_pdf(err_data, name="Implicit"):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.kdeplot(err_data, bw_adjust=1.0, label=name, ax=ax)
    
    # 2. Calculate Percentiles
    # The middle 50% lies between the 25th and 75th percentiles
    q25, q75 = np.percentile(err_data, [25, 75])
    
    # 3. Add Shaded Region for Middle 50%
    # axvspan draws a vertical span (rectangle) across the plot
    ax.axvspan(q25, q75, color='green', alpha=0.2, label='Middle 50% (IQR)')
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5)

    ax.axvline(np.min(err_data), color='firebrick', linestyle='--', alpha=0.6, label=f'Min: {np.min(err_data):.2e}')
    ax.axvline(np.max(err_data), color='darkorange', linestyle='--', alpha=0.6, label=f'Max: {np.max(err_data):.2e}')
    
    # 4. Add Reference Lines
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

    # 5. Formatting & Annotations
    ax.set_title(f"{name} Scheme Error Distribution")
    ax.set_xlabel("Error (K)")
    ax.set_ylabel("Frequency")

    # Add text annotation to show the width of this region
    iqr_width = q75 - q25
    # Place text near the top of the range
    y_limits = ax.get_ylim()
    ax.text(5, y_limits[1]*0.5, f"IQR Width:\n{iqr_width:.3f} K", 
            horizontalalignment='center', color='darkgreen', fontweight='bold')
    ax.legend(loc='upper right', fontsize = 'x-small')

    #skewness and kurtosis
    err_data_skew = skew(err_data, axis=0, bias=True)
    err_data_kurt = kurtosis(err_data, axis=0, bias=True)
    ax.text(0.9,0.3, f"Skewness = {err_data_skew:.3f} \n Kurtosis = {err_data_kurt:.3f}")

    fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
    )

    plt.show()

def plot_log_pdf(err_data, name="Implicit"):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.kdeplot(err_data, bw_adjust=1.0, label=name, ax=ax)
    
    # 2. Calculate Percentiles
    # The middle 50% lies between the 25th and 75th percentiles
    q25, q75 = np.percentile(err_data, [25, 75])
    
    # 3. Add Shaded Region for Middle 50%
    # axvspan draws a vertical span (rectangle) across the plot
    ax.axvspan(q25, q75, color='green', alpha=0.2, label='Middle 50% (IQR)')
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5)

    ax.axvline(np.min(err_data), color='firebrick', linestyle='--', alpha=0.6, label=f'Min: {np.min(err_data):.2e}')
    ax.axvline(np.max(err_data), color='darkorange', linestyle='--', alpha=0.6, label=f'Max: {np.max(err_data):.2e}')
    
    # 4. Add Reference Lines
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

    # 5. Formatting & Annotations
    ax.set_title(f"{name} Scheme Error Distribution")
    ax.set_xlabel("Error (K)")
    ax.set_ylabel("Frequency")

    # Add text annotation to show the width of this region
    iqr_width = q75 - q25
    # Place text near the top of the range
    y_limits = ax.get_ylim()
    ax.text(5, y_limits[1]*0.5, f"IQR Width:\n{iqr_width:.3f} K", 
            horizontalalignment='center', color='darkgreen', fontweight='bold')
    ax.legend(loc='upper right', fontsize = 'x-small')

    #skewness and kurtosis
    err_data_skew = skew(err_data, axis=0, bias=True)
    err_data_kurt = kurtosis(err_data, axis=0, bias=True)
    ax.text(0.9,0.3, f"Skewness = {err_data_skew:.3f} \n Kurtosis = {err_data_kurt:.3f}")
    plt.xlim(-10, 10)        
    plt.yscale("log")      
    plt.ylim(1e-4, 1e2)
    plt.title("Implicit Scheme Error Distribution (Log-Scale)")
    plt.xlabel("Error (K)")
    plt.ylabel("Probability Density (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
    )

    plt.show()

#%% 
#--- 4. Getting Seasonal Datasets-------------------------------------------------

# summer_months_north_index = []
# for i in range(13):
#     summer_months_north = (17.5 + i*12, 18.5 + i*12, 19.5 + i*12)
#     summer_months_north_index.extend(summer_months_north)



# obs_summer_north_ds = observed_temperature_anomaly.sel(TIME=summer_months_north_index, method="nearest")
# obs_summer_north_ds = obs_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

# imp_summer_north_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_north_index, method="nearest")
# imp_summer_north_ds = imp_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

# # Summer months for the Southern Hemisphere 

# summer_months_south_index = []
# for i in range(13):
#     summer_months_south = (11.5 + i*12, 12.5 + i*12, 13.5 +i*12)
#     summer_months_south_index.extend(summer_months_south)


# obs_summer_south_ds = observed_temperature_anomaly.sel(TIME=summer_months_south_index, method="nearest")
# obs_summer_south_ds = obs_summer_south_ds.sel(LATITUDE=slice(-64.5,0))

# imp_summer_south_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_south_index, method="nearest")
# imp_summer_south_ds = imp_summer_south_ds.sel(LATITUDE=slice(-64.5, 0))


# # Winter Seasonal Analysis 

# winter_months_north_index = summer_months_south_index

# obs_winter_north_ds = observed_temperature_anomaly.sel(TIME=winter_months_north_index, method="nearest")
# obs_winter_north_ds = obs_winter_north_ds.sel(LATITUDE=slice(0,79.5))

# imp_winter_north_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_north_index, method="nearest")
# imp_winter_north_ds = imp_winter_north_ds.sel(LATITUDE=slice(0,79.5))

# winter_months_south_index = summer_months_north_index

# obs_winter_south_ds = observed_temperature_anomaly.sel(TIME=winter_months_south_index, method="nearest")
# obs_winter_south_ds = obs_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))

# imp_winter_south_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_south_index, method="nearest")
# imp_winter_south_ds = imp_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))


#%%

#--- 5. Full Distribution Error Analysis ---------------------------------------------

imp_error = get_clean_error_distribution(implicit_model_anomaly_ds, observed_temperature_anomaly)
print(imp_error)
plot_pdf(imp_error)
plot_log_pdf(imp_error)

#%%
#--- 6. Seasonal Distribution Error Analysis ---------------------------------------------------

# Dictionary to store processed error arrays
error_distributions = {}

print("Calculating errors...")

schemes = {
    "Summer (Northern Hemisphere)": imp_summer_north_ds,
    "Summer (Southern Hemisphere)": imp_summer_south_ds,
    "Winter (Northern Hemisphere)": imp_winter_north_ds,
    "Winter (Southern Hemisphere)": imp_winter_south_ds
}

observations = {
    "Summer (Northern Hemisphere)": obs_summer_north_ds,
    "Summer (Southern Hemisphere)": obs_summer_south_ds,
    "Winter (Northern Hemisphere)": obs_winter_north_ds,
    "Winter (Southern Hemisphere)": obs_winter_south_ds
}

for name, data in schemes.items():
    print(f"Processing {name}...")
    error_dataset = data - observations[name]
    flat_error = error_dataset.values.flatten()
    flat_error = flat_error[~np.isnan(flat_error)]
    error_distributions[name] = flat_error


# Create a figure with enough subplots
num_schemes = len(schemes)
cols = 2
rows = int(np.ceil(num_schemes / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows), constrained_layout=True)
axes_flat = axes.flatten()

for i, (name, err_data) in enumerate(error_distributions.items()):
    ax = axes_flat[i]
    
    # 1. Plot Histogram
    sns.kdeplot(err_data, bw_adjust=1.0, label=name, ax=ax)
    
    # 2. Calculate Percentiles
    # The middle 50% lies between the 25th and 75th percentiles
    q25, q75 = np.percentile(err_data, [25, 75])
    
    # 3. Add Shaded Region for Middle 50%
    # axvspan draws a vertical span (rectangle) across the plot
    ax.axvspan(q25, q75, color='green', alpha=0.2, label='Middle 50% (IQR)')
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5)

    ax.axvline(np.min(err_data), color='firebrick', linestyle='--', alpha=0.6, label=f'Min: {np.min(err_data):.2e}')
    ax.axvline(np.max(err_data), color='darkorange', linestyle='--', alpha=0.6, label=f'Max: {np.max(err_data):.2e}')
    
    # 4. Add Reference Lines
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

    # 5. Formatting & Annotations
    ax.set_title(f"{name} Scheme Error Distribution")
    ax.set_xlabel("Error (K)")
    ax.set_ylabel("Frequency")

    # Add text annotation to show the width of this region
    iqr_width = q75 - q25
    # Place text near the top of the range
    y_limits = ax.get_ylim()
    ax.text(5, y_limits[1]*0.5, f"IQR Width:\n{iqr_width:.3f} K", 
            horizontalalignment='center', color='darkgreen', fontweight='bold')
    ax.legend(loc='upper right', fontsize = 'x-small')

    #skewness and kurtosis
    err_data_skew = skew(err_data, axis=0, bias=True)
    err_data_kurt = kurtosis(err_data, axis=0, bias=True)
    ax.text(0.7,0.2, f"Skewness = {err_data_skew:.3f} \n Kurtosis = {err_data_kurt:.3f}")


# Turn off unused subplots
if len(axes_flat) > num_schemes:
    for j in range(num_schemes, len(axes_flat)):
        axes_flat[j].axis('off')

fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
)
plt.show()

#%%
#--- 7. Seasonal Distribution Error Analysis Combined PDF Plot (Log Scale) ---------------------------------

plt.figure(figsize=(12, 7))

for name, err_data in error_distributions.items():
    # KDE plot
    sns.kdeplot(err_data, bw_adjust=1, label=name)

# Formatting
plt.xlim(-7.5, 7.5)        
plt.yscale("log")      
plt.ylim(1e-4, 1e1)
plt.title("Comparison of Error Distributions (Log-Scale)")
plt.xlabel("Error (K)")
plt.ylabel("Probability Density (Log Scale)")
plt.legend()
plt.text(
    7.5, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=12
)
plt.show()



#%%
#--- 8. Seasonal Distribution Error Analysis (CDF) ------

plt.figure(figsize=(12, 7))

# Iterate through schemes and plot their CDF
for name, err_data in error_distributions.items():
    sns.ecdfplot(data=err_data, label=name, linewidth=2, alpha=0.5)

# Add reference lines
plt.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1, label="Zero Error")
plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label="50% Probability")

# Add Guidelines for the 25% - 75% probability zone
plt.axhline(0.25, color='gray', linestyle=':', alpha=0.5)
plt.axhline(0.75, color='gray', linestyle=':', alpha=0.5)
plt.text(plt.xlim()[0], 0.25, " 25%", verticalalignment='bottom', color='gray')
plt.text(plt.xlim()[0], 0.75, " 75%", verticalalignment='bottom', color='gray')

plt.axhspan(0.25, 0.75, color='grey' , alpha=0.2)

# Formatting
plt.title("Cumulative Distribution Function (CDF) of Seasonal Errors")
plt.xlabel("Error (K)")
plt.ylabel("Proportion")
plt.xlim(-3, 3) # Adjust this limit based on your data range
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(
    2, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
)
plt.show()

# %%
