import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('TkAgg')

"""
Naming scheme:
Surface, Ekman, Entrainment, Geostrophic in that order. 0 if off, 1 if on
e.g.
1001
=> surface and geostrophic on, Ekman and entrainment off
"""

INCLUDE_SURFACE = False
INCLUDE_EKMAN = False
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC = False
INCLUDE_GEOSTROPHIC_DISPLACEMENT = False
# geostrophic displacement integral: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-3039/egusphere-2025-3039.pdf
CLEAN_CHRIS_PREV_CUR = True        # only really useful when entrainment is turned on
USE_DOWNLOADED_SSH = False
rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 0.0
g = 9.81

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC, USE_DOWNLOADED_SSH, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_DISPLACEMENT)

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/heat_flux_interpolated_all_contributions.nc"
EKMAN_ANOMALY_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Entrainment_Velocity-(2004-2018).nc"
#H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
T_SUB_DENOISED_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub_denoised.nc"
GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/geostrophic_anomaly_calculated.nc"

temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)

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
    SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc"
    ssh_var_name = "sla"
else:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"
    ssh_var_name = "ssh"
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)


def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60


delta_t = month_to_second(1)

# initialise lists for temperature anomalies for each model
chris_prev_cur_model_anomalies = []
chris_mean_k_model_anomalies = []
chris_prev_k_model_anomalies = []
chris_capped_exponent_model_anomalies = []
explicit_model_anomalies = []
implicit_model_anomalies = []
semi_implicit_model_anomalies = []

# initialise lists for entrainment fluxes for each model; for categorising each component
entrainment_fluxes_prev_cur = []
entrainment_fluxes_mean_k = []
entrainment_fluxes_prev_k = []
entrainment_fluxes_capped_exponent = []
entrainment_fluxes_explicit = []
entrainment_fluxes_implicit = []
entrainment_fluxes_semi_implicit = []

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
        chris_prev_cur_model_anomalies.append(base)
        chris_mean_k_model_anomalies.append(base)
        chris_prev_k_model_anomalies.append(base)
        chris_capped_exponent_model_anomalies.append(base)
        explicit_model_anomalies.append(base)
        implicit_model_anomalies.append(base)
        semi_implicit_model_anomalies.append(base)
        added_baseline = True

    else:
        # store previous readings Tm(n-1)
        if INCLUDE_GEOSTROPHIC_DISPLACEMENT:    # then need to take the previous reading "back-propagated" based on current
            prev_chris_prev_cur_tm_anom_at_cur_loc = chris_prev_cur_model_anomalies[-1].isel(TIME=-1)
            prev_chris_mean_k_tm_anom_at_cur_loc = chris_mean_k_model_anomalies[-1].isel(TIME=-1)
            prev_chris_prev_k_tm_anom_at_cur_loc = chris_prev_k_model_anomalies[-1].isel(TIME=-1)
            prev_chris_capped_exponent_k_tm_anom_at_cur_loc = chris_capped_exponent_model_anomalies[-1].isel(TIME=-1)
            prev_explicit_k_tm_anom_at_cur_loc = explicit_model_anomalies[-1].isel(TIME=-1)
            prev_implicit_k_tm_anom_at_cur_loc = implicit_model_anomalies[-1].isel(TIME=-1)
            prev_semi_implicit_k_tm_anom_at_cur_loc = semi_implicit_model_anomalies[-1].isel(TIME=-1)

            f = coriolis_parameter(sea_surface_grad_ds['LATITUDE']).broadcast_like(sea_surface_grad_ds[ssh_var_name]).broadcast_like(sea_surface_grad_ds[ssh_var_name + '_anomaly_grad_long'])  # broadcasting based on Jason/Julia's usage
            alpha = g / f.sel(TIME=month) * sea_surface_grad_ds[ssh_var_name + '_anomaly_grad_long'].sel(TIME=month)
            beta = g / f.sel(TIME=month) * sea_surface_grad_ds[ssh_var_name + '_anomaly_grad_lat'].sel(TIME=month)
            back_x = sea_surface_grad_ds['LONGITUDE'] + alpha * month_to_second(1)      # just need a list of long/lat
            back_y = sea_surface_grad_ds['LATITUDE'] - beta * month_to_second(1)        # ss_grad is a useful dummy for that

            # interpolate to the "back-propagated" x and y position, but if that turns out nan (due to coastline), then
            # just use the temperature at current position. BC == "coast buffer"
            prev_chris_prev_cur_tm_anom = prev_chris_prev_cur_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_prev_cur_tm_anom_at_cur_loc)
            prev_chris_mean_k_tm_anom = prev_chris_mean_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_mean_k_tm_anom_at_cur_loc)
            prev_chris_prev_k_tm_anom = prev_chris_prev_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_prev_k_tm_anom_at_cur_loc)
            prev_chris_capped_exponent_k_tm_anom = prev_chris_capped_exponent_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_chris_capped_exponent_k_tm_anom_at_cur_loc)
            prev_explicit_k_tm_anom = prev_explicit_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_explicit_k_tm_anom_at_cur_loc)
            prev_implicit_k_tm_anom = prev_implicit_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_implicit_k_tm_anom_at_cur_loc)
            prev_semi_implicit_k_tm_anom = prev_semi_implicit_k_tm_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_semi_implicit_k_tm_anom_at_cur_loc)
        else:
            prev_chris_prev_cur_tm_anom = chris_prev_cur_model_anomalies[-1].isel(TIME=-1)
            prev_chris_mean_k_tm_anom = chris_mean_k_model_anomalies[-1].isel(TIME=-1)
            prev_chris_prev_k_tm_anom = chris_prev_k_model_anomalies[-1].isel(TIME=-1)
            prev_chris_capped_exponent_k_tm_anom = chris_capped_exponent_model_anomalies[-1].isel(TIME=-1)
            prev_explicit_k_tm_anom = explicit_model_anomalies[-1].isel(TIME=-1)
            prev_implicit_k_tm_anom = implicit_model_anomalies[-1].isel(TIME=-1)
            prev_semi_implicit_k_tm_anom = semi_implicit_model_anomalies[-1].isel(TIME=-1)

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

        exponent_prev_cur = prev_k * month_to_second(prev_month) - cur_k * month_to_second(month)
        exponent_mean_k = -0.5 * (prev_k + cur_k) * delta_t
        exponent_prev_k = prev_k * month_to_second(prev_month) - prev_k * month_to_second(month)
        exponent_capped = exponent_prev_cur.where(exponent_prev_cur <= 0, 0)

        # update anomalies
        if INCLUDE_ENTRAINMENT:
            cur_chris_prev_cur_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_prev_cur_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_prev_cur)
            cur_chris_mean_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_mean_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_mean_k)
            cur_chris_prev_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_prev_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_prev_k)
            cur_chris_capped_exponent_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_capped_exponent_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_capped)
        else:
            cur_chris_prev_cur_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_prev_cur_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_prev_cur)
            cur_chris_mean_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_mean_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_mean_k)
            cur_chris_prev_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_prev_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_prev_k)
            cur_chris_capped_exponent_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_capped_exponent_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_capped)

        cur_explicit_k_tm_anom = prev_explicit_k_tm_anom + delta_t * (prev_b - prev_a * prev_explicit_k_tm_anom)
        cur_implicit_k_tm_anom = (prev_implicit_k_tm_anom + delta_t * cur_b) / (1 + delta_t * cur_a)
        cur_semi_implicit_k_tm_anom = (prev_semi_implicit_k_tm_anom + delta_t * prev_b) / (1 + delta_t * cur_a)

        # reformat and save each model
        cur_chris_prev_cur_tm_anom = cur_chris_prev_cur_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_prev_cur_tm_anom = cur_chris_prev_cur_tm_anom.expand_dims(TIME=[month])
        chris_prev_cur_model_anomalies.append(cur_chris_prev_cur_tm_anom)

        cur_chris_mean_k_tm_anom = cur_chris_mean_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_mean_k_tm_anom = cur_chris_mean_k_tm_anom.expand_dims(TIME=[month])
        chris_mean_k_model_anomalies.append(cur_chris_mean_k_tm_anom)

        cur_chris_prev_k_tm_anom = cur_chris_prev_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_prev_k_tm_anom = cur_chris_prev_k_tm_anom.expand_dims(TIME=[month])
        chris_prev_k_model_anomalies.append(cur_chris_prev_k_tm_anom)

        cur_chris_capped_exponent_k_tm_anom = cur_chris_capped_exponent_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_capped_exponent_k_tm_anom = cur_chris_capped_exponent_k_tm_anom.expand_dims(TIME=[month])
        chris_capped_exponent_model_anomalies.append(cur_chris_capped_exponent_k_tm_anom)

        cur_explicit_k_tm_anom = cur_explicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_explicit_k_tm_anom = cur_explicit_k_tm_anom.expand_dims(TIME=[month])
        explicit_model_anomalies.append(cur_explicit_k_tm_anom)

        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.expand_dims(TIME=[month])
        implicit_model_anomalies.append(cur_implicit_k_tm_anom)

        cur_semi_implicit_k_tm_anom = cur_semi_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_semi_implicit_k_tm_anom = cur_semi_implicit_k_tm_anom.expand_dims(TIME=[month])
        semi_implicit_model_anomalies.append(cur_semi_implicit_k_tm_anom)

        # get entrainment flux components; for categorising each component
        if INCLUDE_ENTRAINMENT:
            entrainment_flux_prev_cur = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_prev_cur_tm_anom)
            entrainment_fluxes_prev_cur.append(entrainment_flux_prev_cur)

            entrainment_flux_mean_k = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_mean_k_tm_anom)
            entrainment_fluxes_mean_k.append(entrainment_flux_mean_k)

            entrainment_flux_prev_k = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_prev_k_tm_anom)
            entrainment_fluxes_prev_k.append(entrainment_flux_prev_k)

            entrainment_flux_capped_exponent = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_capped_exponent_k_tm_anom)
            entrainment_fluxes_capped_exponent.append(entrainment_flux_capped_exponent)

            entrainment_flux_explicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_explicit_k_tm_anom)
            entrainment_fluxes_explicit.append(entrainment_flux_explicit)

            entrainment_flux_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_implicit_k_tm_anom)
            entrainment_fluxes_implicit.append(entrainment_flux_implicit)

            entrainment_flux_semi_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_semi_implicit_k_tm_anom)
            entrainment_fluxes_semi_implicit.append(entrainment_flux_semi_implicit)


# concatenate anomalies into a ds
chris_prev_cur_model_anomaly_ds = xr.concat(chris_prev_cur_model_anomalies, 'TIME')
chris_mean_k_model_anomaly_ds = xr.concat(chris_mean_k_model_anomalies, 'TIME')
chris_prev_k_model_anomaly_ds = xr.concat(chris_prev_k_model_anomalies, 'TIME')
chris_capped_exponent_model_anomaly_ds = xr.concat(chris_capped_exponent_model_anomalies, 'TIME')
explicit_model_anomaly_ds = xr.concat(explicit_model_anomalies, 'TIME')
implicit_model_anomaly_ds = xr.concat(implicit_model_anomalies, 'TIME')
semi_implicit_model_anomaly_ds = xr.concat(semi_implicit_model_anomalies, 'TIME')

# rename all models
chris_prev_cur_model_anomaly_ds = chris_prev_cur_model_anomaly_ds.rename("CHRIS_PREV_CUR")
chris_mean_k_model_anomaly_ds = chris_mean_k_model_anomaly_ds.rename("CHRIS_MEAN_K")
chris_prev_k_model_anomaly_ds = chris_prev_k_model_anomaly_ds.rename("CHRIS_PREV_K")
chris_capped_exponent_model_anomaly_ds = chris_capped_exponent_model_anomaly_ds.rename("CHRIS_CAPPED_EXPONENT")
explicit_model_anomaly_ds = explicit_model_anomaly_ds.rename("EXPLICIT")
implicit_model_anomaly_ds = implicit_model_anomaly_ds.rename("IMPLICIT")
semi_implicit_model_anomaly_ds = semi_implicit_model_anomaly_ds.rename("SEMI_IMPLICIT")

# combine to a single ds
all_anomalies_ds = xr.merge([chris_prev_cur_model_anomaly_ds, chris_mean_k_model_anomaly_ds, chris_prev_k_model_anomaly_ds, chris_capped_exponent_model_anomaly_ds, explicit_model_anomaly_ds, implicit_model_anomaly_ds, semi_implicit_model_anomaly_ds])

# remove whatever seasonal cycle remains
model_names = ["CHRIS_PREV_CUR", "CHRIS_MEAN_K", "CHRIS_PREV_K", "CHRIS_CAPPED_EXPONENT", "EXPLICIT", "IMPLICIT", "SEMI_IMPLICIT"]
for variable_name in model_names:
    monthly_mean = get_monthly_mean(all_anomalies_ds[variable_name])
    all_anomalies_ds[variable_name] = get_anomaly(all_anomalies_ds, variable_name, monthly_mean)[variable_name + "_ANOMALY"]
    all_anomalies_ds = all_anomalies_ds.drop_vars(variable_name + "_ANOMALY")

# clean up prev_cur model
if CLEAN_CHRIS_PREV_CUR:
    all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = all_anomalies_ds["CHRIS_PREV_CUR"].where((all_anomalies_ds["CHRIS_PREV_CUR"] > -10) & (all_anomalies_ds["CHRIS_PREV_CUR"] < 10))
    n_modes = 20
    monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
    map_mask = temperature_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5)
    eof_ds, variance, PCs, EOFs = get_eof_with_nan_consideration(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"], map_mask, modes=n_modes, monthly_mean_ds=None, tolerance=1e-2)
    all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = eof_ds.rename("CHRIS_PREV_CUR_CLEAN")
    chris_prev_cur_clean_monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
    all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = get_anomaly(all_anomalies_ds, "CHRIS_PREV_CUR_CLEAN", chris_prev_cur_clean_monthly_mean)["CHRIS_PREV_CUR_CLEAN_ANOMALY"]
    all_anomalies_ds = all_anomalies_ds.drop_vars("CHRIS_PREV_CUR_CLEAN_ANOMALY")

# save
all_anomalies_ds = remove_empty_attributes(all_anomalies_ds) # when doing the seasonality removal, some units are None
all_anomalies_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name + ".nc")

# format entrainment flux datasets
if INCLUDE_ENTRAINMENT:
    entrainment_flux_prev_cur_ds = xr.concat(entrainment_fluxes_prev_cur, 'TIME')
    entrainment_flux_prev_cur_ds = entrainment_flux_prev_cur_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_prev_cur_ds = entrainment_flux_prev_cur_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_prev_cur_ds = entrainment_flux_prev_cur_ds.rename("ENTRAINMENT_FLUX_PREV_CUR_ANOMALY")

    entrainment_flux_mean_k_ds = xr.concat(entrainment_fluxes_mean_k, 'TIME')
    entrainment_flux_mean_k_ds = entrainment_flux_mean_k_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_mean_k_ds = entrainment_flux_mean_k_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_mean_k_ds = entrainment_flux_mean_k_ds.rename("ENTRAINMENT_FLUX_MEAN_K_ANOMALY")

    entrainment_flux_prev_k_ds = xr.concat(entrainment_fluxes_prev_k, 'TIME')
    entrainment_flux_prev_k_ds = entrainment_flux_prev_k_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_prev_k_ds = entrainment_flux_prev_k_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_prev_k_ds = entrainment_flux_prev_k_ds.rename("ENTRAINMENT_FLUX_PREV_K_ANOMALY")

    entrainment_flux_capped_exponent_ds = xr.concat(entrainment_fluxes_capped_exponent, 'TIME')
    entrainment_flux_capped_exponent_ds = entrainment_flux_capped_exponent_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_capped_exponent_ds = entrainment_flux_capped_exponent_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_capped_exponent_ds = entrainment_flux_capped_exponent_ds.rename("ENTRAINMENT_FLUX_CAPPED_EXPONENT_ANOMALY")

    entrainment_flux_explicit_ds = xr.concat(entrainment_fluxes_explicit, 'TIME')
    entrainment_flux_explicit_ds = entrainment_flux_explicit_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_explicit_ds = entrainment_flux_explicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_explicit_ds = entrainment_flux_explicit_ds.rename("ENTRAINMENT_FLUX_EXPLICIT_ANOMALY")

    entrainment_flux_implicit_ds = xr.concat(entrainment_fluxes_implicit, 'TIME')
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.rename("ENTRAINMENT_FLUX_IMPLICIT_ANOMALY")

    entrainment_flux_semi_implicit_ds = xr.concat(entrainment_fluxes_semi_implicit, 'TIME')
    entrainment_flux_semi_implicit_ds = entrainment_flux_semi_implicit_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_semi_implicit_ds = entrainment_flux_semi_implicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_semi_implicit_ds = entrainment_flux_semi_implicit_ds.rename("ENTRAINMENT_FLUX_SEMI_IMPLICIT_ANOMALY")


# merge the relevant fluxes into a single dataset
flux_components_to_merge = []
variable_names = []
if INCLUDE_SURFACE:
    surface_flux_da = surface_flux_da.rename("SURFACE_FLUX_ANOMALY")
    flux_components_to_merge.append(surface_flux_da)
    variable_names.append("SURFACE_FLUX_ANOMALY")
if INCLUDE_EKMAN:
    ekman_anomaly_da = ekman_anomaly_da.rename("EKMAN_FLUX_ANOMALY")
    flux_components_to_merge.append(ekman_anomaly_da)
    variable_names.append("EKMAN_FLUX_ANOMALY")
if INCLUDE_GEOSTROPHIC:
    geostrophic_anomaly_da = geostrophic_anomaly_da.rename("GEOSTROPHIC_FLUX_ANOMALY")
    flux_components_to_merge.append(geostrophic_anomaly_da)
    variable_names.append("GEOSTROPHIC_FLUX_ANOMALY")
if INCLUDE_ENTRAINMENT:
    flux_components_to_merge.append(entrainment_flux_prev_cur_ds)
    flux_components_to_merge.append(entrainment_flux_mean_k_ds)
    flux_components_to_merge.append(entrainment_flux_prev_k_ds)
    flux_components_to_merge.append(entrainment_flux_capped_exponent_ds)
    flux_components_to_merge.append(entrainment_flux_explicit_ds)
    flux_components_to_merge.append(entrainment_flux_implicit_ds)
    flux_components_to_merge.append(entrainment_flux_semi_implicit_ds)
    variable_names.append("ENTRAINMENT_FLUX_PREV_CUR_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_MEAN_K_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_PREV_K_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_CAPPED_EXPONENT_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_EXPLICIT_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_IMPLICIT_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_SEMI_IMPLICIT_ANOMALY")

flux_components_ds = xr.merge(flux_components_to_merge)

# remove whatever seasonal cycle may remain from the components
for variable_name in variable_names:
    monthly_mean = get_monthly_mean(flux_components_ds[variable_name])
    flux_components_ds[variable_name] = get_anomaly(flux_components_ds, variable_name, monthly_mean)[variable_name + "_ANOMALY"]
    flux_components_ds = flux_components_ds.drop_vars(variable_name + "_ANOMALY")

flux_components_ds = remove_empty_attributes(flux_components_ds)
# print(flux_components_ds)

flux_components_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name + "_flux_components.nc")

# make_movie(all_anomalies_ds["EXPLICIT"], -5, 5)
# make_movie(all_anomalies_ds["IMPLICIT"], -2, 2)
# make_movie(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"], -5, 5)
# make_movie(all_anomalies_ds["CHRIS_MEAN_K"], -5, 5)



