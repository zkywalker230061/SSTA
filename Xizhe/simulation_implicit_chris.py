import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from chris_entrainment_vel_anomaly_forcing import entrainment_vel_anomaly_forcing
from chris_utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter, get_month_from_time
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
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

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_ENTRAINMENT = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True
# geostrophic displacement integral: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-3039/egusphere-2025-3039.pdf
USE_OTHER_MLD = False
USE_MAX_GRADIENT_METHOD = False
USE_LOG_FOR_ENTRAINMENT = False
rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15.0
g = 9.81

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION, OTHER_MLD=USE_OTHER_MLD, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, LOG_ENTRAINMENT_VELOCITY=USE_LOG_FOR_ENTRAINMENT)

"""Open all required datasets"""

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "datasets/files for simulate_implicit/heat_flux_interpolated_all_contributions.nc"
EKMAN_ANOMALY_DATA_PATH = "datasets/files for simulate_implicit/Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = "datasets/RG_ArgoClim_Temperature_2019.nc"
ENTRAINMENT_VEL_DATA_PATH = "datasets/files for simulate_implicit/Entrainment_Velocity-(2004-2018).nc"
GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "datasets/files for simulate_implicit/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "datasets/files for simulate_implicit/geostrophic_anomaly_calculated.nc"
EKMAN_MEAN_ADVECTION_DATA_PATH = "datasets/files for simulate_implicit/ekman_mean_advection.nc"
ENTRAINMENT_VEL_ANOMALY_FORCING_DATA_PATH = "datasets/files for simulate_implicit/entrainment_velocity_anomaly_forcing.nc"

if USE_OTHER_MLD:
    MLD_DATA_PATH = "datasets/files for simulate_implicit/other_h.nc"
    H_BAR_DATA_PATH = "datasets/files for simulate_implicit/other_h_bar.nc"
    T_SUB_DATA_PATH = "datasets/files for simulate_implicit/other_t_sub_anomaly.nc"

else:
    # H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
    # H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
    H_BAR_DATA_PATH = "datasets/New MLD & T_sub/hbar.nc"
    MLD_DATA_PATH = "datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
    T_SUB_DATA_PATH = "datasets/New MLD & T_sub/t_sub.nc"

if USE_MAX_GRADIENT_METHOD:
    T_SUB_DATA_PATH = "datasets/New_Entrainment/Tsub_Max_Gradient_Method_h.nc"
    ENTRAINMENT_VEL_DATA_PATH = "datasets/New_Entrainment/Entrainment_Vel_h.nc"
else:
    ENTRAINMENT_VEL_DATA_PATH =  ENTRAINMENT_VEL_DATA_PATH

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
hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

entrainment_vel_anomaly_forcing_ds = xr.open_dataset(ENTRAINMENT_VEL_ANOMALY_FORCING_DATA_PATH, decode_times=False)
entrainment_vel_anomaly_forcing = entrainment_vel_anomaly_forcing_ds["ENTRAINMENT_VEL_ANOMALY_FORCING"]

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
if USE_OTHER_MLD:
    t_sub_da = t_sub_ds["ANOMALY_SUB_TEMPERATURE"]
else:
    t_sub_da = t_sub_ds["SUB_TEMPERATURE"]

entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']


geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
SEA_SURFACE_GRAD_DATA_PATH = "datasets/files for simulate_implicit/sea_surface_calculated_grad.nc"
SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "datasets/files for simulate_implicit/sea_surface_monthly_mean_calculated_grad.nc"
ssh_var_name = "ssh"
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)
sea_surface_monthlymean_ds = xr.open_dataset(SEA_SURFACE_MONTHLY_MEAN_DATA_PATH, decode_times=False)

ekman_mean_advection = xr.open_dataset(EKMAN_MEAN_ADVECTION_DATA_PATH, decode_times=False)


def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60


delta_t = month_to_second(1)

model_anomalies = []
entrainment_fluxes = []
added_baseline = False
for month in heat_flux_anomaly_ds.TIME.values:
    if int(month) % 10 == 0:
        print("Month " + str(int(month)) + " of 180")
    # find the previous and current month from 1 to 12 to access the monthly-averaged data (hbar, entrainment vel.)
    prev_month = month - 1
    month_in_year = get_month_from_time(month)
    prev_month_in_year = get_month_from_time(month - 1)
    if not added_baseline:  # just adds the baseline of a whole bunch of zero
        base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY'] - \
               temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
        base = base.expand_dims(TIME=[month])
        model_anomalies.append(base)
        added_baseline = True
    else:
        prev_anomaly = model_anomalies[-1].isel(TIME=-1)

        """Mean advection"""
        if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
            # initialise alpha, beta (x, y current velocities) as 0
            # alpha = sea_surface_monthlymean_ds['alpha'].sel(MONTH=prev_month_in_year) - sea_surface_monthlymean_ds['alpha'].sel(MONTH=prev_month_in_year)
            # beta = sea_surface_monthlymean_ds['beta'].sel(MONTH=prev_month_in_year) - sea_surface_monthlymean_ds['beta'].sel(MONTH=prev_month_in_year)
            alpha = xr.zeros_like(sea_surface_monthlymean_ds['alpha'].sel(MONTH=prev_month_in_year))
            beta = xr.zeros_like(sea_surface_monthlymean_ds['beta'].sel(MONTH=prev_month_in_year))

            if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
                alpha += sea_surface_monthlymean_ds['alpha'].sel(MONTH=prev_month_in_year)
                beta += sea_surface_monthlymean_ds['beta'].sel(MONTH=prev_month_in_year)

            if INCLUDE_EKMAN_MEAN_ADVECTION:
                alpha = alpha + ekman_mean_advection["ekman_alpha"].sel(MONTH=prev_month_in_year)
                beta = beta + ekman_mean_advection["ekman_beta"].sel(MONTH=prev_month_in_year)

            # calculate mean advection contributions
            earth_radius = 6371000
            latitudes = np.deg2rad(sea_surface_monthlymean_ds['LATITUDE'])  # any ds to get latitude
            dx = (2 * np.pi * earth_radius / 360) * np.cos(latitudes)
            dy = (2 * np.pi * earth_radius / 360) * np.ones_like(latitudes)
            dx = xr.DataArray(dx, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values}, dims=['LATITUDE'])  # convert dx, dy to xarray for use below
            dy = xr.DataArray(dy, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values}, dims=['LATITUDE'])

            CFL_x = (abs(alpha) * delta_t / dx).max()
            CFL_y = (abs(beta) * delta_t / dy).max()
            CFL_max = max(float(CFL_x), float(CFL_y))
            substeps = int(np.ceil(CFL_max)) + 1  # require CFL<1 for stability
            sub_dt = delta_t / substeps

            tm_div_total = xr.zeros_like(prev_anomaly)
            for step in range(substeps):
                # if step % 25 == 0:
                #     print("Step " + str(step) + " of " + str(substeps))
                # get upwind flux
                prev_anom_east = prev_anomaly.shift(LONGITUDE=-1)
                prev_anom_west = prev_anomaly.shift(LONGITUDE=1)
                prev_anom_north = prev_anomaly.shift(LATITUDE=-1)
                prev_anom_south = prev_anomaly.shift(LATITUDE=1)

                # get alpha/beta at the edges of gridboxes
                alpha_east = (alpha + alpha.shift(LONGITUDE=-1)) / 2
                alpha_west = (alpha + alpha.shift(LONGITUDE=1)) / 2
                beta_north = (beta + beta.shift(LATITUDE=-1)) / 2
                beta_south = (beta + beta.shift(LATITUDE=1)) / 2

                # check for nans in neighbouring cells
                ocean_mask = ~prev_anomaly.isnull()
                has_east_ocean = ~prev_anom_east.isnull()
                has_west_ocean = ~prev_anom_west.isnull()
                has_north_ocean = ~prev_anom_north.isnull()
                has_south_ocean = ~prev_anom_south.isnull()

                # get upwind parts
                F_east = xr.where(alpha_east < 0, -alpha_east * prev_anomaly, -alpha_east * prev_anom_east)
                F_west = xr.where(alpha_west < 0, -alpha_west * prev_anom_west, -alpha_west * prev_anomaly)
                G_north = xr.where(beta_north > 0, beta_north * prev_anomaly, beta_north * prev_anom_north)
                G_south = xr.where(beta_south > 0, beta_south * prev_anom_south, beta_south * prev_anomaly)

                # ignore flux if advecting from land
                F_east = xr.where(has_east_ocean, F_east, 0)
                F_west = xr.where(has_west_ocean, F_west, 0)
                G_north = xr.where(has_north_ocean, G_north, 0)
                G_south = xr.where(has_south_ocean, G_south, 0)

                # ignore flux if current cell is land
                F_east = xr.where(ocean_mask, F_east, 0)
                F_west = xr.where(ocean_mask, F_west, 0)
                G_north = xr.where(ocean_mask, G_north, 0)
                G_south = xr.where(ocean_mask, G_south, 0)

                # get flux divergence
                tm_div = (F_east - F_west) / dx + (G_north - G_south) / dy
                tm_div_total += tm_div

                # update working temperature for the substep and apply ocean mask again
                prev_anomaly = prev_anomaly - sub_dt * tm_div
                prev_anomaly = prev_anomaly.where(ocean_mask)
            tm_div = tm_div_total / substeps

        # reset prev_anomaly in case it was adjusted by the mean advection
        prev_anomaly = model_anomalies[-1].isel(TIME=-1)

        # get previous data
        # prev_tsub_anom = t_sub_da.sel(TIME=prev_month)
        # prev_heat_flux_anom = surface_flux_da.sel(TIME=prev_month)
        # prev_ekman_anom = ekman_anomaly_da.sel(TIME=prev_month)
        # prev_entrainment_vel = entrainment_vel_da.sel(MONTH=prev_month_in_year)
        # prev_geo_anom = geostrophic_anomaly_da.sel(TIME=prev_month)
        prev_hbar = hbar_da.sel(MONTH=prev_month_in_year)
        # prev_entrainment_vel_anomaly_forcing = entrainment_vel_anomaly_forcing.sel(TIME=prev_month)

        # get current data
        cur_tsub_anom = t_sub_da.sel(TIME=month)
        cur_heat_flux_anom = surface_flux_da.sel(TIME=month)
        cur_ekman_anom = ekman_anomaly_da.sel(TIME=month)
        cur_entrainment_vel = entrainment_vel_da.sel(MONTH=month_in_year)
        cur_geo_anom = geostrophic_anomaly_da.sel(TIME=month)
        cur_hbar = hbar_da.sel(MONTH=month_in_year)
        cur_entrainment_vel_anomaly_forcing = entrainment_vel_anomaly_forcing.sel(TIME=month)

        # static forcings = surface flux + Ekman anomolous advection + part of entrainment
        cur_static_forcings = xr.zeros_like(cur_ekman_anom)
        #prev_static_forcings = xr.zeros_like(prev_ekman_anom)

        if INCLUDE_SURFACE:
            cur_static_forcings += cur_heat_flux_anom
            #prev_static_forcings += prev_heat_flux_anom

        if INCLUDE_EKMAN_ANOM_ADVECTION:
            cur_static_forcings += cur_ekman_anom
            #prev_static_forcings += prev_ekman_anom

        if INCLUDE_ENTRAINMENT:
            if USE_LOG_FOR_ENTRAINMENT:
                cur_static_forcings += cur_hbar / delta_t * np.log(cur_hbar / prev_hbar) * cur_tsub_anom * rho_0 * c_0
                #prev_static_forcings += #to think about
            else:
                cur_static_forcings += cur_entrainment_vel * cur_tsub_anom * rho_0 * c_0
                #prev_static_forcings += prev_entrainment_vel * prev_tsub_anom * rho_0 * c_0

        if INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING:
            cur_static_forcings += cur_entrainment_vel_anomaly_forcing
            #prev_static_forcings += prev_entrainment_vel_anomaly_forcing

        if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
            cur_static_forcings += cur_geo_anom
            #prev_static_forcings += prev_geo_anom

        # build model components
        cur_b = cur_static_forcings / (rho_0 * c_0 * cur_hbar)
        #prev_b = prev_static_forcings / (rho_0 * c_0 * prev_hbar)

        cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)
        #prev_a = gamma_0 / (rho_0 * c_0 * prev_hbar)
        if INCLUDE_ENTRAINMENT:
            cur_a += cur_entrainment_vel / cur_hbar
            #prev_a += prev_entrainment_vel / prev_hbar
        if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
            cur_a += (- sea_surface_monthlymean_ds["alpha_grad_long"].sel(MONTH=month_in_year) + sea_surface_monthlymean_ds["beta_grad_lat"].sel(MONTH=month_in_year)).clip(-1e-7, 1e-7)
            #prev_a += (- sea_surface_monthlymean_ds["alpha_grad_long"].sel(MONTH=prev_month_in_year) + sea_surface_monthlymean_ds["beta_grad_lat"].sel(MONTH=prev_month_in_year)).clip(-1e-7, 1e-7)
        if INCLUDE_EKMAN_MEAN_ADVECTION:
            cur_a += (- ekman_mean_advection["ekman_alpha_grad_long"].sel(MONTH=month_in_year) + ekman_mean_advection["ekman_beta_grad_lat"].sel(MONTH=month_in_year)).clip(-1e-7, 1e-7)
            #prev_a += (- ekman_mean_advection["ekman_alpha_grad_long"].sel(MONTH=prev_month_in_year) + ekman_mean_advection["ekman_beta_grad_lat"].sel(MONTH=prev_month_in_year)).clip(-1e-7, 1e-7)

        # update anomaly
        cur_anomaly = (prev_anomaly + delta_t * cur_b) / (1 + delta_t * cur_a)

        if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
            cur_anomaly = cur_anomaly - tm_div * delta_t
            cur_anomaly = cur_anomaly.where(ocean_mask, cur_anomaly)

        cur_anomaly = cur_anomaly.drop_vars('MONTH', errors='ignore')
        cur_anomaly = cur_anomaly.expand_dims(TIME=[month])
        model_anomalies.append(cur_anomaly)

        # calculate flux due to entrainment
        if INCLUDE_ENTRAINMENT:
            entrainment_flux = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_anomaly)
            entrainment_fluxes.append(entrainment_flux)

# concatenate into dataset
model_anomalies_ds = xr.concat(model_anomalies, 'TIME')
model_anomalies_ds = model_anomalies_ds.rename("IMPLICIT")
model_anomalies_ds = model_anomalies_ds.to_dataset(name="IMPLICIT")


# remove whatever seasonal cycle remains
monthly_mean = get_monthly_mean(model_anomalies_ds["IMPLICIT"])
model_anomalies_ds["IMPLICIT"] = get_anomaly(model_anomalies_ds, "IMPLICIT", monthly_mean)["IMPLICIT_ANOMALY"]
model_anomalies_ds = model_anomalies_ds.drop_vars("IMPLICIT_ANOMALY")

# save
model_anomalies_ds = remove_empty_attributes(model_anomalies_ds) # when doing the seasonality removal, some units are None
model_anomalies_ds.to_netcdf("datasets/implicit_model/" + save_name + ".nc")

# save contributions from each component
if INCLUDE_ENTRAINMENT:
    entrainment_fluxes = xr.concat(entrainment_fluxes, 'TIME')
    entrainment_fluxes = entrainment_fluxes.drop_vars(["MONTH", "PRESSURE"])
    entrainment_fluxes = entrainment_fluxes.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_fluxes = entrainment_fluxes.rename("ENTRAINMENT_ANOMALY")

# merge the relevant fluxes into a single dataset
flux_components_to_merge = []
variable_names = []
if INCLUDE_SURFACE:
    surface_flux_da = surface_flux_da.rename("SURFACE_FLUX_ANOMALY")
    flux_components_to_merge.append(surface_flux_da)
    variable_names.append("SURFACE_FLUX_ANOMALY")

if INCLUDE_EKMAN_ANOM_ADVECTION:
    ekman_anomaly_da = ekman_anomaly_da.rename("EKMAN_ANOM_ADVECTION_ANOMALY")
    flux_components_to_merge.append(ekman_anomaly_da)
    variable_names.append("EKMAN_ANOM_ADVECTION_ANOMALY")

if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
    geostrophic_anomaly_da = geostrophic_anomaly_da.rename("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY")
    flux_components_to_merge.append(geostrophic_anomaly_da)
    variable_names.append("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY")

if INCLUDE_ENTRAINMENT:
    flux_components_to_merge.append(entrainment_fluxes)
    variable_names.append("ENTRAINMENT_ANOMALY")

flux_components_ds = xr.merge(flux_components_to_merge)

# remove whatever seasonal cycle may remain from the components
for variable_name in variable_names:
    monthly_mean = get_monthly_mean(flux_components_ds[variable_name])
    flux_components_ds[variable_name] = get_anomaly(flux_components_ds, variable_name, monthly_mean)[variable_name + "_ANOMALY"]
    flux_components_ds = flux_components_ds.drop_vars(variable_name + "_ANOMALY")

# save flux components
flux_components_ds = remove_empty_attributes(flux_components_ds)
flux_components_ds.to_netcdf("datasets/implicit_model/" + save_name + "_flux_components.nc")

#make_movie(model_anomalies_ds["IMPLICIT"], -3, 3)