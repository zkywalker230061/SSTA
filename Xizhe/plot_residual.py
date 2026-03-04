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
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, get_month_from_time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# ----------------------------------------------------------------------------
# 0) Inputs (use your existing objects from the notebook)
# ----------------------------------------------------------------------------
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
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING = True
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = False
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
observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
observed_path_Reynold = "/Users/julia/Desktop/SSTA/datasets/Reynold_sst_anomalies-(2004-2018).nc"

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

observed_temp_ds_reynold = xr.open_dataset(observed_path_Reynold, decode_times=False)['anom']
observed_temperature_anomaly_reynold = observed_temp_ds_reynold

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
# model_anomalies_ds = remove_empty_attributes(model_anomalies_ds) # when doing the seasonality removal, some units are None
# model_anomalies_ds.to_netcdf("datasets/implicit_model/" + save_name + ".nc")

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
# flux_components_ds = remove_empty_attributes(flux_components_ds)
# flux_components_ds.to_netcdf("datasets/implicit_model/" + save_name + "_flux_components.nc")

#make_movie(model_anomalies_ds["IMPLICIT"], -3, 3)

#--- 2.1 Prepare Observed Temperature Anomaly ------------------------------------------------------------
implicit_model_anomaly_ds = model_anomalies_ds["IMPLICIT"]


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
    Qent = entrainment_fluxes.rename("Qent")  # W/m^2 anomaly by construction
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

print (flux_all)

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
# Initialize with zeros in the correct shape to preserve xarray coords
Qsum = xr.zeros_like(Tobs) 
if INCLUDE_SURFACE:
    Qsum = Qsum + flux_all["Qsurf"]
if INCLUDE_EKMAN_ANOM_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:          # FIXED flag name
    Qsum = Qsum + flux_all["Qek"]
if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION or INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:    # FIXED flag name
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
    vmin=-2, vmax=0.5
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