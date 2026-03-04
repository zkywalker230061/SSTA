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
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, get_month_from_time


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
# if INCLUDE_ENTRAINMENT:
#     entrainment_fluxes = xr.concat(entrainment_fluxes, 'TIME')
#     entrainment_fluxes = entrainment_fluxes.drop_vars(["MONTH", "PRESSURE"])
#     entrainment_fluxes = entrainment_fluxes.transpose("TIME", "LATITUDE", "LONGITUDE")
#     entrainment_fluxes = entrainment_fluxes.rename("ENTRAINMENT_ANOMALY")

# # merge the relevant fluxes into a single dataset
# flux_components_to_merge = []
# variable_names = []
# if INCLUDE_SURFACE:
#     surface_flux_da = surface_flux_da.rename("SURFACE_FLUX_ANOMALY")
#     flux_components_to_merge.append(surface_flux_da)
#     variable_names.append("SURFACE_FLUX_ANOMALY")

# if INCLUDE_EKMAN_ANOM_ADVECTION:
#     ekman_anomaly_da = ekman_anomaly_da.rename("EKMAN_ANOM_ADVECTION_ANOMALY")
#     flux_components_to_merge.append(ekman_anomaly_da)
#     variable_names.append("EKMAN_ANOM_ADVECTION_ANOMALY")

# if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
#     geostrophic_anomaly_da = geostrophic_anomaly_da.rename("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY")
#     flux_components_to_merge.append(geostrophic_anomaly_da)
#     variable_names.append("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY")

# if INCLUDE_ENTRAINMENT:
#     flux_components_to_merge.append(entrainment_fluxes)
#     variable_names.append("ENTRAINMENT_ANOMALY")

# flux_components_ds = xr.merge(flux_components_to_merge)

# # remove whatever seasonal cycle may remain from the components
# for variable_name in variable_names:
#     monthly_mean = get_monthly_mean(flux_components_ds[variable_name])
#     flux_components_ds[variable_name] = get_anomaly(flux_components_ds, variable_name, monthly_mean)[variable_name + "_ANOMALY"]
#     flux_components_ds = flux_components_ds.drop_vars(variable_name + "_ANOMALY")

# save flux components
# flux_components_ds = remove_empty_attributes(flux_components_ds)
# flux_components_ds.to_netcdf("datasets/implicit_model/" + save_name + "_flux_components.nc")

#make_movie(model_anomalies_ds["IMPLICIT"], -3, 3)

#--- 2.1 Prepare Observed Temperature Anomaly ------------------------------------------------------------
implicit_model_anomaly_ds = model_anomalies_ds["IMPLICIT"]


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