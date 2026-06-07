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
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, get_month_from_time
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.stats import kurtosis, skew, pearsonr, t
from chris_entrainment_vel_anomaly_forcing import entrainment_vel_anomaly_forcing


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
    # fig.text(
    #     0.99, 0.01,
    #     f"Gamma = {gamma_0}\n"
    #     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    #     f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    #     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
    #     f"INCLUDE_GEOSTROPHIC = {INCLUDE_GEOSTROPHIC_MEAN}\n"
    #     f"INCLUDE_GEOSTROPHIC_DISPLACEMENT = {INCLUDE_GEOSTROPHIC_ANOM}",
    #     ha='right', va='bottom', fontsize=8
    #     )
    plt.tight_layout()
    plt.show()

#%%
#--- 3. Cross Correlation Calculation ----------------------------------------------------------------

# 3.1 Setup Parameters
lags = np.arange(-12, 13)
scheme_name = "Implicit"
lat = observed_temperature_anomaly_reynold['LATITUDE'].values
lon = observed_temperature_anomaly_reynold['LONGITUDE'].values

# 3.2 Pre-calculate Autocorrelation (Persistence)
# High autocorrelation reduces the Effective Degrees of Freedom (N_eff).
r_x = xr.corr(observed_temperature_anomaly_reynold, 
              observed_temperature_anomaly_reynold.shift(TIME=1), dim="TIME")
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
results = [calculate_lag_stats(observed_temperature_anomaly_reynold, 
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
#%%
#_-------------------

# --- 0. Set Publication Plotting Parameters ---
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300, # High resolution for publications
    'font.family': 'sans-serif'
})

# Extract significance mask (True where p < 0.05)
# Assuming p_values_da was calculated previously
significant_mask = p_values_da.isel(lag=best_idx) < 0.05

def format_cartopy_axes(ax):
    """Adds coastlines, land features, and gridlines to a Cartopy axis."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=1)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    return ax

def add_significance_stippling(ax, lon, lat, sig_mask):
    """Adds dots to regions that are statistically significant."""
    # We use contourf with hatches to overlay stippling where sig_mask is True
    ax.contourf(lon, lat, sig_mask, levels=[0.5, 1.5], 
                colors='none', hatches=['...', None], 
                transform=ccrs.PlateCarree(), zorder=2)

# ==============================================================================
# --- FIGURE 1: Maximum Cross-Correlation with Significance Stippling ---
# ==============================================================================
fig1 = plt.figure(figsize=(10, 5))
ax1 = plt.axes(projection=ccrs.Robinson(central_longitude=0)) # Great for global ocean
format_cartopy_axes(ax1)

# Plot correlation using a diverging colormap
plot_corr = best_corr.plot(
    ax=ax1, 
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',       # Standard for diverging data (red=positive, blue=negative)
    vmin=-1, vmax=1, 
    add_colorbar=False,  # We will add a custom one below
    zorder=0
)

# Add significance stippling
add_significance_stippling(ax1, best_corr['LONGITUDE'], best_corr['LATITUDE'], significant_mask)

# Custom Colorbar
cbar1 = plt.colorbar(plot_corr, ax=ax1, orientation='horizontal', pad=0.08, aspect=40, shrink=0.8)
cbar1.set_label('Maximum Cross-Correlation (r)', fontsize=12)

ax1.set_title(f"a) {scheme_name} Scheme: Maximum Cross-Correlation\n(Stippling indicates 95% significance)", loc='left', pad=15)
plt.tight_layout()
plt.savefig("Max_Correlation_Map.pdf", bbox_inches='tight', dpi=300)
plt.show()

# ==============================================================================
# --- FIGURE 2: Lag of Maximum Correlation ---
# ==============================================================================
fig2 = plt.figure(figsize=(10, 5))
ax2 = plt.axes(projection=ccrs.Robinson(central_longitude=0))
format_cartopy_axes(ax2)

# Define discrete levels and a cyclic/diverging colormap for lags
lag_levels = np.arange(lags.min() - 0.5, lags.max() + 1.5, 1)
cmap_lag = plt.get_cmap('PiYG', len(lag_levels)-1) # Pink to Green diverging

# Only plot lag where correlation is significant (common practice in papers)
best_lag_masked = best_lag.where(significant_mask)

plot_lag = best_lag_masked.plot(
    ax=ax2, 
    transform=ccrs.PlateCarree(),
    cmap=cmap_lag,
    levels=lag_levels,
    add_colorbar=False,
    zorder=0
)

cbar2 = plt.colorbar(plot_lag, ax=ax2, orientation='horizontal', pad=0.08, aspect=40, shrink=0.8)
cbar2.set_label('Lag (Months)', fontsize=12)
cbar2.set_ticks(np.arange(lags.min(), lags.max() + 1, 2)) # Tick every 2 months

ax2.set_title(f"b) {scheme_name} Scheme: Lag of Maximum Correlation\n(Masked to significant regions)", loc='left', pad=15)
plt.tight_layout()
# plt.savefig("Max_Lag_Map.pdf", bbox_inches='tight', dpi=300)
plt.show()

# ==============================================================================
# --- FIGURE 3: Zonal Mean Correlation (Very Common in Research Papers) ---
# ==============================================================================
# A plot showing how the correlation changes purely by latitude
zonal_mean_corr = best_corr.mean(dim='LONGITUDE', skipna=True)

fig3, ax3 = plt.subplots(figsize=(5, 6))
ax3.plot(zonal_mean_corr, zonal_mean_corr['LATITUDE'], color='k', linewidth=2)
ax3.fill_betweenx(zonal_mean_corr['LATITUDE'], 0, zonal_mean_corr, 
                  where=(zonal_mean_corr > 0), color='red', alpha=0.3)
ax3.fill_betweenx(zonal_mean_corr['LATITUDE'], 0, zonal_mean_corr, 
                  where=(zonal_mean_corr < 0), color='blue', alpha=0.3)

ax3.axvline(0, color='gray', linestyle='--')
ax3.set_ylim(-80, 80) # Adjust based on your latitude bounds
ax3.set_xlim(-1, 1)
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Zonal Mean Correlation')
ax3.set_title(f"{scheme_name}: Zonal Mean\nMax Correlation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig("Zonal_Mean_Corr.pdf", bbox_inches='tight', dpi=300)
plt.show()