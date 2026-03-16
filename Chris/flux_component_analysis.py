import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from Chris.utils import compute_gradient_lat, compute_gradient_lon, get_month_from_time, get_anomaly, format_cartopy, \
    get_autocorrelation, plot_autocorrelation
from utils import get_monthly_mean, make_movie, get_save_name
import cartopy.crs as ccrs
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_ENTRAINMENT = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True

SPLIT_SURFACE = True
INCLUDE_RADIATIVE_SURFACE = True
INCLUDE_TURBULENT_SURFACE = True

USE_DOWNLOADED_SSH = False
USE_OTHER_MLD = False
USE_MAX_GRADIENT_METHOD = True
USE_LOG_FOR_ENTRAINMENT = False
DATA_TO_2025 = True

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15.0
g = 9.81

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION, USE_DOWNLOADED_SSH, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION, OTHER_MLD=USE_OTHER_MLD, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, LOG_ENTRAINMENT_VELOCITY=USE_LOG_FOR_ENTRAINMENT, SPLIT_SURFACE=SPLIT_SURFACE, INCLUDE_RADIATIVE_SURFACE=INCLUDE_RADIATIVE_SURFACE, INCLUDE_TURBULENT_SURFACE=INCLUDE_TURBULENT_SURFACE, DATA_TO_2025=DATA_TO_2025)
FLUX_CONTRIBUTIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + "_flux_components.nc"
IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + ".nc"
REYNOLDS_OBS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sst_anomalies-(2004-2018).nc"
OBSERVATIONS_2025_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Temperature_Anomalies-(2004-2025).nc"

if DATA_TO_2025:
    T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Sub_Layer_Temperature_Anomalies-(2004-2025).nc"
    ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Entrainment_Velocity-(2004-2025).nc"
    H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Depth-(2004-2025).nc"      # Actually this is h, but named H_BAR for consistency
    EKMAN_MEAN_ADVECTION_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/2025_ekman_mean_advection.nc"
    SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/2025_sea_surface_calculated_grad.nc"
    SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/2025_sea_surface_monthly_mean_calculated_grad.nc"

else:
    T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
    ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Entrainment_Velocity-(2004-2018).nc"
    H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/hbar.nc"
    EKMAN_MEAN_ADVECTION_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/ekman_mean_advection.nc"
    SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"
    SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_monthly_mean_calculated_grad.nc"

if DATA_TO_2025:
    obs_da = xr.open_dataset(OBSERVATIONS_2025_DATA_PATH, decode_times=False)['ANOMALY_ML_TEMPERATURE']
else:
    obs_da = xr.open_dataset(REYNOLDS_OBS_DATA_PATH, decode_times=False)['anom']

rho_0 = 1025.0
c_0 = 4100.0
g = 9.81

all_components = xr.open_dataset(FLUX_CONTRIBUTIONS_DATA_PATH, decode_times=False)
# t_sub = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)

entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
if DATA_TO_2025:
    entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['w_e'])
else:
    entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

tm_anomaly_ds = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
if DATA_TO_2025:
    h_da = hbar_ds["MLD"]
    hbar_da = get_monthly_mean(h_da)
else:
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

ekman_mean_advection = xr.open_dataset(EKMAN_MEAN_ADVECTION_DATA_PATH, decode_times=False)
sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)
sea_surface_monthlymean_ds = xr.open_dataset(SEA_SURFACE_MONTHLY_MEAN_DATA_PATH, decode_times=False)

model = tm_anomaly_ds["IMPLICIT"]

all_components["TOTAL_FLUX_ANOMALY"] = xr.zeros_like(all_components["EKMAN_ANOM_ADVECTION_ANOMALY"])
all_components["SIGNED_TOTAL_FLUX_ANOMALY"] = xr.zeros_like(all_components["EKMAN_ANOM_ADVECTION_ANOMALY"])
component_list = []
component_list_separate_surface_flux = []
readable_component_list = []
readable_component_list_separate_surface_flux = []


if INCLUDE_SURFACE:
    # all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["SURFACE_FLUX_ANOMALY"])
    all_components["SURFACE_FLUX_ANOMALY_SUM"] = (all_components["SURFACE_LATENT_HF_ANOMALY"]) + (all_components["SURFACE_LW_RADIATION_FLUX_ANOMALY"]) + (all_components["SURFACE_SW_RADIATION_FLUX_ANOMALY"]) + (all_components["SURFACE_SENSIBLE_HF_ANOMALY"])

    all_components["RADIATIVE_FLUX_ANOMALY"] = (all_components["SURFACE_LW_RADIATION_FLUX_ANOMALY"]) + (all_components["SURFACE_SW_RADIATION_FLUX_ANOMALY"])
    all_components["TURBULENT_FLUX_ANOMALY"] = (all_components["SURFACE_LATENT_HF_ANOMALY"]) + (all_components["SURFACE_SENSIBLE_HF_ANOMALY"])

    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["RADIATIVE_FLUX_ANOMALY"]) + abs(all_components["TURBULENT_FLUX_ANOMALY"])
    all_components["SIGNED_TOTAL_FLUX_ANOMALY"] = all_components["RADIATIVE_FLUX_ANOMALY"] + all_components["TURBULENT_FLUX_ANOMALY"]

    component_list.append("RADIATIVE_FLUX_ANOMALY")
    readable_component_list.append("Radiative Air-Sea Heat Flux")
    component_list.append("TURBULENT_FLUX_ANOMALY")
    readable_component_list.append("Turbulent Air-Sea Heat Flux")


if INCLUDE_EKMAN_ANOM_ADVECTION:
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["EKMAN_ANOM_ADVECTION_ANOMALY"])
    all_components["SIGNED_TOTAL_FLUX_ANOMALY"] += all_components["EKMAN_ANOM_ADVECTION_ANOMALY"]
    component_list.append("EKMAN_ANOM_ADVECTION_ANOMALY")
    readable_component_list.append("Ekman Anomalous Advection")

if INCLUDE_ENTRAINMENT:
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["ENTRAINMENT_ANOMALY"])
    all_components["SIGNED_TOTAL_FLUX_ANOMALY"] += all_components["ENTRAINMENT_ANOMALY"]
    component_list.append("ENTRAINMENT_ANOMALY")
    readable_component_list.append("Entrainment")

if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["GEOSTROPHIC_ANOM_ADVECTION_ANOMALY"])
    all_components["SIGNED_TOTAL_FLUX_ANOMALY"] += all_components["GEOSTROPHIC_ANOM_ADVECTION_ANOMALY"]
    component_list.append("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY")
    readable_component_list.append("Geostrophic Anomalous Advection")


if INCLUDE_EKMAN_MEAN_ADVECTION or INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
    tm_anomaly_grad_lat = compute_gradient_lat(tm_anomaly_ds["IMPLICIT"])
    tm_anomaly_grad_long = compute_gradient_lon(tm_anomaly_ds["IMPLICIT"])

if INCLUDE_EKMAN_MEAN_ADVECTION:
    alpha = ekman_mean_advection["ekman_alpha"]
    beta = ekman_mean_advection["ekman_beta"]
    ekman_mean_advection_contributions = []
    for time in tm_anomaly_grad_lat.TIME.values:
        month = get_month_from_time(time)
        ekman_mean_advection_contribution = (alpha.sel(MONTH=month) * tm_anomaly_grad_long.sel(TIME=time) + beta.sel(MONTH=month) * tm_anomaly_grad_lat.sel(TIME=time)) * rho_0 * c_0 * hbar_da.sel(MONTH=month)
        ekman_mean_advection_contributions.append(ekman_mean_advection_contribution)
    ekman_mean_advection_contributions_ds = xr.concat(ekman_mean_advection_contributions, 'TIME')
    all_components["EKMAN_MEAN_ADVECTION_ANOMALY"] = ekman_mean_advection_contributions_ds
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["EKMAN_MEAN_ADVECTION_ANOMALY"])
    all_components["SIGNED_TOTAL_FLUX_ANOMALY"] += all_components["EKMAN_MEAN_ADVECTION_ANOMALY"]
    component_list.append("EKMAN_MEAN_ADVECTION_ANOMALY")
    readable_component_list.append("Ekman Mean Advection")


if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
    alpha = sea_surface_monthlymean_ds['alpha']
    beta = sea_surface_monthlymean_ds['beta']
    geostrophic_mean_advection_contributions = []
    for time in tm_anomaly_grad_lat.TIME.values:
        month = get_month_from_time(time)
        geostrophic_mean_advection_contribution = (alpha.sel(MONTH=month) * tm_anomaly_grad_long.sel(TIME=time) + beta.sel(MONTH=month) * tm_anomaly_grad_lat.sel(TIME=time)) * rho_0 * c_0 * hbar_da.sel(MONTH=month)
        geostrophic_mean_advection_contribution = geostrophic_mean_advection_contribution.assign_coords(TIME=time)
        geostrophic_mean_advection_contributions.append(geostrophic_mean_advection_contribution)
    geostrophic_mean_advection_contributions_ds = xr.concat(geostrophic_mean_advection_contributions, 'TIME')
    all_components["GEOSTROPHIC_MEAN_ADVECTION_ANOMALY"] = geostrophic_mean_advection_contributions_ds
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["GEOSTROPHIC_MEAN_ADVECTION_ANOMALY"])
    all_components["SIGNED_TOTAL_FLUX_ANOMALY"] += all_components["GEOSTROPHIC_MEAN_ADVECTION_ANOMALY"]
    component_list.append("GEOSTROPHIC_MEAN_ADVECTION_ANOMALY")
    readable_component_list.append("Geostrophic Mean Advection")

if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION and INCLUDE_EKMAN_MEAN_ADVECTION:
    all_components["MEAN_ADVECTION_ANOMALY"] = all_components["GEOSTROPHIC_MEAN_ADVECTION_ANOMALY"] + all_components["EKMAN_MEAN_ADVECTION_ANOMALY"]



def get_flux_proportion(component, save_file=False, signed=False):
    if signed:
        flux_proportion = all_components[component] / all_components["SIGNED_TOTAL_FLUX_ANOMALY"]
    else:
        flux_proportion = abs(all_components[component]) / all_components["TOTAL_FLUX_ANOMALY"]
    if save_file:
        make_movie(flux_proportion, 0, 1, "Proportion of total flux due to " + component, cmap='Reds', savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/videos/" + component + "flux_proportion_" + save_name + ".mp4")
    else:
        make_movie(flux_proportion, 0, 1, "Proportion of total flux due to " + component, cmap='Reds')


def get_surface_flux_proportion(component, save_file=False):
    flux_proportion = abs(all_components[component]) / all_components["SURFACE_FLUX_ANOMALY"]
    if save_file:
        make_movie(flux_proportion, 0, 1, "Proportion of total flux due to " + component, cmap='Reds', savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/videos/" + component + "flux_proportion_of_surface_" + save_name + ".mp4")
    else:
        make_movie(flux_proportion, 0, 1, "Proportion of total flux due to " + component, cmap='Reds')


def plot_proportions_over_time(plot_list, readable_list, signed=False):
    plt.grid()
    for i in range(len(plot_list)):
        if signed:
            plt.plot(all_components[plot_list[i]].TIME / 12 + 2004, (all_components[plot_list[i]] / all_components['SIGNED_TOTAL_FLUX_ANOMALY']).mean(dim="LATITUDE").mean(dim="LONGITUDE"), label=readable_list[i])
            plt.ylim([-5, 5])
            plt.ylabel("Proportion of Each Contribution to Total Flux")
        else:
            plt.plot(all_components[plot_list[i]].TIME / 12 + 2004, (abs(all_components[plot_list[i]]) / all_components['TOTAL_FLUX_ANOMALY']).mean(dim="LATITUDE").mean(dim="LONGITUDE"), label=readable_list[i])
            plt.ylabel("Proportion of Each Contribution to Total Flux Magnitude")
    plt.xlabel("Year")
    plt.legend()
    plt.show()

def plot_over_time(plot_list, readable_list):
    plt.grid()
    for i in range(len(plot_list)):
        plt.plot(all_components[plot_list[i]].TIME / 12 + 2004, (all_components[plot_list[i]]).mean(dim="LATITUDE").mean(dim="LONGITUDE"), label=readable_list[i])
    plt.ylabel("Contribution to Total Flux")
    plt.xlabel("Year")
    plt.legend()
    plt.show()


def correlate_turbulent_ekman():
    correlation = xr.corr(all_components["TURBULENT_FLUX_ANOMALY"], all_components["EKMAN_ANOM_ADVECTION_ANOMALY"], dim='TIME')
    correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
    plt.title("")
    plt.xlabel("Longitude (º)")
    plt.ylabel("Latitude (º)")
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Pearson Correlation Coefficient', rotation=90)
    plt.show()


def covariance_ratio(component_list, component_names, make_plots=False):
    # tendency = []
    # for time in all_components["TOTAL_FLUX_ANOMALY"].TIME.values:
    #     month = get_month_from_time(time)
    #     tendency.append(all_components["TOTAL_FLUX_ANOMALY"].sel(TIME=time) / (rho_0 * c_0 * hbar_da.sel(MONTH=month)))
    # tendency = xr.concat(tendency, "TIME")

    tendency = all_components["SIGNED_TOTAL_FLUX_ANOMALY"]
    tendency_var = ((tendency - tendency.mean(dim="TIME")) ** 2).mean(dim="TIME")
    ratios = {}
    for forcing in component_list:
        cov = ((all_components[forcing] - all_components[forcing].mean(dim="TIME")) * (tendency - tendency.mean(dim="TIME"))).mean(dim="TIME")
        ratios[forcing] = (cov / tendency_var).rename(forcing)
    ratios = xr.Dataset(ratios)
    ratios.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/covariances/" + save_name + "_covariances.nc")

    if make_plots:
        for i, forcing in enumerate(component_list):
            name = component_names[i]
            ratios[forcing].plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=1)
            plt.title("")
            plt.xlabel("Longitude (º)")
            plt.ylabel("Latitude (º)")
            cbar = plt.gcf().axes[-1]
            cbar.set_ylabel('Covariance of ' + name, rotation=90)
            plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/covariances/" + save_name + "_covariances_" + forcing + ".jpg", dpi=400)
            plt.show()

def correlate_mld_to_fluxes(component, mld):
    if DATA_TO_2025:
        flux = all_components[component]
        print(flux)
        print(mld)

        print(flux.dims)
        print(mld.dims)

        # fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        # correlation = xr.corr(flux, mld, dim='TIME')
        # correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
        # plt.title("")
        # plt.xlabel("Longitude (º)")
        # plt.ylabel("Latitude (º)")
        # cbar = plt.gcf().axes[-1]
        # cbar.set_ylabel('Pearson Correlation Coefficient', rotation=90)
        # plt.title("Correlation between a heat flux and MLD")
        # ax = format_cartopy(ax)
        # plt.show()

        # assign MONTH coordinate to MLD
        times = mld.TIME.values
        months = []
        for time in times:
            month = get_month_from_time(time)
            months.append(month)
        mld = mld.assign_coords(MONTH=("TIME", months))

        lags = np.arange(-5, 6)
        month_name_list = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

        fig, axs = plt.subplots(4, 3, figsize=(15, 12), sharey=True, sharex=True)
        axs = axs.flatten()

        for ax, (month, month_name) in zip(axs, month_name_list.items()):
            lagged_correlations = []

            for lag in lags:
                flux_shifted = flux.shift(TIME=lag)
                flux_month = flux_shifted.isel(TIME=(flux.MONTH == month))
                mld_month = mld.isel(TIME=(mld.MONTH == month))
                corr_map = xr.corr(flux_month, mld_month, dim='TIME')
                weights = np.cos(np.deg2rad(corr_map.LATITUDE))
                corr_mean = float(corr_map.weighted(weights).mean(dim=['LATITUDE', 'LONGITUDE']))
                lagged_correlations.append(corr_mean)

            lagged_correlations = np.array(lagged_correlations)
            ax.plot(lags, lagged_correlations, marker='x')
            ax.set_title(month_name)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-0.3, 0.3)
            ax.grid()

        fig.supxlabel('Lag (months)')
        fig.supylabel('Pearson Correlation Coefficient')
        plt.tight_layout()
        plt.show()

    else:
        print("Only valid for 2025 data")

def get_each_forcing_persistence(full_model, obs, components, readable_components):
    def individual_forcing(to_plot, name):
        print(name)
        print(type(to_plot))
        print(to_plot)
        to_plot_no_unchanging_cells = to_plot.where(to_plot.std("TIME") != 0)
        autocorrelation_functions_ds = get_autocorrelation(to_plot_no_unchanging_cells, min_lag=-3, max_lag=15)  # all time
        plot_autocorrelation(autocorrelation_functions_ds, lag=None, model_name=name, show=False, label=name)
    individual_forcing(full_model, "Full model")
    individual_forcing(obs, "Observations")
    for i, component in enumerate(components):
        individual_forcing(all_components[component], readable_components[i])
    plt.legend()
    plt.show()


NA_LAT_BOUNDS = slice(0, 80)
NA_LONG_BOUNDS = slice(-80, 10)
NEP_LAT_BOUNDS = slice(25, 60)
NEP_LONG_BOUNDS = slice(-180, -100)
SA_LAT_BOUNDS = slice(-60, -40)
SA_LONG_BOUNDS = slice(-50, 10)
SO_LAT_BOUNDS = slice(-60, -40)
SO_LONG_BOUNDS = slice(50, 120)
SEP_LAT_BOUNDS = slice(-60, -40)
SEP_LONG_BOUNDS = slice(-180, -80)

all_components = all_components.where(((all_components.LATITUDE >= 20) & (all_components.LATITUDE <= 60)) | ((all_components.LATITUDE >= -60) & (all_components.LATITUDE <= -20)), drop=True)
model = model.where(((model.LATITUDE >= 20) & (model.LATITUDE <= 60)) | ((model.LATITUDE >= -60) & (model.LATITUDE <= -20)), drop=True)
obs_da = obs_da.where(((obs_da.LATITUDE >= 20) & (obs_da.LATITUDE <= 60)) | ((obs_da.LATITUDE >= -60) & (obs_da.LATITUDE <= -20)), drop=True)


#all_components = all_components.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)
# all_components = all_components.sel(LATITUDE=NEP_LAT_BOUNDS).sel(LONGITUDE=NEP_LONG_BOUNDS)
# all_components = all_components.sel(LATITUDE=SO_LAT_BOUNDS).sel(LONGITUDE=SO_LONG_BOUNDS)
# all_components = all_components.sel(LATITUDE=SEP_LAT_BOUNDS).sel(LONGITUDE=SEP_LONG_BOUNDS)
# obs_da = obs_da.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)


# all_components = all_components.sel(LATITUDE=slice(0, 90))      # NH
# all_components = all_components.sel(LATITUDE=slice(-90, 0))     # SH
# all_components = all_components.sel(LATITUDE=slice(20, 60))      # NH midlat
# all_components = all_components.sel(LATITUDE=slice(-60, -20))     # SH  midlat

# get_flux_proportion("SURFACE_FLUX_ANOMALY", save_file=True)
# get_flux_proportion("EKMAN_ANOM_ADVECTION_ANOMALY", save_file=False)
# get_flux_proportion("ENTRAINMENT_ANOMALY", save_file=True)
# get_flux_proportion("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY", save_file=True)
# get_flux_proportion("EKMAN_MEAN_ADVECTION_ANOMALY", save_file=True)
# get_flux_proportion("GEOSTROPHIC_MEAN_ADVECTION_ANOMALY", save_file=True)

# get_surface_flux_proportion("SURFACE_LATENT_HF_ANOMALY", save_file=False)
# get_surface_flux_proportion("SURFACE_LW_RADIATION_FLUX_ANOMALY", save_file=False)
# get_surface_flux_proportion("SURFACE_SW_RADIATION_FLUX_ANOMALY", save_file=False)
# get_surface_flux_proportion("SURFACE_SENSIBLE_HF_ANOMALY", save_file=False)

# get_surface_flux_proportion("RADIATIVE_FLUX_ANOMALY", save_file=False)
# get_surface_flux_proportion("TURBULENT_FLUX_ANOMALY", save_file=False)

# plot_proportions_over_time(component_list, readable_component_list, signed=False)
# plot_over_time(component_list, readable_component_list)


# correlate_turbulent_ekman()

# covariance_ratio(component_list, readable_component_list, True)

# correlate_mld_to_fluxes("SURFACE_FLUX_ANOMALY_SUM", h_da)
# correlate_mld_to_fluxes("EKMAN_MEAN_ADVECTION_ANOMALY", h_da)
get_each_forcing_persistence(model, obs_da, component_list, readable_component_list)

