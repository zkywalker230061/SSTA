import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from Chris.utils import compute_gradient_lat, compute_gradient_lon, get_month_from_time, get_anomaly
from utils import get_monthly_mean, make_movie, get_save_name

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = False
INCLUDE_ENTRAINMENT = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = False

USE_DOWNLOADED_SSH = False
USE_OTHER_MLD = False
USE_MAX_GRADIENT_METHOD = True
USE_LOG_FOR_ENTRAINMENT = False
rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15.0
g = 9.81

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION, USE_DOWNLOADED_SSH, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION, OTHER_MLD=USE_OTHER_MLD, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, LOG_ENTRAINMENT_VELOCITY=USE_LOG_FOR_ENTRAINMENT)

FLUX_CONTRIBUTIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + "_flux_components.nc"
T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Entrainment_Velocity-(2004-2018).nc"
IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + ".nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/hbar.nc"
EKMAN_MEAN_ADVECTION_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/ekman_mean_advection.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"
SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_monthly_mean_calculated_grad.nc"

rho_0 = 1025.0
c_0 = 4100.0
g = 9.81

all_components = xr.open_dataset(FLUX_CONTRIBUTIONS_DATA_PATH, decode_times=False)
t_sub = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
tm_anomaly_ds = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

ekman_mean_advection = xr.open_dataset(EKMAN_MEAN_ADVECTION_DATA_PATH, decode_times=False)
sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)
sea_surface_monthlymean_ds = xr.open_dataset(SEA_SURFACE_MONTHLY_MEAN_DATA_PATH, decode_times=False)

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


def plot_over_time(plot_list, readable_list, signed=False):
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


def correlate_turbulent_ekman():
    correlation = xr.corr(all_components["TURBULENT_FLUX_ANOMALY"], all_components["EKMAN_ANOM_ADVECTION_ANOMALY"], dim='TIME')
    correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
    plt.title("")
    plt.xlabel("Longitude (º)")
    plt.ylabel("Latitude (º)")
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Pearson Correlation Coefficient', rotation=90)
    # plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/correlations/" + save_name + "_correlation.jpg")
    # plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_poster/full_model_correlation.png", dpi=400)
    plt.show()


# get_flux_proportion("SURFACE_FLUX_ANOMALY", save_file=True)
# get_flux_proportion("EKMAN_ANOM_ADVECTION_ANOMALY", save_file=True)
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

plot_over_time(component_list, readable_component_list, signed=True)

# correlate_turbulent_ekman()
