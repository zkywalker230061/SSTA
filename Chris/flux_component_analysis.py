import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from SSTA.Chris.utils import compute_gradient_lat, compute_gradient_lon, get_month_from_time
from utils import get_monthly_mean, make_movie, get_save_name

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True
USE_DOWNLOADED_SSH = False
gamma_0 = 15.0

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION, USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION)

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

all_components["TOTAL_FLUX_ANOMALY"] = xr.zeros_like(all_components["SURFACE_FLUX_ANOMALY"])

if INCLUDE_SURFACE:
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["SURFACE_FLUX_ANOMALY"])
if INCLUDE_EKMAN_ANOM_ADVECTION:
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["EKMAN_ANOM_ADVECTION_ANOMALY"])
if INCLUDE_ENTRAINMENT:
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["ENTRAINMENT_ANOMALY"])
if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
    all_components["TOTAL_FLUX_ANOMALY"] += abs(all_components["GEOSTROPHIC_ANOM_ADVECTION_ANOMALY"])

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


def get_flux_proportion(component, save_file=False):
    flux_proportion = abs(all_components[component]) / all_components["TOTAL_FLUX_ANOMALY"]
    print(all_components[component])
    print(all_components["TOTAL_FLUX_ANOMALY"])
    if save_name is not None:
        make_movie(flux_proportion, 0, 1, "Proportion of total flux due to " + component, cmap='Reds', savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/videos/" + component + "flux_proportion" + save_name + ".mp4")
    else:
        make_movie(flux_proportion, 0, 1, "Proportion of total flux due to " + component, cmap='Reds')

get_flux_proportion("SURFACE_FLUX_ANOMALY", save_file=True)
get_flux_proportion("EKMAN_ANOM_ADVECTION_ANOMALY", save_file=True)
get_flux_proportion("ENTRAINMENT_ANOMALY", save_file=True)
get_flux_proportion("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY", save_file=True)
get_flux_proportion("EKMAN_MEAN_ADVECTION_ANOMALY", save_file=True)
get_flux_proportion("GEOSTROPHIC_MEAN_ADVECTION_ANOMALY", save_file=True)


# all_components["TOTAL_FLUX_ANOMALY"] = abs(all_components["SURFACE_FLUX_ANOMALY"]) + abs(all_components["EKMAN_ANOM_ADVECTION_ANOMALY"]) + abs(all_components["ENTRAINMENT_FLUX_IMPLICIT_ANOMALY"]) + abs(all_components["GEOSTROPHIC_FLUX_ANOMALY"])
# all_components["SURFACE_FLUX_PROPORTION"] = abs(all_components["SURFACE_FLUX_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]
# all_components["EKMAN_FLUX_PROPORTION"] = abs(all_components["EKMAN_FLUX_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]
# all_components["ENTRAINMENT_FLUX_PROPORTION"] = abs(all_components["ENTRAINMENT_FLUX_IMPLICIT_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]
# all_components["GEOSTROPHIC_FLUX_PROPORTION"] = abs(all_components["GEOSTROPHIC_FLUX_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]

# print(all_components)
# print((all_components["GEOSTROPHIC_FLUX_ANOMALY"]).max().item())
# print((all_components["GEOSTROPHIC_FLUX_ANOMALY"]).min().item())
# print(abs(all_components["GEOSTROPHIC_FLUX_ANOMALY"]).mean().item())


# def make_movie(dataset, vmin, vmax, colorbar_label):
#     times = dataset.TIME.values
#
#     fig, ax = plt.subplots()
#     pcolormesh = ax.pcolormesh(dataset.LONGITUDE.values, dataset.LATITUDE.values, dataset.isel(TIME=0), cmap='Reds')
#     title = ax.set_title(f'Time = {times[0]}')
#
#     cbar = plt.colorbar(pcolormesh, ax=ax, label=colorbar_label)
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#
#
#     def update(frame):
#         month = int((times[frame] + 0.5) % 12)
#         if month == 0:
#             month = 12
#         year = 2004 + int((times[frame]) / 12)
#         pcolormesh.set_array(dataset.isel(TIME=frame).values.ravel())
#         pcolormesh.set_clim(vmin=vmin, vmax=vmax)
#         cbar.update_normal(pcolormesh)
#         title.set_text(f'Year: {year}; Month: {month}')
#         return [pcolormesh, title]
#
#     animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
#     plt.show()


# make_movie(all_components["TOTAL_FLUX_ANOMALY"], -50, 50, "Total Heat Flux")
# make_movie(all_components["SURFACE_FLUX_PROPORTION"], 0, 1, "Surface Flux Proportion of Total Flux")
# make_movie(all_components["EKMAN_FLUX_PROPORTION"], 0, 1, "Ekman Flux Proportion of Total Flux")
# make_movie(all_components["ENTRAINMENT_FLUX_PROPORTION"], 0, 1, "Entrainment Flux Proportion of Total Flux")
# make_movie(all_components["GEOSTROPHIC_FLUX_PROPORTION"], 0, 1, "Geostrophic Flux Proportion of Total Flux", savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/videos/" + save_name + "_geostrophic_component.mp4")


# make_movie(t_sub['T_sub'], 0, 30, "T_sub Anomaly")
# make_movie(t_sub['T_sub_ANOMALY'], 0, 3, "T_sub Anomaly")
# make_movie(entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] * 26298000, 0, 1500, "Entrainment velocity")
# print((entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] * 26298000).max().item())
