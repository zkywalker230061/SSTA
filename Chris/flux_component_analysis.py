import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from utils import get_monthly_mean, make_movie, get_save_name

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC = True
INCLUDE_GEOSTROPHIC_DISPLACEMENT = False
USE_DOWNLOADED_SSH = False
gamma_0 = 15.0

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC, USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_DISPLACEMENT)

FLUX_CONTRIBUTIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name + "_flux_components.nc"
T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Entrainment_Velocity-(2004-2018).nc"
all_components = xr.open_dataset(FLUX_CONTRIBUTIONS_DATA_PATH, decode_times=False)
t_sub = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])

all_components["TOTAL_FLUX_ANOMALY"] = abs(all_components["SURFACE_FLUX_ANOMALY"]) + abs(all_components["EKMAN_FLUX_ANOMALY"]) + abs(all_components["ENTRAINMENT_FLUX_IMPLICIT_ANOMALY"]) + abs(all_components["GEOSTROPHIC_FLUX_ANOMALY"])
all_components["SURFACE_FLUX_PROPORTION"] = abs(all_components["SURFACE_FLUX_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]
all_components["EKMAN_FLUX_PROPORTION"] = abs(all_components["EKMAN_FLUX_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]
all_components["ENTRAINMENT_FLUX_PROPORTION"] = abs(all_components["ENTRAINMENT_FLUX_IMPLICIT_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]
all_components["GEOSTROPHIC_FLUX_PROPORTION"] = abs(all_components["GEOSTROPHIC_FLUX_ANOMALY"]) / all_components["TOTAL_FLUX_ANOMALY"]

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
make_movie(all_components["GEOSTROPHIC_FLUX_PROPORTION"], 0, 1, "Geostrophic Flux Proportion of Total Flux", savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/videos/" + save_name + "_geostrophic_component.mp4")


# make_movie(t_sub['T_sub'], 0, 30, "T_sub Anomaly")
# make_movie(t_sub['T_sub_ANOMALY'], 0, 3, "T_sub Anomaly")
# make_movie(entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] * 26298000, 0, 1500, "Entrainment velocity")
# print((entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] * 26298000).max().item())
