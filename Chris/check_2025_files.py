import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter, get_month_from_time
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib

USE_MAX_GRADIENT_METHOD = False


HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Surface_Heat_Flux-(2004-2025).nc"
EKMAN_ANOMALY_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Simulation-Ekman_Heat_Flux-(2004-2025).nc"
TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Temperature_Anomalies-(2004-2025).nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Simulation-Geostrophic_Heat_Flux-(2004-2025).nc"
WIND_STRESS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Turbulent_Surface_Stress-(2004-2025).nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Depth-(2004-2025).nc"
SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Sea_Surface_Height-(2004-2025).nc"

if USE_MAX_GRADIENT_METHOD:
    T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/mld_other_method/Tsub_Max_Gradient_Method_h.nc"
    ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/mld_other_method/Entrainment_Vel_h.nc"
else:
    T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Sub_Layer_Temperature_Anomalies-(2004-2025).nc"
    ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Entrainment_Velocity-(2004-2025).nc"

heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
ekman_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
obs_ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
geo_anom_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
wind_stress_ds = xr.open_dataset(WIND_STRESS_DATA_PATH, decode_times=False)
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
tsub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
sea_surface_ds = xr.open_dataset(SEA_SURFACE_DATA_PATH, decode_times=False)

print(heat_flux_ds)
print(ekman_ds)
print(obs_ds)
print(geo_anom_ds)
print(wind_stress_ds)
print(hbar_ds)
print(tsub_ds)
print(entrainment_vel_ds)
print(sea_surface_ds)
