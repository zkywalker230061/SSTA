import xarray as xr
import numpy as np
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Datasets.nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = r"C:\Users\jason\MSciProject\heat_flux_interpolated_all_contributions.nc"
# HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = r"C:\Users\jason\MSciProject\Ekman_Anomaly_Full_Datasets.nc"
TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
ENTRAINMENT_VEL_DATA_PATH = r"C:\Users\jason\MSciProject\Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"
# H_BAR_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"

# Note we were using the uncapped hbar previously 
H_BAR_DATA_PATH = r"C:\Users\jason\MSciProject\hbar.nc"
NEW_H_BAR_DATA_PATH = r"C:\Users\jason\MSciProject\new_hbar.nc"

T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\t_sub.nc"
NEW_T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\new_T_sub_prime.nc"

GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_calculated.nc"
SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_interpolated_grad.nc"


hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
new_hbar_ds = xr.open_dataset(NEW_H_BAR_DATA_PATH, decode_times=False)

tsub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
new_tsub_ds = xr.open_dataset(NEW_T_SUB_DATA_PATH, decode_times=False)

ekman_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)

observed_temp_ds_full = xr.open_dataset(observed_path, decode_times=False)
observed_temp_ds = observed_temp_ds_full["UPDATED_MIXED_LAYER_TEMP"]
obs_temp_mean = get_monthly_mean(observed_temp_ds)
obs_temp_anom = get_anomaly(observed_temp_ds_full, "UPDATED_MIXED_LAYER_TEMP", obs_temp_mean)
print(obs_temp_anom)
obs_temp_anom = obs_temp_anom["UPDATED_MIXED_LAYER_TEMP_ANOMALY"]
print(obs_temp_anom)


# print(hbar_ds)
# print(new_hbar_ds)
# print(tsub_ds)
# print(new_tsub_ds)
# print(ekman_ds)
# print(obs)