#%%
# --- 1. Loading Salinity Model Data  ---------------------------------- 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from chris_utils import make_movie, get_eof_with_nan_consideration
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.stats import kurtosis, skew, pearsonr

matplotlib.use('TkAgg')

OBSERVED_SALINITY_ANOM_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Salinity_Anomalies-(2004-2018).nc"
MODEL_SALINITY_ANOM_PATH = r"C:\Users\jason\MSciProject\salinity_model_anomalies.nc"

observed_salinity_anomaly = load_and_prepare_dataset(OBSERVED_SALINITY_ANOM_PATH)
observed_salinity_anomaly = observed_salinity_anomaly["ANOMALY_ML_SALINITY"]

model_salinity_anomaly = load_and_prepare_dataset(MODEL_SALINITY_ANOM_PATH)
model_salinity_anomaly = model_salinity_anomaly["IMPLICIT"]


print(observed_salinity_anomaly)
print(model_salinity_anomaly)

#------------------------------------------------------------------------------------------------------------
#%%
def calculate_RMSE_normalised (obs, model, dim = 'TIME'):
    """
    Calculates Root Mean Square Error.
    Formula: sqrt( mean( (obs - model)^2 ) )
    """
    # First Datapoint was set to 0 in the sim
    # We removed it from rmse analysis 

    model = model.isel(TIME=slice(1,None))
    obs = obs.isel(TIME=slice(1,None))

    error = model - obs
    squared_error = error ** 2
    mean_squared_error = squared_error.mean(dim=dim)
    rmse = np.sqrt(mean_squared_error)

    normal_squared_error = obs**2
    normal_mean_squared_error = normal_squared_error.mean(dim=dim)
    normal_rmse = np.sqrt(normal_mean_squared_error)

    corrected_rmse = rmse / normal_rmse
    return corrected_rmse

def calculate_RMSE (obs, model, dim = 'TIME'):
    """
    Calculates Root Mean Square Error.
    Formula: sqrt( mean( (obs - model)^2 ) )
    """
    error = model - obs
    squared_error = error ** 2
    mean_squared_error = squared_error.mean(dim=dim)
    rmse = np.sqrt(mean_squared_error)
    return rmse



fig, axes = plt.subplots(1, 1, figsize=(8,5))


scheme_name = "Implicit"
rmse_map = calculate_RMSE_normalised(observed_salinity_anomaly, model_salinity_anomaly, dim='TIME')
print(rmse_map)

# Plotting
# ax = plt.subplot(3, 2, i + 1)
rmse_map.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'{scheme_name} Scheme - Normalised RMSE')
max_rmse = rmse_map.max().item()
print("Maximum RMSE Value", scheme_name, max_rmse)
max_rmse_location = rmse_map.where(rmse_map == rmse_map.max(), drop=True).squeeze()
print("Maximum RMSE Location:", max_rmse_location)
mean_rmse = rmse_map.mean().item()
print(mean_rmse)
plt.tight_layout()
# fig.text(
#     0.99, 0.01,
#     f"Gamma = {gamma_0}\n"
#     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
#     f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
#     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
#     f"INCLUDE_GEOSTROPHIC = {INCLUDE_GEOSTROPHIC}",
#     ha='right', va='bottom', fontsize=18
# )
plt.show()


#%%

# Seasonal Analysis (Summer for the Northern Hemisphere)
# summer_months_north_index = []
# for i in range(13):
#     summer_months_north = (17.5 + i*12, 18.5 + i*12, 19.5 + i*12)
#     summer_months_north_index.extend(summer_months_north)



# obs_summer_north_ds = observed_temperature_anomaly.sel(TIME=summer_months_north_index, method="nearest")
# obs_summer_north_ds = obs_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

# imp_summer_north_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_north_index, method="nearest")
# imp_summer_north_ds = imp_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

# # Summer months for the Southern Hemisphere 

# summer_months_south_index = []
# for i in range(13):
#     summer_months_south = (11.5 + i*12, 12.5 + i*12, 13.5 +i*12)
#     summer_months_south_index.extend(summer_months_south)


# obs_summer_south_ds = observed_temperature_anomaly.sel(TIME=summer_months_south_index, method="nearest")
# obs_summer_south_ds = obs_summer_south_ds.sel(LATITUDE=slice(-64.5,0))

# imp_summer_south_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_south_index, method="nearest")
# imp_summer_south_ds = imp_summer_south_ds.sel(LATITUDE=slice(-64.5, 0))
# # Calculating RMSE for Summer season

# rmse_summer_north = calculate_RMSE(obs_summer_north_ds, imp_summer_north_ds, dim="TIME")
# rmse_summer_south = calculate_RMSE(obs_summer_south_ds, imp_summer_south_ds, dim="TIME")

# rmse_summer = xr.concat([rmse_summer_south, rmse_summer_north], dim="LATITUDE")

# fig, axes = plt.subplots(1, 1, figsize=(8,5))
# rmse_summer.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
# axes.set_xlabel("Longitude")
# axes.set_ylabel("Lattitude")
# axes.set_title(f'{scheme_name} Scheme - Summer RMSE')
# max_rmse = rmse_summer.max().item()
# print(scheme_name, max_rmse)
# max_rmse_location_summer = rmse_summer.where(rmse_summer == rmse_summer.max(), drop=True).squeeze()
# print(max_rmse_location_summer)
# min_rmse = rmse_summer.min().item()
# print(scheme_name, min_rmse)
# min_rmse_location_summer = rmse_summer.where(rmse_summer == rmse_summer.min(), drop=True).squeeze()
# print(min_rmse_location_summer)
# plt.tight_layout()
# fig.text(
#     0.99, 0.01,
#     f"Gamma = {gamma_0}\n"
#     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
#     f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
#     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
#     ha='right', va='bottom', fontsize=10
# )
# plt.show()

# #%%

# # Winter Seasonal Analysis 

# winter_months_north_index = summer_months_south_index

# obs_winter_north_ds = observed_temperature_anomaly.sel(TIME=winter_months_north_index, method="nearest")
# obs_winter_north_ds = obs_winter_north_ds.sel(LATITUDE=slice(0,79.5))

# imp_winter_north_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_north_index, method="nearest")
# imp_winter_north_ds = imp_winter_north_ds.sel(LATITUDE=slice(0,79.5))

# winter_months_south_index = summer_months_north_index

# obs_winter_south_ds = observed_temperature_anomaly.sel(TIME=winter_months_south_index, method="nearest")
# obs_winter_south_ds = obs_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))

# imp_winter_south_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_south_index, method="nearest")
# imp_winter_south_ds = imp_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))

# rmse_winter_north = calculate_RMSE(obs_winter_north_ds, imp_winter_north_ds)
# rmse_winter_south = calculate_RMSE(obs_winter_south_ds, imp_winter_south_ds)

# rmse_winter = xr.concat([rmse_winter_south, rmse_winter_north], dim="LATITUDE")

# fig, axes = plt.subplots(1, 1, figsize=(8,5))
# rmse_winter.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
# axes.set_xlabel("Longitude")
# axes.set_ylabel("Lattitude")
# axes.set_title(f'{scheme_name} Scheme - Winter RMSE')
# max_rmse = rmse_winter.max().item()
# print(scheme_name, max_rmse)
# max_rmse_location_winter = rmse_winter.where(rmse_winter == rmse_winter.max(), drop=True).squeeze()
# print(max_rmse_location_winter)
# min_rmse = rmse_winter.min().item()
# print(scheme_name, min_rmse)
# min_rmse_location_winter = rmse_winter.where(rmse_winter == rmse_winter.min(), drop=True).squeeze()
# print(min_rmse_location_winter)
# plt.tight_layout()
# fig.text(
#     0.99, 0.01,
#     f"Gamma = {gamma_0}\n"
#     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
#     f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
#     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
#     ha='right', va='bottom', fontsize=10
# )
# plt.show()

#%%

