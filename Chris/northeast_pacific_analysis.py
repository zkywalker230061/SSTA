import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import pearsonr
from scipy.signal import correlate, correlation_lags
import pandas as pd

from Chris.correlation_significance import get_significance
from Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter, get_month_from_time, plot_eof_for_regional_analysis, get_eof, get_monthly_eof, get_seasonal_eof, get_eofs
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib

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
gamma_0 = 15.0

DATA_TO_2025 = True

IMPLICIT_MODEL = True

NEP_LAT_BOUNDS = slice(25, 60)
NEP_LONG_BOUNDS = slice(-180, -100)

NAO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/nao.txt"
save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION,
                          USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH, gamma0=gamma_0,
                          INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION ,OTHER_MLD=USE_OTHER_MLD, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, LOG_ENTRAINMENT_VELOCITY=USE_LOG_FOR_ENTRAINMENT, SPLIT_SURFACE=SPLIT_SURFACE, INCLUDE_RADIATIVE_SURFACE=INCLUDE_RADIATIVE_SURFACE, INCLUDE_TURBULENT_SURFACE=INCLUDE_TURBULENT_SURFACE, DATA_TO_2025=DATA_TO_2025)
IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + ".nc"
ARGO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
REYNOLDS_OBS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sst_anomalies-(2004-2018).nc"
OBSERVATIONS_2025_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Temperature_Anomalies-(2004-2025).nc"

model_da = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)["IMPLICIT"]
model_da = model_da.sel(LATITUDE=NEP_LAT_BOUNDS).sel(LONGITUDE=NEP_LONG_BOUNDS)   # only take northeast Pacific

if DATA_TO_2025:
    obs_da = xr.open_dataset(OBSERVATIONS_2025_DATA_PATH, decode_times=False)['ANOMALY_ML_TEMPERATURE']
else:
    obs_da = xr.open_dataset(REYNOLDS_OBS_DATA_PATH, decode_times=False)['anom']
obs_da = obs_da.sel(LATITUDE=NEP_LAT_BOUNDS).sel(LONGITUDE=NEP_LONG_BOUNDS)

# assign MONTH coordinate as well
times = model_da.TIME.values
months = []
seasons = []
for time in times:
    month = get_month_from_time(time)
    months.append(month)
    if month == 2 or month == 12 or month == 1:
        seasons.append(0)
    if month == 5 or month == 3 or month == 4:
        seasons.append(1)
    if month == 8 or month == 6 or month == 7:
        seasons.append(2)
    if month == 11 or month == 9 or month == 10:
        seasons.append(3)
model_da = model_da.assign_coords(MONTH=("TIME", months))
model_da = model_da.assign_coords(SEASON=("TIME", seasons))
obs_da = obs_da.assign_coords(MONTH=("TIME", months))
obs_da = obs_da.assign_coords(SEASON=("TIME", seasons))

# print(model_da)
# print(model_da.sel(MONTH=12))

argo_observations_ds = load_and_prepare_dataset(ARGO_DATA_PATH)
map_mask = argo_observations_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")
map_mask = map_mask.sel(LATITUDE=NEP_LAT_BOUNDS).sel(LONGITUDE=NEP_LONG_BOUNDS)

# def get_eofs(model, start_mode, end_mode):
#     monthly_mean = get_monthly_mean(model)
#     eof_modes, explained_variance, PCs, EOFs = get_eof_with_nan_consideration(model, modes=end_mode, mask=map_mask,
#                                                                               tolerance=1e-15,
#                                                                               monthly_mean_ds=monthly_mean,
#                                                                               start_mode=start_mode, max_iterations=4)
#     PCs = PCs * -1
#     EOFs = EOFs * -1
#     PCs_standard = (PCs - PCs.mean(axis=0)) / PCs.std(axis=0)  # standardise
#     EOFs_standard = (EOFs - EOFs.mean(dim=["LATITUDE", "LONGITUDE"])) / EOFs.std(dim=["LATITUDE", "LONGITUDE"])
#     return [EOFs_standard, PCs_standard, eof_modes, explained_variance]

def get_blob_index(model):
    first_eof = get_eofs(model, 0, 1, map_mask=map_mask)[2]
    second_eof = get_eofs(model, 1, 2, map_mask=map_mask)[2]
    blob_long_range = slice(-165, -146)
    blob_lat_range = slice(35, 45)

    blob_region_first_eof = first_eof.sel(LONGITUDE=blob_long_range).sel(LATITUDE=blob_lat_range)
    blob_region_second_eof = second_eof.sel(LONGITUDE=blob_long_range).sel(LATITUDE=blob_lat_range)
    indices = []
    for time in model.TIME.values:
        blob_mean_first_eof = float(blob_region_first_eof.sel(TIME=time).mean().values)
        blob_mean_second_eof = float(blob_region_second_eof.sel(TIME=time).mean().values)
        if abs(blob_mean_first_eof) > abs(blob_mean_second_eof):
            blob_index = abs(blob_mean_first_eof)
        else:
            blob_index = abs(blob_mean_second_eof)
        indices.append(blob_index)
    indices = np.array(indices)
    indices = (indices - indices.mean()) / indices.std()
    plt.grid()
    plt.plot((model.TIME.values / 12) + 2004, indices)
    plt.show()
    return indices


def get_rolling_eof(model_da, average_years=4, season=None):
    start_time = 0.5
    centred_years = []
    rolling_eofs = []
    explained_variance_first_mode = []
    explained_variance_second_mode = []
    while start_time <= 180.5 - 12 * average_years:
        end_time = start_time + 12 * average_years
        if season is not None:
            truncated_model_da = model_da.sel(TIME=slice(start_time, end_time)).sel(SEASON=season)
        else:
            truncated_model_da = model_da.sel(TIME=slice(start_time, end_time))
        EOFPCs = get_eofs(truncated_model_da, 0, 2)
        truncated_eofs = EOFPCs[0]
        centred_years.append((start_time-0.5) / 12 + 2004 + average_years / 2)
        rolling_eofs.append(truncated_eofs)
        explained_variance_first_mode.append(EOFPCs[3][0])
        explained_variance_second_mode.append(EOFPCs[3][1])
        start_time += 12
    rolling_eofs = xr.concat(rolling_eofs, dim=pd.Index(centred_years, name="CENTRED_YEAR"))
    # explained_variance_first_mode = xr.concat(explained_variance_first_mode, dim=pd.Index(centred_years, name="CENTRED_YEAR")).rename("FIRST_MODE_EXPVAR")
    # explained_variance_second_mode = xr.concat(explained_variance_second_mode, dim=pd.Index(centred_years, name="CENTRED_YEAR")).rename("SECOND_MODE_EXPVAR")
    # explained_variance = xr.merge([explained_variance_first_mode, explained_variance_second_mode])
    print(rolling_eofs)
    return [rolling_eofs, [centred_years, explained_variance_first_mode, explained_variance_second_mode]]

def plot_rolling_eofs(rolling_eofs, obs=False):
    for year in rolling_eofs.CENTRED_YEAR.values:
        if obs:
            save_name = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/northeast_pacific_analysis/rolling_EOFs/rolling_EOFs_obs_" + str(year) + ".jpg"
        else:
            save_name = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/northeast_pacific_analysis/rolling_EOFs/rolling_EOFs_" + str(year) + ".jpg"
        plot_eof_for_regional_analysis(rolling_eofs.sel(CENTRED_YEAR=year), 2, save_name=save_name)

def plot_explained_variance(rolling_eof_explained_variances, rolling_eof_explained_variances_obs):
    print(rolling_eof_explained_variances)
    plt.grid()
    plt.plot(rolling_eof_explained_variances[0], rolling_eof_explained_variances[1], label="First mode explained variance")
    plt.plot(rolling_eof_explained_variances[0], rolling_eof_explained_variances[2], label="Second mode explained variance")
    plt.xlabel("Year")
    plt.ylabel("Explained variance")
    plt.legend()
    plt.show()

    plt.grid()
    plt.plot(rolling_eof_explained_variances_obs[0], rolling_eof_explained_variances_obs[1],label="First mode explained variance")
    plt.plot(rolling_eof_explained_variances_obs[0], rolling_eof_explained_variances_obs[2],label="Second mode explained variance")
    plt.xlabel("Year")
    plt.ylabel("Explained variance")
    plt.legend()
    plt.show()


# rolling_eofs = get_rolling_eof(model_da, average_years=4, season=1)
# rolling_eofs_obs = get_rolling_eof(obs_da, average_years=4, season=1)

# plot_rolling_eofs(rolling_eofs[0])
# plot_rolling_eofs(rolling_eofs_obs[0], obs=True)

# plot_explained_variance(rolling_eofs[1], rolling_eofs_obs[1])

# get_eof(model_da, obs=False, obs_da=obs_da, save_folder="northeast_pacific_analysis/", map_mask=map_mask)
# get_monthly_eof(model_da, obs=False, obs_da=obs_da, save_folder="northeast_pacific_analysis/", map_mask=map_mask)
# get_seasonal_eof(model_da, obs=True, obs_da=obs_da, save_folder="northeast_pacific_analysis/", map_mask=map_mask)

get_blob_index(model_da)
get_blob_index(obs_da)
