import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import pearsonr
from scipy.signal import correlate, correlation_lags
import pandas as pd

from Chris.correlation_significance import get_significance
from Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter, get_month_from_time
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

IMPLICIT_MODEL = True

NA_LAT_BOUNDS = slice(0, 80)
NA_LONG_BOUNDS = slice(-80, 10)

NAO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/nao.txt"
save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION,
                          USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH, gamma0=gamma_0,
                          INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION ,OTHER_MLD=USE_OTHER_MLD, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, LOG_ENTRAINMENT_VELOCITY=USE_LOG_FOR_ENTRAINMENT, SPLIT_SURFACE=SPLIT_SURFACE, INCLUDE_RADIATIVE_SURFACE=INCLUDE_RADIATIVE_SURFACE, INCLUDE_TURBULENT_SURFACE=INCLUDE_TURBULENT_SURFACE)
IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + ".nc"
ARGO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
REYNOLDS_OBS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sst_anomalies-(2004-2018).nc"

model_da = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)["IMPLICIT"]
model_da = model_da.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)   # only take north atlantic

obs_da = xr.open_dataset(REYNOLDS_OBS_DATA_PATH, decode_times=False)['anom']
obs_da = obs_da.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)

# assign MONTH coordinate as well
times = model_da.TIME.values
months = []
seasons = []
for time in times:
    month = get_month_from_time(time)
    months.append(month)
    # if month == 11 or month == 12 or month == 1:
    #     seasons.append(0)
    # if month == 2 or month == 3 or month == 4:
    #     seasons.append(1)
    # if month == 5 or month == 6 or month == 7:
    #     seasons.append(2)
    # if month == 8 or month == 9 or month == 10:
    #     seasons.append(3)
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
# print(model_da)
# print(model_da.sel(MONTH=12))

argo_observations_ds = load_and_prepare_dataset(ARGO_DATA_PATH)
map_mask = argo_observations_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")
map_mask = map_mask.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)

def read_nao(file):
    nao_list = []
    with open(file, "r") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
            else:
                line = line.strip()
                split_line = line.split()
                year = int(split_line[0])
                if year >= 2004:
                    nao_indices = split_line[1:13]
                    for nao_index in nao_indices:
                        nao_list.append(float(nao_index))
                        if len(nao_list) >= 180:
                            nao_list = np.array(nao_list)
                            nao_list = (nao_list - nao_list.mean()) / nao_list.std()
                            return nao_list
    nao_list = np.array(nao_list)
    nao_list = (nao_list - nao_list.mean()) / nao_list.std()
    return nao_list

def get_eofs(model, start_mode, end_mode):
    monthly_mean = get_monthly_mean(model)
    eof_modes, explained_variance, PCs, EOFs = get_eof_with_nan_consideration(model, modes=end_mode, mask=map_mask,
                                                                              tolerance=1e-15,
                                                                              monthly_mean_ds=monthly_mean,
                                                                              start_mode=start_mode, max_iterations=4)
    PCs = PCs * -1
    EOFs = EOFs * -1
    PCs_standard = (PCs - PCs.mean(axis=0)) / PCs.std(axis=0)  # standardise
    EOFs_standard = (EOFs - EOFs.mean(dim=["LATITUDE", "LONGITUDE"])) / EOFs.std(dim=["LATITUDE", "LONGITUDE"])
    return [EOFs_standard, PCs_standard, eof_modes]

def plot_eof(EOFs, number, save_name=None):
    fig, axs = plt.subplots(number, 1)
    fig.tight_layout()
    norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    for k in range(number):
        axs[k].grid()
        pcolormesh = axs[k].pcolormesh(EOFs.LONGITUDE.values, EOFs.LATITUDE.values, EOFs.isel(MODE=k), cmap='RdBu_r', norm=norm)
        if k == number - 1:
            axs[k].set_xlabel("Longitude")
        axs[k].set_ylabel("Latitude")
    cbar = fig.colorbar(pcolormesh, ax=axs, label="EOF spatial pattern (standardised)")
    if save_name is not None:
        plt.savefig(save_name ,dpi=400)
    # plt.show()

def get_monthly_eof(model):
    for month in range(1, 13):
        EOFsPCs = get_eofs(model.sel(MONTH=month), 0, 3)
        EOFs = EOFsPCs[0]
        PCs = EOFsPCs[1]
        plot_eof(EOFs, 2, save_name="/Volumes/G-DRIVE ArmorATD/Extension/datasets/north_atlantic_analysis/EOFs_month" + str(month) + ".jpg")

def get_seasonal_eof(model):
    for season in range(0, 4):
        EOFsPCs = get_eofs(model.sel(SEASON=season), 0, 3)
        EOFs = EOFsPCs[0]
        PCs = EOFsPCs[1]
        plot_eof(EOFs, 2, save_name="/Volumes/G-DRIVE ArmorATD/Extension/datasets/north_atlantic_analysis/EOFs_season" + str(season) + ".jpg")

def get_nat_index(model, is_obs=False):
    first_eof = get_eofs(model, 0, 1)[2]
    # make_movie(first_eof, -2, 2)
    if is_obs:
        upper_long_range = slice(-40, -20)
        upper_lat_range = slice(45, 55)
        lower_long_range = slice(-70, -60)
        lower_lat_range = slice(22, 32)
    else:
        lower_long_range = slice(-70, -50)
        lower_lat_range = slice(20, 30)
        upper_long_range = slice(-40, -20)
        upper_lat_range = slice(35, 60)

    lower_region_eof = first_eof.sel(LONGITUDE=lower_long_range).sel(LATITUDE=lower_lat_range)
    upper_region_eof = first_eof.sel(LONGITUDE=upper_long_range).sel(LATITUDE=upper_lat_range)
    indices = []
    for time in model.TIME.values:
        upper_max = upper_region_eof.sel(TIME=time).mean().values
        lower_max = lower_region_eof.sel(TIME=time).mean().values
        indices.append(float(upper_max - lower_max))
    indices = np.array(indices)
    indices = (indices - indices.mean()) / indices.std()
    return indices

def compare_nat(time, model_nat, obs_nat):
    correlation = pearsonr(model_nat, obs_nat)
    print(correlation)
    plt.grid()
    plt.plot(time, model_nat, label="Model")
    plt.plot(time, obs_nat, label="Observations")
    plt.xlabel("Year")
    plt.ylabel("North Atlantic Tripole Index")
    plt.legend()
    plt.show()

def compare_nat_nao(time, nat, nao):
    # plt.grid()
    # plt.plot(time, nat, label="NA Tripole Index")
    # plt.plot(time, nao, label="NA Oscillation Index")
    # plt.xlabel("Year")
    # plt.ylabel("Index")
    # plt.legend()
    # plt.show()

    dates = pd.date_range('2004-01', periods=len(nat), freq='MS')   # use datetime format for lags
    df = pd.DataFrame({'nat': nat, 'nao': nao}, index=dates)
    df = df.rolling(2, center=True).mean().dropna()     # smooth with rolling mean

    month_name_list = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    lags = np.arange(-15, 16)

    fig, axs = plt.subplots(4, 3, figsize=(15, 12), sharey=True, sharex=True)
    axs = axs.flatten()
    # get correlation at various lags for every month
    for ax, (month, month_name) in zip(axs, month_name_list.items()):
        nat_month = df[df.index.month == month]['nat']
        lagged_correlations = []
        for lag in lags:
            nao_shifted = df['nao'].shift(lag)
            nao_month = nao_shifted[nao_shifted.index.month == month]
            nat_nao_shifted = pd.concat([nat_month, nao_month], axis=1).dropna()
            corr = np.corrcoef(nat_nao_shifted.iloc[:, 0], nat_nao_shifted.iloc[:, 1])[0, 1]
            lagged_correlations.append(corr)
        lagged_correlations = np.array(lagged_correlations)

        ax.plot(lags, lagged_correlations, marker='x')
        ax.set_title(month_name)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-0.8, 0.8)
        ax.grid()

    fig.supxlabel('Lag (months)')
    fig.supylabel('Pearson Correlation Coefficient')
    plt.suptitle('North Atlantic Tripole Index Correlation to North Atlantic Oscillation Index')
    plt.tight_layout()
    plt.show()

nao_list = read_nao(NAO_DATA_PATH)
# get_monthly_eof(model_da)
# get_seasonal_eof(model_da)
nat_list = get_nat_index(model_da)
nat_list_obs = get_nat_index(obs_da, is_obs=True)
# compare_nat((model_da.TIME.values / 12) + 2004, nat_list, nat_list_obs)
# compare_nat_nao((model_da.TIME.values / 12) + 2004, nat_list_obs, nao_list)

correlation, significant_correlation = get_significance(model_da, obs_da, resamples=100, test_statistic="MI")

fig, ax = plt.subplots()
correlation.plot(cmap='nipy_spectral', vmin=-1, vmax=1)
lons = significant_correlation.LONGITUDE.values
lats = significant_correlation.LATITUDE.values
lon_grid, lat_grid = np.meshgrid(lons, lats)
ax.contourf(lon_grid, lat_grid, significant_correlation.values.astype(float), levels=[0.5, 1.5], hatches=['///'], colors='none')
plt.show()
