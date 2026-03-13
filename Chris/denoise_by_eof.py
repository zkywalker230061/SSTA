import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from eofs.tools.standard import correlation_map
from scipy import stats
import cartopy.crs as ccrs
import ssl

from scipy.signal import welch

ssl._create_default_https_context = ssl._create_unverified_context

from Chris.correlation_significance import get_significance
from Chris.utils import make_movie, get_eofs, get_eof, get_eof_with_nan_consideration, get_eof_from_ppca_py, \
    get_save_name, \
    get_month_from_time, format_cartopy, mask_dataset
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from correlation_significance import mutual_information

MASK_TROPICS = True
MASK_TROPICS_LATITUDE = 15

MASK_SOME_REGIONS = False
MASK_REGIONS = [(slice(15, 25), slice(-75, -65)), (slice(-50, -30), slice(125, 150))]

NORTH_ATLANTIC = False
NA_LAT_BOUNDS = slice(0, 80)
NA_LONG_BOUNDS = slice(-80, 10)

ARGO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
OBSERVATIONS_2025_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/reynolds_sst_Anomalies-(2004-2025)_no_2004.nc"
obs_ds = xr.open_dataset(OBSERVATIONS_2025_DATA_PATH, decode_times=False)
observed_anomaly = obs_ds['ANOMALY_SST']
if MASK_TROPICS:
    observed_anomaly = observed_anomaly.where((observed_anomaly.LATITUDE > MASK_TROPICS_LATITUDE) | (observed_anomaly.LATITUDE < -1 * MASK_TROPICS_LATITUDE), np.nan)
if NORTH_ATLANTIC:
    observed_anomaly = observed_anomaly.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)

save_name_full = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name_full + ".nc"
model = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)["IMPLICIT"]

argo_observations_ds = load_and_prepare_dataset(ARGO_DATA_PATH)
map_mask = argo_observations_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")

def significant_correlation(to_plot):
    correlation, significant_correlation = get_significance(to_plot, observed_anomaly, resamples=1000,
                                                            test_statistic="PEARSON")
    print(correlation.mean())
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
    lons = significant_correlation.LONGITUDE.values
    lats = significant_correlation.LATITUDE.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    ax.contourf(lon_grid, lat_grid, significant_correlation.values.astype(float), levels=[0.5, 1.5], hatches=['///'],
                colors='none')
    ax = format_cartopy(ax)
    plt.title("")
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Pearson Correlation Coefficient', rotation=90)
    ax.set_xlabel("Longitude (º)")
    ax.set_ylabel("Latitude (º)")
    plt.show()

def correlate_eofs(model):
    EOFPCs = get_eofs(model, 0, 20, map_mask, standardise=False, invert=False)
    eof_modes = EOFPCs[2]
    significant_correlation(eof_modes)
    significant_correlation(model)

def get_correlation_with_varying_eofs(model, max_eofs=20):
    eofs = np.linspace(1, max_eofs, max_eofs)
    correlations = []
    for i in eofs:
        eof_modes = get_eofs(model, 0, int(i), map_mask, standardise=False, invert=False)[2]
        correlation = xr.corr(eof_modes, observed_anomaly, dim='TIME')
        correlations.append(correlation.mean().values)
    plt.grid()
    plt.plot(eofs, correlations)
    plt.xlabel("Number of EOF Modes Included")
    plt.ylabel("Pearson Correlation")
    plt.show()

#correlate_eofs(model)
get_correlation_with_varying_eofs(model, max_eofs=80)
