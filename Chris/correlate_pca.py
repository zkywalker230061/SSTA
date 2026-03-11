import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from eofs.tools.standard import correlation_map
from scipy import stats
import cartopy.crs as ccrs
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from Chris.correlation_significance import get_significance
from Chris.utils import make_movie, get_eofs, get_eof, get_eof_with_nan_consideration, get_eof_from_ppca_py, \
    get_save_name, \
    get_month_from_time, format_cartopy, mask_dataset
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from correlation_significance import mutual_information

MASK_TROPICS = True
MASK_TROPICS_LATITUDE = 15

MASK_SOME_REGIONS = True
MASK_REGIONS = [(slice(15, 25), slice(-75, -65)), (slice(-50, -30), slice(125, 150))]

# MASK_SOUTH_AMERICA_LAT_BOUNDS = slice(-60, -40)
# MASK_SOUTH_AMERICA_LONG_BOUNDS = slice(-70, -30)

ARGO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
OBSERVATIONS_2025_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/Mixed_Layer_Temperature_Anomalies-(2004-2025).nc"
obs_ds = xr.open_dataset(OBSERVATIONS_2025_DATA_PATH, decode_times=False)
observed_anomaly = obs_ds['ANOMALY_ML_TEMPERATURE']
if MASK_TROPICS:
    observed_anomaly = observed_anomaly.where((observed_anomaly.LATITUDE > MASK_TROPICS_LATITUDE) | (observed_anomaly.LATITUDE < -1 * MASK_TROPICS_LATITUDE), np.nan)

save_name_full = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_rad = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=False, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_turb = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=False, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_ekanom = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=False, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_ekmean = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_entrainment = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=False, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_geoanom = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=False, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_geomean = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=False, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_turb_or_ekman = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=False, INCLUDE_EKMAN=False, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_meanadv = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=False, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_entrainment_or_meanadv = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=False, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=False, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)

save_names = [save_name_full, save_name_no_rad, save_name_no_turb, save_name_no_ekanom, save_name_no_ekmean, save_name_no_entrainment, save_name_no_geoanom, save_name_no_geomean, save_name_no_turb_or_ekman, save_name_no_meanadv, save_name_no_entrainment_or_meanadv]
readable_labels = ["Full model", "No radiative air-sea flux", "No turbulent air-sea flux", "No Ekman anom. adv.", "No Ekman mean adv.", "No entrainment", "No geostrophic anom. adv.", "No geostrophic mean adv.", "No turbulent air-sea flux or Ekman adv.", "No mean adv.", "No entrainment or mean adv."]

argo_observations_ds = load_and_prepare_dataset(ARGO_DATA_PATH)
map_mask = argo_observations_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")
# map_mask = map_mask.sel(LATITUDE=NEP_LAT_BOUNDS).sel(LONGITUDE=NEP_LONG_BOUNDS)

def plot_pcs(save_names, readable_labels, map_mask, obs=False):
    plt.grid()
    for i, save_name in enumerate(save_names):
        IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + ".nc"
        model = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)["IMPLICIT"]
        if MASK_TROPICS:
            model = model.where((model.LATITUDE > MASK_TROPICS_LATITUDE) | (model.LATITUDE < -1 * MASK_TROPICS_LATITUDE), np.nan)
        if readable_labels[i] == "No entrainment":
            model = mask_dataset(model, MASK_REGIONS)
            map_mask = mask_dataset(map_mask, MASK_REGIONS)
        print(readable_labels[i])
        pcs = get_eofs(model, 0, 1, map_mask=map_mask, standardise=True)[1]
        plt.plot(model.TIME.values / 12 + 2004, pcs, label=readable_labels[i])
    if obs:
        pcs = get_eofs(observed_anomaly, 0, 1, map_mask=map_mask, invert=False, standardise=True)[1]
        plt.plot(observed_anomaly.TIME.values / 12 + 2004, pcs, label="Observations")
    plt.legend()
    plt.show()

plot_pcs(save_names[0:8], readable_labels[0:8], obs=True, map_mask=map_mask)
plot_pcs(save_names[0:1], readable_labels[0:1], obs=True, map_mask=map_mask)
plot_pcs(save_names[5:6], readable_labels[5:6], obs=True, map_mask=map_mask)
plot_pcs(save_names[8:9], readable_labels[8:9], obs=True, map_mask=map_mask)
plot_pcs(save_names[9:10], readable_labels[9:10], obs=True, map_mask=map_mask)
plot_pcs(save_names[10:11], readable_labels[10:11], obs=True, map_mask=map_mask)


