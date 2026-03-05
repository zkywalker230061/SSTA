import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from eofs.tools.standard import correlation_map
from scipy import stats

from SSTA.Chris.utils import make_movie, get_eof, get_eof_with_nan_consideration, get_eof_from_ppca_py, get_save_name, \
    get_month_from_time
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

INCLUDE_SURFACE_1 = True
INCLUDE_EKMAN_ANOM_ADVECTION_1 = True
INCLUDE_EKMAN_MEAN_ADVECTION_1 = False
INCLUDE_ENTRAINMENT_1 = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING_1 = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION_1 = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION_1 = False

USE_DOWNLOADED_SSH_1 = False
USE_OTHER_MLD_1 = False
gamma_0_1 = 15.0
IMPLICIT_MODEL_1 = True
USE_MAX_GRADIENT_METHOD_1 = True

INCLUDE_SURFACE_2 = True
INCLUDE_EKMAN_ANOM_ADVECTION_2 = True
INCLUDE_EKMAN_MEAN_ADVECTION_2 = True
INCLUDE_ENTRAINMENT_2 = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING_2 = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION_2 = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION_2 = True

USE_DOWNLOADED_SSH_2 = False
USE_OTHER_MLD_2 = False
gamma_0_2 = 15.0
IMPLICIT_MODEL_2 = True
USE_MAX_GRADIENT_METHOD_2 = True

save_name_1 = get_save_name(INCLUDE_SURFACE_1, INCLUDE_EKMAN_ANOM_ADVECTION_1, INCLUDE_ENTRAINMENT_1, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION_1,
                          USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH_1, gamma0=gamma_0_1,
                          INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION_1, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION_1, OTHER_MLD=USE_OTHER_MLD_1, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD_1, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING_1)

save_name_2 = get_save_name(INCLUDE_SURFACE_2, INCLUDE_EKMAN_ANOM_ADVECTION_2, INCLUDE_ENTRAINMENT_2, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION_2,
                          USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH_2, gamma0=gamma_0_2,
                          INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION_2, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION_2, OTHER_MLD=USE_OTHER_MLD_2, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD_2, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING_2)

ALL_SCHEMES_DATA_PATH_1 = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name_1 + ".nc"
IMPLICIT_SCHEME_DATA_PATH_1 = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name_1 + ".nc"
ALL_SCHEMES_DATA_PATH_2 = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name_2 + ".nc"
IMPLICIT_SCHEME_DATA_PATH_2 = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name_2 + ".nc"

if IMPLICIT_MODEL_1:
    all_schemes_ds_1 = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH_1, decode_times=False)
else:
    all_schemes_ds_1 = xr.open_dataset(ALL_SCHEMES_DATA_PATH_1, decode_times=False)

if IMPLICIT_MODEL_2:
    all_schemes_ds_2 = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH_2, decode_times=False)
else:
    all_schemes_ds_2 = xr.open_dataset(ALL_SCHEMES_DATA_PATH_2, decode_times=False)

correlation = xr.corr(all_schemes_ds_1["IMPLICIT"], all_schemes_ds_2["IMPLICIT"], dim='TIME')
correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=1)
plt.title("")
plt.xlabel("Longitude (º)")
plt.ylabel("Latitude (º)")
cbar = plt.gcf().axes[-1]
cbar.set_ylabel('Pearson Correlation Coefficient', rotation=270, labelpad=15)
#plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_poster/self_correlations/no_geostrophic.png", dpi=400)
plt.show()
