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
from Chris.utils import make_movie, get_eof, get_eof_with_nan_consideration, get_eof_from_ppca_py, get_save_name, \
    get_month_from_time, format_cartopy, mask_dataset, get_autocorrelation, start_at_month, plot_autocorrelation, normalised_rmse
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from correlation_significance import mutual_information

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = False
INCLUDE_ENTRAINMENT = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = False

SPLIT_SURFACE = True
INCLUDE_RADIATIVE_SURFACE = True
INCLUDE_TURBULENT_SURFACE = True

OBSERVATIONS_2025_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/reynolds_sst_Anomalies-(2004-2025)_no_2004.nc"
obs_ds = xr.open_dataset(OBSERVATIONS_2025_DATA_PATH, decode_times=False)
observed_anomaly = obs_ds['ANOMALY_SST']

def compare_gammas(gamma_list, obs):
    mean_rms = []
    mean_abs_values = []
    nrmse = []
    mean_nrmse = []
    autocorrelations = []
    for gamma in gamma_list:
        model_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION, USE_DOWNLOADED_SSH=False, gamma0=gamma, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, LOG_ENTRAINMENT_VELOCITY=False, SPLIT_SURFACE=SPLIT_SURFACE, INCLUDE_RADIATIVE_SURFACE=INCLUDE_RADIATIVE_SURFACE, INCLUDE_TURBULENT_SURFACE=INCLUDE_TURBULENT_SURFACE, DATA_TO_2025=True, adjust_mld=10)
        model_path = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + model_name + ".nc"
        model = xr.open_dataset(model_path, decode_times=False)["IMPLICIT"]
        # make_movie(model, vmin=-3, vmax=3)
        mean_rms.append(np.sqrt((model**2).mean(dim="TIME")).mean().values)
        mean_abs_values.append(abs(model).mean().values)
        nrmse.append(normalised_rmse(model, obs))
        mean_nrmse.append(normalised_rmse(model, obs).mean().values)
        autocorrelations.append(get_autocorrelation(model, max_lag=15))
    print("Obs results")
    print("Mean RMS")
    print(np.sqrt((obs**2).mean(dim="TIME")).mean().values)
    print("Mean absolute value")
    print(abs(obs).mean().values)
    obs_autocorrelation = get_autocorrelation(obs, max_lag=15)

    print(mean_rms)
    print(mean_abs_values)
    print(nrmse)
    print(nrmse[0])
    print(mean_nrmse)

    # plt.grid()
    # plt.scatter(mld_list, mean_rms)
    # plt.xlabel("Mixed Layer Depth Adjustment (dbar)")
    # plt.ylabel("Mean Model RMS")
    # plt.show()
    #
    # plt.grid()
    # plt.scatter(mld_list, mean_abs_values)
    # plt.xlabel("Mixed Layer Depth Adjustment (dbar)")
    # plt.ylabel("Mean Model Absolute Value")
    # plt.show()
    #
    # plt.grid()
    # plt.scatter(mld_list, mean_nrmse)
    # plt.xlabel("Mixed Layer Depth Adjustment (dbar)")
    # plt.ylabel("Mean Model NRMSE with respect to Observations")
    # plt.show()

    # for i, mld in enumerate(mld_list):
    #     nrmse_at_this_mld = nrmse[i]
    #     fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    #     nrmse_at_this_mld.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=2)
    #     ax = format_cartopy(ax)
    #     cbar = plt.gcf().axes[-1]
    #     cbar.set_ylabel('Normalised RMSE', rotation=90)
    #     ax.set_xlabel("Longitude (º)")
    #     ax.set_ylabel("Latitude (º)")
    #     plt.show()

    fig, ax = plt.subplots()
    plt.grid()
    plot_autocorrelation(obs_autocorrelation, lag=None, model_name="Implicit Model", show=False, label="Observations", thicken=True, blackline=True)
    for i, gamma in enumerate(gamma_list):
        autocorrelation = autocorrelations[i]
        label = "γ = " + str(gamma)
        plot_autocorrelation(autocorrelation, lag=None, model_name="Implicit Model", show=False, label=label)
        # if mld == 0.0:
        #     plot_autocorrelation(autocorrelation, lag=None, model_name="Implicit Model", show=False, label=label, thicken=True)
        # else:
        #     plot_autocorrelation(autocorrelation, lag=None, model_name="Implicit Model", show=False, label=label)
    plt.legend()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_report/gamma_sensitivity_at_10m_deeper_MLD.png", dpi=400, transparent=True, bbox_inches='tight')
    #plt.show()

gamma_list = [2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
# mld_list = [1.0, 0.0]
compare_gammas(gamma_list, observed_anomaly)
