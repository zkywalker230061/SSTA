import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from eofs.tools.standard import correlation_map
from scipy import stats

from Chris.utils import make_movie, get_eof, get_eof_with_nan_consideration, get_eof_from_ppca_py, get_save_name, \
    get_month_from_time
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_ENTRAINMENT = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True

SPLIT_SURFACE = True
INCLUDE_RADIATIVE_SURFACE = False
INCLUDE_TURBULENT_SURFACE = True

USE_DOWNLOADED_SSH = False
USE_OTHER_MLD = False
USE_MAX_GRADIENT_METHOD = True
USE_LOG_FOR_ENTRAINMENT = False
gamma_0 = 15.0

IMPLICIT_MODEL = True

MASK_TROPICS = True
MASK_TROPICS_LATITUDE = 15

CONSIDER_OBSERVATIONS = True
USE_REYNOLDS = True
# note: changed everything to functions to be cleaner; no longer need these conditionals
# MAKE_MOVIE = False
# PLOT_MODE_CONTRIBUTIONS = False
# PLOT_EOFS = False
# PLOT_PCs_OVER_TIME = False
# PLOT_ENSO = False
# MAKE_REGRESSION_MAPS = False
# TRACK_WARMING_EFFECTS = True

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION,
                          USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH, gamma0=gamma_0,
                          INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION ,OTHER_MLD=USE_OTHER_MLD, MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD, ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, LOG_ENTRAINMENT_VELOCITY=USE_LOG_FOR_ENTRAINMENT, SPLIT_SURFACE=SPLIT_SURFACE, INCLUDE_RADIATIVE_SURFACE=INCLUDE_RADIATIVE_SURFACE, INCLUDE_TURBULENT_SURFACE=INCLUDE_TURBULENT_SURFACE)
ALL_SCHEMES_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name + ".nc"
IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + ".nc"
DENOISED_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/cur_prev_denoised.nc"
OBSERVATIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
OBSERVATIONS_JJ_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/observed_anomaly_JJ.nc"
REYNOLDS_OBS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sst_anomalies-(2004-2018).nc"
ENSO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/nina34.anom.nc"

if IMPLICIT_MODEL:
    all_schemes_ds = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)
    print(all_schemes_ds["IMPLICIT"].values)
else:
    all_schemes_ds = xr.open_dataset(ALL_SCHEMES_DATA_PATH, decode_times=False)
    all_schemes_ds["CHRIS_PREV_CUR_NAN"] = all_schemes_ds["CHRIS_PREV_CUR"].where(
        (all_schemes_ds["CHRIS_PREV_CUR"] > -10) & (all_schemes_ds["CHRIS_PREV_CUR"] < 10))


argo_observations_ds = load_and_prepare_dataset(OBSERVATIONS_DATA_PATH)
processed_observations_ds = xr.open_dataset(OBSERVATIONS_JJ_DATA_PATH, decode_times=False)
if CONSIDER_OBSERVATIONS:
    if USE_REYNOLDS:
        obs_ds = xr.open_dataset(REYNOLDS_OBS_DATA_PATH, decode_times=False)
        observed_anomaly = obs_ds['anom']
    else:
        # observed_anomaly_before_processing = observations_ds["ARGO_TEMPERATURE_ANOMALY"].sel(PRESSURE=2.5)
        # observed_anomaly_before_processing_monthly_mean = get_monthly_mean(observed_anomaly_before_processing)
        # observations_ds = get_anomaly(observations_ds, "ARGO_TEMPERATURE_ANOMALY", observed_anomaly_before_processing_monthly_mean)
        # observed_anomaly = observations_ds["ARGO_TEMPERATURE_ANOMALY_ANOMALY"].sel(PRESSURE=2.5)
        observed_anomaly_before_processing = processed_observations_ds["__xarray_dataarray_variable__"]  # bad name...
        observed_anomaly_before_processing_monthly_mean = get_monthly_mean(observed_anomaly_before_processing)
        processed_observations_ds = get_anomaly(processed_observations_ds, "__xarray_dataarray_variable__",
                                                observed_anomaly_before_processing_monthly_mean)
        observed_anomaly = processed_observations_ds["__xarray_dataarray_variable___ANOMALY"]  # worse name...

    if MASK_TROPICS:
        observed_anomaly = observed_anomaly.where((observed_anomaly.LATITUDE > MASK_TROPICS_LATITUDE) | (
                    observed_anomaly.LATITUDE < -1 * MASK_TROPICS_LATITUDE), np.nan)

map_mask = argo_observations_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")

enso_indices_ds = xr.open_dataset(ENSO_DATA_PATH, decode_times=False)
enso_indices_ds = enso_indices_ds.assign_coords(time=np.arange(len(enso_indices_ds.time)))
enso_indices_ds = enso_indices_ds.sel(time=slice(672, 852))  # between 2004 and 2019
enso_indices_ds = enso_indices_ds.assign_coords(time=np.arange(len(enso_indices_ds.time)))

"""Plot results"""


def plot_full_model(to_plot="IMPLICIT", obs=False, save_path=None):
    if obs:
        make_movie(observed_anomaly, -3, 3, colorbar_label="Argo Anomaly", ENSO_ds=enso_indices_ds, savepath=save_path)
    else:
        if save_path is not None:
            make_movie(all_schemes_ds[to_plot], -3, 3, colorbar_label=(to_plot + " Scheme"), ENSO_ds=enso_indices_ds, savepath=save_path)
        else:
            make_movie(all_schemes_ds[to_plot], -3, 3, colorbar_label=(to_plot + " Scheme"), ENSO_ds=enso_indices_ds)

    # make_movie(all_schemes_ds["CHRIS_PREV_CUR"], -10, 10, colorbar_label="Chris Prev-Cur Scheme", ENSO_ds=enso_indices_ds)
    # make_movie(all_schemes_ds["CHRIS_MEAN_K"], -10, 10, colorbar_label="Chris Mean-k Scheme", ENSO_ds=enso_indices_ds)
    # make_movie(all_schemes_ds["CHRIS_PREV_K"], -10, 10, colorbar_label="Chris Prev-k Scheme", ENSO_ds=enso_indices_ds)
    # make_movie(all_schemes_ds["CHRIS_CAPPED_EXPONENT"], -10, 10, colorbar_label="Chris Capped Exponent Scheme", ENSO_ds=enso_indices_ds)
    # make_movie(all_schemes_ds["EXPLICIT"], -10, 10, colorbar_label="Explicit Scheme", ENSO_ds=enso_indices_ds)
    # make_movie(all_schemes_ds["IMPLICIT"], -3, 3, colorbar_label="Implicit Scheme", ENSO_ds=enso_indices_ds, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/videos/implicit" + save_name + ".mp4")
    # make_movie(all_schemes_ds["SEMI_IMPLICIT"], -10, 10, colorbar_label="Semi-Implicit Scheme", ENSO_ds=enso_indices_ds)
    # make_movie(all_schemes_ds["CHRIS_PREV_CUR_CLEAN"], -3, 3, colorbar_label="Chris Prev-Cur Scheme Denoised", ENSO_ds=enso_indices_ds, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/videos/chris_clean" + save_name + ".mp4")
    # make_movie(observed_anomaly, -3, 3, colorbar_label="Argo Anomaly", ENSO_ds=enso_indices_ds, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/videos/observations.mp4")


"""EOF analysis"""
# get EOF modes and PCs for a given model
start_mode = 0
end_mode = 3
to_plot_name = "IMPLICIT"
to_plot = all_schemes_ds[to_plot_name]
if MASK_TROPICS:
    to_plot = to_plot.where(
        (to_plot.LATITUDE > MASK_TROPICS_LATITUDE) | (to_plot.LATITUDE < -1 * MASK_TROPICS_LATITUDE), np.nan)
monthly_mean = get_monthly_mean(to_plot)
eof_modes, explained_variance, PCs, EOFs = get_eof_with_nan_consideration(to_plot, modes=end_mode, mask=map_mask,
                                                                          tolerance=1e-15, monthly_mean_ds=monthly_mean,
                                                                          start_mode=start_mode, max_iterations=4)
#eof_modes.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/implicit_" + save_name + "_" + str(end_mode) + "eofs.nc")
PCs = PCs * -1
EOFs = EOFs * -1
PCs_standard = (PCs - PCs.mean(axis=0)) / PCs.std(axis=0)  # standardise

if CONSIDER_OBSERVATIONS:  # get EOF modes and PCs for the observations
    monthly_mean_obs = get_monthly_mean(observed_anomaly)
    eof_modes_obs, explained_variance_obs, PCs_obs, EOFs_obs = get_eof_with_nan_consideration(observed_anomaly,
                                                                                              modes=end_mode,
                                                                                              mask=map_mask,
                                                                                              tolerance=1e-15,
                                                                                              monthly_mean_ds=monthly_mean_obs,
                                                                                              start_mode=start_mode)
    PCs_obs_standard = (PCs_obs - PCs_obs.mean(axis=0)) / PCs_obs.std(axis=0)


# components, explained_variance, PCs = get_eof(to_plot, end_mode, mask=map_mask, clean_nan=True)
# components_obs, explained_variance_obs, PCs_obs = get_eof(observed_anomaly, end_mode, mask=map_mask, clean_nan=True)
# k = 0
# eof_modes = components.isel(mode=k) * PCs.isel(mode=k)
# eof_modes_obs = components_obs.isel(mode=k) * PCs_obs.isel(mode=k)


def plot_enso():  # make a plot of ENSO index over time
    plt.grid()
    plt.plot(enso_indices_ds.time.values / 12, enso_indices_ds.value.values, marker='x')
    plt.xlabel("Time (years since 2004)")
    plt.ylabel("ENSO index")
    plt.savefig("../results/enso.jpg", dpi=400)
    plt.show()


print("Variance explained by modes " + str(start_mode) + " to " + str(end_mode) + ": " + str(
    explained_variance[start_mode:end_mode].sum().item()))


def eof_movie():  # movie of EOFxPC modes over time
    make_movie(eof_modes, -3, 3, "Anomaly from EOF Modes " + str(start_mode) + " to " + str(end_mode),
               ENSO_ds=enso_indices_ds)
    #make_movie(eof_modes, -3, 3, "Anomaly from EOF Modes " + str(start_mode) + " to " + str(end_mode), ENSO_ds=enso_indices_ds, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/videos/implicit_" + save_name + "_" + str(end_mode) + "eofs.mp4")
    if CONSIDER_OBSERVATIONS:
        make_movie(eof_modes_obs, -1.5, 1.5, "Anomaly from EOF Modes " + str(start_mode) + " to " + str(end_mode),
                   ENSO_ds=enso_indices_ds)


def explained_variance_from_each_mode():  # plot explained variance from each mode
    # plot contribution of each mode
    modes = np.arange(start_mode, start_mode + len(explained_variance))
    max_limit = start_mode + len(explained_variance)  # e.g. for explicit scheme, cap at some small number
    cumulative_explained_variance = np.cumsum(explained_variance)

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True)
    fig.suptitle("Explained variance by modes of " + to_plot_name + " scheme")
    ax1.grid()
    ax1.plot(modes[:max_limit], explained_variance[:max_limit], marker='x')
    ax1.set_ylabel("Explained variance")

    ax2.grid()
    ax2.plot(modes[:max_limit], cumulative_explained_variance[:max_limit], marker='x')
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Cumulative explained variance")

    # same in log scale
    ax3.grid()
    ax3.plot(modes[:max_limit], explained_variance[:max_limit], marker='x')
    ax3.set_yscale("log")

    ax4.grid()
    ax4.plot(modes[:max_limit], cumulative_explained_variance[:max_limit], marker='x')
    ax4.set_xlabel("Mode")
    ax4.set_yscale("log")
    plt.savefig("../results/explained_variance_" + to_plot_name + ".jpg", dpi=400)
    plt.show()


def plot_spatial_pattern_EOFs():  # plot EOFs (spatial patterns) for the first k modes
    k_range = 3
    fig, axs = plt.subplots(k_range, 1)
    fig.suptitle("EOF of " + to_plot_name + " scheme")
    fig.tight_layout()
    norm = colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    for k in range(k_range):
        axs[k].grid()
        EOF_standard = (EOFs.isel(MODE=k) - EOFs.isel(MODE=k).mean(dim=["LATITUDE", "LONGITUDE"])) / EOFs.isel(
            MODE=k).std(dim=["LATITUDE", "LONGITUDE"])
        pcolormesh = axs[k].pcolormesh(EOFs.LONGITUDE.values, EOFs.LATITUDE.values, EOF_standard, cmap='RdBu_r',
                                       norm=norm)
        if k == k_range - 1:
            axs[k].set_xlabel("Longitude")
        axs[k].set_ylabel("Latitude")
    cbar = fig.colorbar(pcolormesh, ax=axs, label="EOF spatial pattern (standardised)")
    #pcolormesh.set_clim(vmin=-10, vmax=10)
    # plt.savefig("../results/eof_spatial_" + to_plot_name + ".jpg", dpi=400)
    plt.show()
    if CONSIDER_OBSERVATIONS:
        fig, axs = plt.subplots(k_range, 1)
        fig.suptitle("EOF of observations")
        fig.tight_layout()
        norm = colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
        for k in range(k_range):
            axs[k].grid()
            EOF_obs_standard = (EOFs_obs.isel(MODE=k) - EOFs_obs.isel(MODE=k).mean(
                dim=["LATITUDE", "LONGITUDE"])) / EOFs_obs.isel(MODE=k).std(dim=["LATITUDE", "LONGITUDE"])
            pcolormesh = axs[k].pcolormesh(EOFs_obs.LONGITUDE.values, EOFs_obs.LATITUDE.values, EOF_obs_standard,
                                           cmap='RdBu_r', norm=norm)
            if k == k_range - 1:
                axs[k].set_xlabel("Longitude")
            axs[k].set_ylabel("Latitude")
        cbar = fig.colorbar(pcolormesh, ax=axs, label="EOF spatial pattern (standardised)")
        #pcolormesh.set_clim(vmin=-10, vmax=10)
        # plt.savefig("../results/eof_spatial_obs.jpg", dpi=400)
        plt.show()

# plot PCs over time
def plot_PCs_over_time():
    if CONSIDER_OBSERVATIONS:
        k_range = 3
        fig, axs = plt.subplots(k_range, 1)
        fig.suptitle("PCs of " + to_plot_name + " scheme")
        fig.tight_layout()
        for k in range(k_range):
            # if k>0:
            #     PCs[:, k] = PCs[:, k] * -1
            # NOTE: RMSE of mode 2 (//k==1) decrease from 1.6 to 1.2 when doing this. However it's hard to argue they track
            pcs_rmse = np.sqrt(np.mean((PCs_standard[:, k] - PCs_obs_standard[:, k]) ** 2))
            print("RMSE of mode " + str(k + 1) + ": " + str(pcs_rmse))
            axs[k].grid()
            axs[k].set_title("RMSE of mode " + str(k + 1) + ": " + str(pcs_rmse))
            axs[k].plot(to_plot.coords["TIME"].values / 12, PCs_standard[:, k], label="Simulated mode " + str(k + 1))
            axs[k].plot(to_plot.coords["TIME"].values / 12, PCs_obs_standard[:, k], label="Observed mode " + str(k + 1))
            if k == k_range - 1:
                axs[k].set_xlabel("Time (years since January 2004)")
            axs[k].set_ylabel("PC")
            axs[k].legend()
        plt.savefig("../results/pcs_" + to_plot_name + ".jpg", dpi=400)
        plt.show()

        # plot difference in PCs over time
        fig, axs = plt.subplots(k_range, 1)
        fig.suptitle("Difference in PCs of " + to_plot_name + " scheme")
        for k in range(k_range):
            axs[k].grid()
            axs[k].plot(to_plot.coords["TIME"].values / 12, PCs_standard[:, k] - PCs_obs_standard[:, k],
                        label="Error mode " + str(k + 1))
            if k == k_range - 1:
                axs[k].set_xlabel("Time (months since January 2004)")
            axs[k].set_ylabel("PC error")
            axs[k].legend()
        plt.savefig("../results/pcs_difference_" + to_plot_name + ".jpg", dpi=400)
        plt.show()
    else:
        print("Requires `consider_observations`")

# regression map
def regression_map():
    field = to_plot - to_plot.mean(dim='TIME')
    correlation_maps = correlation_map(PCs_standard, field.values) * field.std(dim='TIME').values
    regression_maps = xr.DataArray(np.stack(correlation_maps), dims=['MODE', 'LATITUDE', 'LONGITUDE'],
                                   coords={'MODE': np.arange(PCs.shape[1]), 'LATITUDE': to_plot.LATITUDE,
                                           'LONGITUDE': to_plot.LONGITUDE})

    k_range = PCs.shape[1]
    fig, axs = plt.subplots(PCs.shape[1], 1)
    fig.suptitle("Regression Map of " + to_plot_name + " scheme")
    fig.tight_layout()
    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for k in range(PCs.shape[1]):
        axs[k].grid()
        pcolormesh = axs[k].pcolormesh(regression_maps.LONGITUDE.values, regression_maps.LATITUDE.values,
                                       regression_maps.isel(MODE=k), cmap='RdBu_r', norm=norm)
        if k == PCs.shape[1] - 1:
            axs[k].set_xlabel("Longitude")
        axs[k].set_ylabel("Latitude")
    cbar = fig.colorbar(pcolormesh, ax=axs, label="Temperature per PC from Regression (K)")
    #pcolormesh.set_clim(vmin=-2, vmax=2)
    plt.savefig("../results/regression_map_" + to_plot_name + ".jpg", dpi=400)
    plt.show()

    if CONSIDER_OBSERVATIONS:
        field_obs = observed_anomaly - observed_anomaly.mean(dim='TIME')
        correlation_maps_obs = correlation_map(PCs_obs_standard, field_obs.values) * field_obs.std(dim='TIME').values
        regression_maps_obs = xr.DataArray(np.stack(correlation_maps_obs), dims=['MODE', 'LATITUDE', 'LONGITUDE'],
                                           coords={'MODE': np.arange(PCs_obs.shape[1]), 'LATITUDE': to_plot.LATITUDE,
                                                   'LONGITUDE': to_plot.LONGITUDE})

        k_range = PCs_obs.shape[1]
        fig, axs = plt.subplots(PCs_obs.shape[1], 1)
        fig.suptitle("Regression Map of Observations")
        fig.tight_layout()
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        for k in range(PCs_obs.shape[1]):
            axs[k].grid()
            pcolormesh = axs[k].pcolormesh(regression_maps_obs.LONGITUDE.values, regression_maps_obs.LATITUDE.values,
                                           regression_maps_obs.isel(MODE=k), cmap='RdBu_r', norm=norm)
            if k == PCs_obs.shape[1] - 1:
                axs[k].set_xlabel("Longitude")
            axs[k].set_ylabel("Latitude")
        cbar = fig.colorbar(pcolormesh, ax=axs, label="Temperature per PC from Regression (K)")
        #pcolormesh.set_clim(vmin=-5, vmax=5)
        plt.savefig("../results/regression_map_obs.jpg", dpi=400)
        plt.show()

"""Plot mean anomaly over time to see if there are warming effects"""
def track_warming_effects():
    if CONSIDER_OBSERVATIONS:
        times = []
        mean_anomalies = []
        mean_obs_anomalies = []
        abs_mean_anomalies = []
        abs_mean_obs_anomalies = []
        for time in to_plot.TIME.values:
            mean_anomalies.append(to_plot.sel(TIME=time).mean(skipna=True).item())
            abs_mean_anomalies.append(abs(to_plot.sel(TIME=time)).mean(skipna=True).item())
            mean_obs_anomalies.append(observed_anomaly.sel(TIME=time).mean(skipna=True).item())
            abs_mean_obs_anomalies.append(abs(observed_anomaly.sel(TIME=time)).mean(skipna=True).item())
            times.append((time - 0.5) / 12 + 2004)
        plt.grid()
        plt.scatter(times, mean_anomalies, marker='x', label="Modelled mean anomaly")
        plt.scatter(times, mean_obs_anomalies, marker='x', label="Observed mean anomaly")
        plt.xlabel("Year")
        plt.ylabel("Mean anomaly (ºC)")
        plt.title("Mean anomaly over time")
        plt.legend()
        plt.show()
        plt.grid()
        plt.scatter(times, abs_mean_anomalies, marker='x', label="Modelled mean absolute anomaly")
        plt.scatter(times, abs_mean_obs_anomalies, marker='x', label="Observed mean absolute anomaly")
        plt.xlabel("Year")
        plt.ylabel("Mean absolute anomaly (ºC)")
        plt.title("Mean absolute anomaly over time")
        plt.legend()
        plt.show()

        # t-test statistical test
        t_stat_mean_anomalies, p_val_mean_anomalies = stats.ttest_ind(a=mean_anomalies, b=mean_obs_anomalies)
        t_stat_abs_mean_anomalies, p_val_abs_mean_anomalies = stats.ttest_ind(a=abs_mean_anomalies,
                                                                              b=abs_mean_obs_anomalies)
        print("t-statistic for mean anomalies: " + str(t_stat_mean_anomalies))
        print("p-value for mean anomalies: " + str(p_val_mean_anomalies))
        print()
        print("t-statistic for absolute mean anomalies: " + str(t_stat_abs_mean_anomalies))
        print("p-value for absolute mean anomalies: " + str(p_val_abs_mean_anomalies))

    else:
        print("TRACK_WARMING_EFFECTS requires CONSIDER_OBSERVATIONS=True")


def track_anomaly_persistence(lag, start_month=None):
    to_plot_no_unchanging_cells = to_plot.where(to_plot.std("TIME") != 0)     # remove parts where there is no change to avoid spuriously strong autocorrelation signals

    def get_autocorrelation(start_da, lag_da=None, max_lag=36):
        if lag_da is None:      # the only reason they might differ is if start_da only contains results of a certain month
            lag_da = start_da
        autocorrelation_functions = []
        for lag in range(1, max_lag + 1):
            correlated_months = (xr.ufuncs.isfinite(start_da) & xr.ufuncs.isfinite(lag_da.shift(TIME=lag))).sum(dim="TIME")   # check that enough data have gone into the correlation calculation
            autocorrelation_functions.append(xr.corr(start_da, lag_da.shift(TIME=lag).sel(TIME=start_da.TIME), dim='TIME').where(correlated_months >= max_lag/2))       # require certain amount of correlation data
        autocorrelation_functions_ds = xr.concat(autocorrelation_functions, dim="LAG")
        autocorrelation_functions_ds = autocorrelation_functions_ds.assign_coords(LAG=list(range(1, max_lag + 1)))
        return autocorrelation_functions_ds

    # get persistence given a particular starting month
    def start_at_month(da, month):
        data_at_this_month = []
        for time in da.TIME.values:
            month_of_this_time = get_month_from_time(time)
            if month_of_this_time == month:
                data_at_this_month.append(da.sel(TIME=time))
        return xr.concat(data_at_this_month, dim="TIME")

    def plot_autocorrelation(acf, lag, model_name, acf_obs=None):
        plt.figure()
        acf.sel(LAG=lag).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=-1, vmax=1)
        plt.title("Autocorrelation with lag=" + str(lag) + " of " + str(model_name))
        plt.show()

        mean_autocorrelation = acf.mean(("LATITUDE", "LONGITUDE"))
        print(mean_autocorrelation.sel(LAG=3).values)
        print(mean_autocorrelation.sel(LAG=6).values)
        plt.figure()
        plt.grid()
        plt.plot(mean_autocorrelation.LAG.values, mean_autocorrelation.values, label=model_name)
        if acf_obs is not None:
            mean_autocorrelation_obs = acf_obs.mean(("LATITUDE", "LONGITUDE"))
            plt.plot(mean_autocorrelation_obs.LAG.values, mean_autocorrelation_obs.values, label="Observations")
            plt.title("Autocorrelation Mean of " + str(model_name))
            plt.legend()
        else:
            plt.title("Autocorrelation Mean of " + str(model_name))
        plt.xlabel("Lag (months)")
        plt.ylabel("Autocorrelation")
        plt.show()

    if start_month is None:
        autocorrelation_functions_ds = get_autocorrelation(to_plot_no_unchanging_cells, max_lag=36)  # all time
    else:
        autocorrelation_functions_ds = get_autocorrelation(start_at_month(to_plot_no_unchanging_cells, start_month), lag_da=to_plot_no_unchanging_cells, max_lag=12)

    if CONSIDER_OBSERVATIONS:
        obs_no_unchanging_cells = observed_anomaly.where(to_plot.std("TIME") != 0)
        if start_month is None:
            obs_autocorrelation_functions_ds = get_autocorrelation(obs_no_unchanging_cells, max_lag=36)  # all time
        else:
            obs_autocorrelation_functions_ds = get_autocorrelation(start_at_month(obs_no_unchanging_cells, start_month), lag_da=obs_no_unchanging_cells, max_lag=12)
        plot_autocorrelation(autocorrelation_functions_ds, lag, "Implicit Model", obs_autocorrelation_functions_ds)
        #plot_autocorrelation(obs_autocorrelation_functions_ds, lag, "observations")
    else:
        plot_autocorrelation(autocorrelation_functions_ds, lag, "implicit model")


def plot_correlation():
    if CONSIDER_OBSERVATIONS:
        correlation = xr.corr(to_plot, observed_anomaly, dim='TIME')
        print(correlation.mean().values)
        correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
        plt.title("")
        plt.xlabel("Longitude (º)")
        plt.ylabel("Latitude (º)")
        cbar = plt.gcf().axes[-1]
        cbar.set_ylabel('Pearson Correlation Coefficient', rotation=90)
        plt.title("Correlation between model and SST observation")
        #plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/correlations/" + save_name + "_correlation.jpg")
        #plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_poster/full_model_correlation.png", dpi=400)
        plt.show()
    else:
        print("PLOT_CORRELATION requires CONSIDER_OBSERVATIONS=True")

# plot_full_model(save_path="/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/videos/" + save_name + ".mp4")
# plot_full_model(obs=True, save_path="/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/videos/Reynolds_observations.mp4")

# plot_full_model()
# plot_enso()
# eof_movie()
# explained_variance_from_each_mode()
plot_spatial_pattern_EOFs()
# plot_PCs_over_time()
# regression_map()
# track_warming_effects()
# track_anomaly_persistence(6, 9)
plot_correlation()
