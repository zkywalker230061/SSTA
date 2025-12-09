import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie, get_eof, get_eof_with_nan_consideration, get_eof_from_ppca_py
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

ALL_SCHEMES_DATA_PATH = "../datasets/all_anomalies.nc"
DENOISED_DATA_PATH = "../datasets/cur_prev_denoised.nc"
OBSERVATIONS_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"
CONSIDER_OBSERVATIONS = True
MAKE_MOVIE = True
PLOT_MODE_CONTRIBUTIONS = True

all_schemes_ds = xr.open_dataset(ALL_SCHEMES_DATA_PATH, decode_times=False)
denoised_scheme_ds = xr.open_dataset(DENOISED_DATA_PATH, decode_times=False)
all_schemes_ds["CHRIS_PREV_CUR_NAN"] = all_schemes_ds["CHRIS_PREV_CUR"].where((all_schemes_ds["CHRIS_PREV_CUR"] > -10) & (all_schemes_ds["CHRIS_PREV_CUR"] < 10))

observations_ds = load_and_prepare_dataset(OBSERVATIONS_DATA_PATH)
if CONSIDER_OBSERVATIONS:
    observed_anomaly_before_processing = observations_ds["ARGO_TEMPERATURE_ANOMALY"].sel(PRESSURE=2.5)
    observed_anomaly_before_processing_monthly_mean = get_monthly_mean(observed_anomaly_before_processing)
    observations_ds = get_anomaly(observations_ds, "ARGO_TEMPERATURE_ANOMALY", observed_anomaly_before_processing_monthly_mean)
    observed_anomaly = observations_ds["ARGO_TEMPERATURE_ANOMALY_ANOMALY"].sel(PRESSURE=2.5)

map_mask = observations_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")

"""Plot results"""
# make_movie(all_schemes_ds["CHRIS_PREV_CUR"], -10, 10, colorbar_label="Chris Prev-Cur Scheme")
# make_movie(all_schemes_ds["CHRIS_MEAN_K"], -10, 10, colorbar_label="Chris Mean-k Scheme")
# make_movie(all_schemes_ds["CHRIS_PREV_K"], -10, 10, colorbar_label="Chris Prev-k Scheme")
# make_movie(all_schemes_ds["CHRIS_CAPPED_EXPONENT"], -10, 10, colorbar_label="Chris Capped Exponent Scheme")
# make_movie(all_schemes_ds["EXPLICIT"], -10, 10, colorbar_label="Explicit Scheme")
# make_movie(all_schemes_ds["IMPLICIT"], -4, 4, colorbar_label="Implicit Scheme")
# make_movie(all_schemes_ds["SEMI_IMPLICIT"], -10, 10, colorbar_label="Semi-Implicit Scheme")
# make_movie(denoised_scheme_ds["EMEOF_DENOISED_ANOMALY"], -10, 10, colorbar_label="Chris Prev-Cur Scheme Denoised")
# make_movie(observed_anomaly, -10, 10, colorbar_label="Argo Anomaly")


"""EOF analysis"""
# plot EOF modes in a defined range
start_mode = 0
end_mode = 3
to_plot_name = "IMPLICIT"
to_plot = all_schemes_ds[to_plot_name]
#to_plot = denoised_scheme_ds["EMEOF_DENOISED_ANOMALY"]
monthly_mean = get_monthly_mean(to_plot)
eof_modes, explained_variance, PCs = get_eof_with_nan_consideration(to_plot, modes=end_mode, mask=map_mask, tolerance=1e-15, monthly_mean_ds=monthly_mean, start_mode=start_mode, max_iterations=4)
if CONSIDER_OBSERVATIONS:
    eof_modes_obs, explained_variance_obs, PCs_obs = get_eof_with_nan_consideration(observed_anomaly, modes=end_mode, mask=map_mask, tolerance=1e-15, monthly_mean_ds=monthly_mean, start_mode=start_mode)

# components, explained_variance, PCs = get_eof(to_plot, end_mode, mask=map_mask, clean_nan=True)
# components_obs, explained_variance_obs, PCs_obs = get_eof(observed_anomaly, end_mode, mask=map_mask, clean_nan=True)
# k = 0
# eof_modes = components.isel(mode=k) * PCs.isel(mode=k)
# eof_modes_obs = components_obs.isel(mode=k) * PCs_obs.isel(mode=k)
print("Variance explained by modes " + str(start_mode) + " to " + str(end_mode) + ": " + str(explained_variance[start_mode:end_mode].sum().item()))
if MAKE_MOVIE:
    make_movie(eof_modes, -1.5, 1.5, "Anomaly from EOF Modes " + str(start_mode) + " to " + str(end_mode))
    if CONSIDER_OBSERVATIONS:
        make_movie(eof_modes_obs, -1.5, 1.5, "Anomaly from EOF Modes " + str(start_mode) + " to " + str(end_mode))

if PLOT_MODE_CONTRIBUTIONS:
    # plot contribution of each mode
    modes = np.arange(start_mode, start_mode+len(explained_variance))
    max_limit = start_mode+len(explained_variance)      # e.g. for explicit scheme, cap at some small number
    cumulative_explained_variance = np.cumsum(explained_variance)

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=True)
    fig.suptitle("Explained variance by modes of " + to_plot_name + " scheme")
    ax1.grid()
    ax1.plot(modes[:max_limit], explained_variance[:max_limit], marker='x')
    #ax1.set_xlabel("Mode")
    ax1.set_ylabel("Explained variance")
    #ax1.show()

    ax2.grid()
    ax2.plot(modes[:max_limit], cumulative_explained_variance[:max_limit], marker='x')
    ax2.set_xlabel("Mode")
    ax2.set_ylabel("Cumulative explained variance")
    #ax2.show()

    # same in log scale
    ax3.grid()
    ax3.plot(modes[:max_limit], explained_variance[:max_limit], marker='x')
    #ax3.set_xlabel("Mode")
    #ax3.set_ylabel("Explained variance")
    ax3.set_yscale("log")
    #ax3.show()

    ax4.grid()
    ax4.plot(modes[:max_limit], cumulative_explained_variance[:max_limit], marker='x')
    ax4.set_xlabel("Mode")
    #ax4.set_ylabel("Cumulative explained variance")
    ax4.set_yscale("log")
    plt.show()

# plot PCs over time
if CONSIDER_OBSERVATIONS:
    k_range = 3
    fig, axs = plt.subplots(k_range, 1)
    fig.suptitle("PCs of " + to_plot_name + " scheme")
    fig.tight_layout()
    for k in range(k_range):
        pcs_rmse = np.sqrt(np.mean((PCs[:, k] - PCs_obs[:, k]) ** 2))
        print("RMSE of mode " + str(k+1) + ": " + str(pcs_rmse))
        axs[k].grid()
        axs[k].set_title("RMSE of mode " + str(k+1) + ": " + str(pcs_rmse))
        axs[k].plot(to_plot.coords["TIME"].values, PCs[:, k], label="Simulated mode " + str(k+1))
        axs[k].plot(to_plot.coords["TIME"].values, PCs_obs[:, k], label="Observed mode " + str(k+1))
        if k == k_range-1:
            axs[k].set_xlabel("Time (months since January 2004)")
        axs[k].set_ylabel("PC")
        axs[k].legend()
        #plt.show()
    plt.show()

    # plot difference in PCs over time
    fig, axs = plt.subplots(k_range, 1)
    fig.suptitle("Difference in PCs of " + to_plot_name + " scheme")
    for k in range(k_range):
        axs[k].grid()
        axs[k].plot(to_plot.coords["TIME"].values, PCs[:, k] - PCs_obs[:, k], label="Error mode " + str(k+1))
        if k == k_range - 1:
            axs[k].set_xlabel("Time (months since January 2004)")
        axs[k].set_ylabel("PC error")
        axs[k].legend()
        #plt.show()
    plt.show()

# temp_anomaly_eof.to_netcdf("../datasets/chris_prev_cur_scheme_denoised.nc")
