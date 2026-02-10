import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import get_month_from_time
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
OBSERVATIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
OBSERVATIONS_JJ_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/observed_anomaly_JJ.nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
OTHER_H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/other_h_bar.nc"
#MAX_GRADIENT_H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/Tsub_Max_Gradient_Method_New_h.nc"

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
observations_ds = load_and_prepare_dataset(OBSERVATIONS_DATA_PATH)
processed_observations_ds = xr.open_dataset(OBSERVATIONS_JJ_DATA_PATH, decode_times=False)
tm_observed = processed_observations_ds["__xarray_dataarray_variable__"]  # bad name...
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
other_hbar_ds = xr.open_dataset(OTHER_H_BAR_DATA_PATH, decode_times=False)
#max_grad_hbar_ds = xr.open_dataset(MAX_GRADIENT_H_BAR_DATA_PATH, decode_times=False)

# t_sub_ds['T_sub'] = t_sub_ds['SUB_TEMPERATURE']
# t_sub_monthly_mean = get_monthly_mean(t_sub_ds['T_sub'])
# t_sub_ds = get_anomaly(t_sub_ds, 'T_sub', t_sub_monthly_mean)
# t_sub_ds.to_netcdf("../datasets/t_sub_2.nc")

# print(t_sub_ds["T_sub_ANOMALY"].max().item())
# print(t_sub_ds["T_sub_ANOMALY"].min().item())
# print(abs(t_sub_ds["T_sub_ANOMALY"]).mean().item())


def make_a_lot_of_plots():
    vmin_anom = -2
    vmax_anom = 2

    t_sub_ds['T_sub'].sel(TIME=132.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=0, vmax=30)
    plt.show()

    t_sub_ds['T_sub'].sel(TIME=138.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=0, vmax=30)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=0.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=6.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=60.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=66.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=132.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=135.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=138.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=141.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=168.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=171.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=174.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()

    t_sub_ds['T_sub_ANOMALY'].sel(TIME=177.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
    plt.show()


def correlate_tsub(observations_ds, tm_observed, hbar_ds, nh=True):
    max_depth = 300

    observed_temps = observations_ds["ARGO_TEMPERATURE_ANOMALY"].sel(PRESSURE=slice(0, max_depth))

    times = observed_temps.TIME.values
    months = []
    for time in times:
        month = get_month_from_time(time)
        months.append(month)
    observed_temps = observed_temps.assign_coords(MONTH=("TIME", months))
    tm_observed = tm_observed.assign_coords(MONTH=("TIME", months))

    depths = observed_temps.PRESSURE.values

    if nh:
        observed_temps = observed_temps.sel(LATITUDE=slice(0, 90))
        tm_observed = tm_observed.sel(LATITUDE=slice(0, 90))
        hbar_ds = hbar_ds.sel(LATITUDE=slice(0, 90))
    else:
        observed_temps = observed_temps.sel(LATITUDE=slice(-90, 0))
        tm_observed = tm_observed.sel(LATITUDE=slice(-90, 0))
        hbar_ds = hbar_ds.sel(LATITUDE=slice(-90, 0))

    def get_correlation_in_month(month):
        hbar = hbar_ds.sel(MONTH=month)["MONTHLY_MEAN_MLD_PRESSURE"]
        other_hbar = other_hbar_ds.sel(MONTH=month)["MONTHLY_MEAN_MLD"]
        mean_hbar = np.nanmean(hbar.values[np.isfinite(hbar.values)])
        mean_other_hbar = np.nanmean(other_hbar.values[np.isfinite(other_hbar.values)])
        mean_correlations = []
        for depth in depths:
            tsub = observed_temps.sel(PRESSURE=depth)
            correlation = xr.corr(tm_observed.where(tm_observed.MONTH == month), tsub.where(tm_observed.MONTH == month), dim='TIME')
            # correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
            # plt.show()
            mean_correlation = correlation.mean()
            mean_correlations.append(mean_correlation)
        plt.grid()
        plt.plot(depths, mean_correlations)
        plt.axvline(mean_hbar, color='red', label="Mean hbar")
        plt.axvline(mean_other_hbar, color='green', label="Mean `other` hbar")
        plt.xlabel("Depth (dbar)")
        plt.ylabel("Correlation of Temperature at Depth with Tm")
        plt.legend()
        plt.show()

    get_correlation_in_month(1)
    get_correlation_in_month(7)


correlate_tsub(observations_ds, tm_observed, hbar_ds)


