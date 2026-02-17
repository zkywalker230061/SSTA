import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import get_month_from_time
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
ARGO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
OBSERVATIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Temperature-(2004-2018).nc"
OBSERVATIONS_JJ_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/observed_anomaly_JJ.nc"
H_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/h.nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/hbar.nc"
OTHER_H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/other_h_bar.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Entrainment_Velocity-(2004-2018).nc"
MAX_GRADIENT_T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/Tsub_Max_Gradient_Method_h.nc"
MAX_GRADIENT_ENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/Entrainment_Vel_h.nc"


t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
argo_ds = load_and_prepare_dataset(OBSERVATIONS_DATA_PATH)
# observations_ds["ARGO_OBSERVED"] = observations_ds["ARGO_TEMPERATURE_MEAN"] + observations_ds["ARGO_TEMPERATURE_ANOMALY"]
observations_ds = xr.open_dataset(OBSERVATIONS_DATA_PATH, decode_times=False)
processed_observations_ds = xr.open_dataset(OBSERVATIONS_JJ_DATA_PATH, decode_times=False)
tm_observed = processed_observations_ds["__xarray_dataarray_variable__"]  # bad name...
h_ds = xr.open_dataset(H_DATA_PATH, decode_times=False)
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
other_hbar_ds = xr.open_dataset(OTHER_H_BAR_DATA_PATH, decode_times=False)
ent_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
max_grad_t_sub_ds = xr.open_dataset(MAX_GRADIENT_T_SUB_DATA_PATH, decode_times=False)
max_grad_ent_vel_ds = xr.open_dataset(MAX_GRADIENT_ENT_VEL_DATA_PATH, decode_times=False)



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


def correlate_tsub(observations_ds, tm_observed, hbar_ds, month_to_check, nh=True):
    max_depth = 300

    observed_temps = observations_ds["TEMPERATURE"].sel(PRESSURE=slice(0, max_depth))

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
        hbar = hbar_ds.sel(MONTH=month)["MONTHLY_MEAN_MLD"]
        # hbar.plot(x='LONGITUDE', y='LATITUDE', cmap='Blues', vmin=0, vmax=300)
        # plt.show()
        other_hbar = other_hbar_ds.sel(MONTH=month)["MONTHLY_MEAN_MLD"]
        mean_hbar = np.nanmean(hbar.values[np.isfinite(hbar.values)])
        mean_other_hbar = np.nanmean(other_hbar.values[np.isfinite(other_hbar.values)])
        mean_correlations = []
        mean_temperatures = []
        for depth in depths:
            tsub = observed_temps.sel(PRESSURE=depth)
            mean_temperatures.append(tsub.where(tsub.MONTH == month).mean())
            correlation = xr.corr(tm_observed.where(tm_observed.MONTH == month), tsub.where(tsub.MONTH == month), dim='TIME')
            if depth > 50:
                correlation.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
                plt.show()
            mean_correlation = correlation.mean()
            mean_correlations.append(mean_correlation)

        fig, ax1 = plt.subplots()
        ax1.plot(depths, mean_correlations, label="Mean correlation of Argo Temperature with Tm")
        ax1.axvline(mean_hbar, color='red', label="Mean hbar")
        ax1.axvline(mean_other_hbar, color='green', label="Mean `other` hbar")
        ax1.set_ylabel("Mean correlation of Argo Temperature with Tm")
        ax1.set_ylim([0, 1])
        ax1.grid()
        ax2 = ax1.twinx()
        ax2.plot(depths, mean_temperatures, color="purple", label="Mean Argo Temperature")
        ax2.set_ylabel("Mean Argo Temperature (ºC)")
        ax1.set_xlabel("Depth (dbar)")
        fig.legend()
        fig.tight_layout()
        if nh:
            title = "Northern Hemisphere, month " + str(month)
        else:
            title = "Southern Hemisphere, month " + str(month)
        plt.title(title)
        plt.show()

        # plt.grid()
        # plt.plot(depths, mean_correlations)
        # plt.axvline(mean_hbar, color='red', label="Mean hbar")
        # plt.axvline(mean_other_hbar, color='green', label="Mean `other` hbar")
        # plt.xlabel("Depth (dbar)")
        # plt.ylabel("Correlation of Temperature at Depth with Tm")
        # plt.legend()
        # plt.show()

    get_correlation_in_month(month_to_check)


def compare_tsub():
    # print(ent_vel_ds)
    # print(max_grad_ent_vel_ds)
    # print(max_grad_t_sub_ds)
    # print(argo_ds)
    # print(h_ds)
    mld = h_ds["MLD"]

    SAVED_TSUB_DEPTHS = True
    if not SAVED_TSUB_DEPTHS:
        def get_depth_of_temperature(argo, target, pressure):
            temperatures_below_target = np.where(argo <= target)[0]
            if len(temperatures_below_target) == 0:  # indicates a continent
                return np.nan
            below_target_index = temperatures_below_target[0]
            above_target_index = below_target_index - 1
            return np.interp(target, [argo[below_target_index], argo[above_target_index]], [pressure[below_target_index], pressure[above_target_index]])

        max_grad_depth = xr.apply_ufunc(get_depth_of_temperature, argo_ds['TEMPERATURE'], max_grad_t_sub_ds['SUB_TEMPERATURE'], argo_ds['PRESSURE'], input_core_dims=[['PRESSURE'], [], ['PRESSURE']], vectorize=True)
        t_sub_depth = xr.apply_ufunc(get_depth_of_temperature, argo_ds['TEMPERATURE'], t_sub_ds['T_sub'], argo_ds['PRESSURE'], input_core_dims=[['PRESSURE'], [], ['PRESSURE']], vectorize=True)
        depths_of_tsub = xr.Dataset({'MAX_GRAD_DEPTH': max_grad_depth, 'T_SUB_DEPTH': t_sub_depth})
        depths_of_tsub.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/depths_of_tsub.nc")
    else:
        depths_of_tsub = xr.open_dataset("/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/depths_of_tsub.nc", decode_times=False)

    max_grad_mean_depth = depths_of_tsub["MAX_GRAD_DEPTH"].mean(dim=['LATITUDE', 'LONGITUDE'])
    t_sub_mean_depth = depths_of_tsub["T_SUB_DEPTH"].mean(dim=['LATITUDE', 'LONGITUDE'])
    # print(mld.values)
    mld = mld.where(np.isfinite(mld))
    mld_mean = mld.mean(dim=['LATITUDE', 'LONGITUDE'])
    plt.grid()
    plt.plot((max_grad_mean_depth['TIME'] - 0.5) / 12 + 2004, max_grad_mean_depth, label="Max Gradient Tsub depth")
    plt.plot((t_sub_mean_depth['TIME'] - 0.5) / 12 + 2004, t_sub_mean_depth, label="Regular Tsub depth")
    plt.plot((mld_mean['TIME'] - 0.5) / 12 + 2004, mld_mean, label="Actual MLD")
    plt.xlabel('Time (year)')
    plt.ylabel('Mean Depth of Tsub')
    plt.legend()
    plt.show()


#correlate_tsub(observations_ds, tm_observed, hbar_ds, nh=False, month_to_check=7)
compare_tsub()

