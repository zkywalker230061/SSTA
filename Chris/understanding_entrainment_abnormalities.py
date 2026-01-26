import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, coriolis_parameter
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib

INCLUDE_SURFACE = False
INCLUDE_EKMAN = False
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC = False
INCLUDE_GEOSTROPHIC_DISPLACEMENT = False
USE_DOWNLOADED_SSH = False
gamma_0 = 30.0

save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC, USE_DOWNLOADED_SSH=USE_DOWNLOADED_SSH, gamma0=gamma_0, INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_DISPLACEMENT)
ALL_SCHEMES_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/all_anomalies/" + save_name + ".nc"
MLD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
OBSERVATIONS_JJ_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/observed_anomaly_JJ.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Entrainment_Velocity-(2004-2018).nc"

all_models_ds = xr.open_dataset(ALL_SCHEMES_DATA_PATH, decode_times=False)
mld_ds = xr.open_dataset(MLD_DATA_PATH, decode_times=False)
tsub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
obs_ds = xr.open_dataset(OBSERVATIONS_JJ_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)

hbar_da = hbar_ds["MONTHLY_MEAN_MLD_PRESSURE"]
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

tm = obs_ds['__xarray_dataarray_variable__']
tm_monthly_mean = get_monthly_mean(tm)
obs_ds = get_anomaly(obs_ds, '__xarray_dataarray_variable__', tm_monthly_mean)
tm_anomaly = obs_ds['__xarray_dataarray_variable___ANOMALY']

tsub_anomaly = tsub_ds['T_sub_ANOMALY']

implicit_model = all_models_ds["IMPLICIT"]

# make_movie(tm_anomaly, -3, 3, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/obs_tm_anom.mp4")
# make_movie(tsub_anomaly, -3, 3, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/obs_tsub_anom.mp4")
# make_movie(implicit_model, -3, 3, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/entrainment_only_model.mp4")

# (xr.corr(tm_anomaly, tsub_anomaly, dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# plt.title("Correlation between Tsub and Tm observation")
# plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/tm_tsub_correlation.jpg")
# plt.show()
#
# (xr.corr(tm_anomaly, implicit_model, dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# plt.title("Correlation between entrainment-only model and Tm observation")
# plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/entrainment_only_model_obs_correlation.jpg")
# plt.show()
#
# (xr.corr(tsub_anomaly, implicit_model, dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# plt.title("Correlation between entrainment-only model and Tsub observation")
# plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/entrainment_only_model_tsub_correlation.jpg")
# plt.show()

# split into entraining and detraining periods
def get_nan_mask(month, da):
    da = da.sel(MONTH=month)
    entrain_mask = da > 0
    detrain_mask = da == 0        # == no entrainment occurring
    return [entrain_mask, detrain_mask]

monthly_entrainment_masks = []
monthly_detrainment_masks = []
for month in range(1, 13):
    masks = get_nan_mask(month, entrainment_vel_da)
    monthly_entrainment_masks.append(masks[0])
    monthly_detrainment_masks.append(masks[1])
monthly_entrainment_masks_ds = xr.concat(monthly_entrainment_masks, 'MONTH')
monthly_detrainment_masks_ds = xr.concat(monthly_detrainment_masks, 'MONTH')

# monthly_entrainment_masks_ds.sel(MONTH=11).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# plt.show()

tm_anomaly_entrainment = []
tm_anomaly_detrainment = []
tsub_anomaly_entrainment = []
tsub_anomaly_detrainment = []
implicit_model_entrainment = []
implicit_model_detrainment = []
for time in tm_anomaly.TIME.values:
    month = (time + 0.5) % 12
    if month == 0:
        month = 12.0
    tm_anomaly_entrainment.append(tm_anomaly.sel(TIME=time).where(monthly_entrainment_masks_ds.sel(MONTH=month)))
    tm_anomaly_detrainment.append(tm_anomaly.sel(TIME=time).where(monthly_detrainment_masks_ds.sel(MONTH=month)))
    tsub_anomaly_entrainment.append(tsub_anomaly.sel(TIME=time).where(monthly_entrainment_masks_ds.sel(MONTH=month)))
    tsub_anomaly_detrainment.append(tsub_anomaly.sel(TIME=time).where(monthly_detrainment_masks_ds.sel(MONTH=month)))
    implicit_model_entrainment.append(implicit_model.sel(TIME=time).where(monthly_entrainment_masks_ds.sel(MONTH=month)))
    implicit_model_detrainment.append(implicit_model.sel(TIME=time).where(monthly_detrainment_masks_ds.sel(MONTH=month)))

tm_anomaly_entrainment_ds = xr.concat(tm_anomaly_entrainment, 'TIME')
tm_anomaly_entrainment_ds = tm_anomaly_entrainment_ds.drop_vars('MONTH')

tm_anomaly_detrainment_ds = xr.concat(tm_anomaly_detrainment, 'TIME')
tm_anomaly_detrainment_ds = tm_anomaly_detrainment_ds.drop_vars('MONTH')

tsub_anomaly_entrainment_ds = xr.concat(tsub_anomaly_entrainment, 'TIME')
tsub_anomaly_entrainment_ds = tsub_anomaly_entrainment_ds.drop_vars('MONTH')

tsub_anomaly_detrainment_ds = xr.concat(tsub_anomaly_detrainment, 'TIME')
tsub_anomaly_detrainment_ds = tsub_anomaly_detrainment_ds.drop_vars('MONTH')

implicit_model_entrainment_ds = xr.concat(implicit_model_entrainment, 'TIME')
implicit_model_entrainment_ds = implicit_model_entrainment_ds.drop_vars('MONTH')

implicit_model_detrainment_ds = xr.concat(implicit_model_detrainment, 'TIME')
implicit_model_detrainment_ds = implicit_model_detrainment_ds.drop_vars('MONTH')

# implicit_model_detrainment_ds.sel(TIME=11.5).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# plt.show()

# (xr.corr(tm_anomaly_entrainment_ds, implicit_model_entrainment_ds, dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# plt.title("Correlation between entrainment-only model and Tm observation in entraining regions")
# # plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/entrainment_only_model_obs_correlation.jpg")
# plt.show()
#
# (xr.corr(tm_anomaly_detrainment_ds, implicit_model_detrainment_ds, dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# plt.title("Correlation between entrainment-only model and Tm observation in detraining regions")
# # plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/entrainment_only_model_obs_correlation.jpg")
# plt.show()


def take_from_same_month(month, da):
    da_at_month = []
    for time in da.TIME.values:
        month_of_this_time = (time + 0.5) % 12
        if month_of_this_time == 0:
            month_of_this_time = 12.0
        if month_of_this_time == month:
            da_at_month.append(da.sel(TIME=time))
    da_at_month = xr.concat(da_at_month, 'TIME')
    return da_at_month

for month in range(1, 13):
    plt.figure()
    (xr.corr(take_from_same_month(month, tm_anomaly_entrainment_ds), take_from_same_month(month, implicit_model_entrainment_ds), dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
    plt.title("Correlation between entrainment-only model and Tm observation in entraining regions")
    plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/entrainment_only_model_obs_correlation_entrainingregions_month" + str(month) + ".jpg")
    #plt.show()

    plt.figure()
    (xr.corr(take_from_same_month(month, tm_anomaly_detrainment_ds), take_from_same_month(month, implicit_model_detrainment_ds), dim='TIME')).plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
    plt.title("Correlation between entrainment-only model and Tm observation in detraining regions")
    plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment/entrainment_only_model_obs_correlation_detrainingregions_month" + str(month) + ".jpg")
    #plt.show()



