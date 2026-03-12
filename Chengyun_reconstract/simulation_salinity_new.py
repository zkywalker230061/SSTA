"""
Simulate Sea Surface Salinity (SSS) then Sea Surface Salinity Anomalies (SSSA).

Chengyun Zhu
2026-3-12
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file
from utilities_plot import make_movie_2  # , make_movie

# matplotlib.use('TkAgg')

SURFACE = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = False
LAMBDA_A = 0.00

RHO_O = 1025  # kg / m^3
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


q_surface_ds = load_and_prepare_dataset(
    "datasets/Surface_Water_Rate-(2004-2018).nc"
)
s_m = load_and_prepare_dataset(
    'datasets/.test-ml.nc'
)['MIXED_LAYER_SALINITY']
q_surface = (
    - q_surface_ds['avg_ie']
    - q_surface_ds['avg_tprate']
) * s_m
if not SURFACE:
    q_surface = q_surface - q_surface

q_ekman = load_and_prepare_dataset(
    "datasets/Simulation-Ekman_Water_Rate_new-(2004-2018).nc"
)['EKMAN_WATER_RATE']
q_ekman = q_ekman.where(
    (q_ekman['LATITUDE'] > 5) | (q_ekman['LATITUDE'] < -5), 0
)
if not EKMAN:
    q_ekman = q_ekman - q_ekman

# ----------------------------------------------------------------------------
q_geostrophic = load_and_prepare_dataset(
    "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2018).nc"
)['ANOMALY_GEOSTROPHIC_HEAT_FLUX']
# q_geostrophic = load_and_prepare_dataset(
#     "datasets/geostrophic_anomaly_calculated.nc"
# )['GEOSTROPHIC_ANOMALY']
q_geostrophic = q_geostrophic.where(
    (q_geostrophic['LATITUDE'] > 5) | (q_geostrophic['LATITUDE'] < -5), 0
)
if not GEOSTROPHIC:
    q_geostrophic = q_geostrophic - q_geostrophic
# ----------------------------------------------------------------------------

s_m = load_and_prepare_dataset(
    "datasets/.test-ml.nc"
)['MIXED_LAYER_SALINITY']
s_m_monthly_mean = get_monthly_mean(s_m)
s_m_a = get_anomaly(s_m, s_m_monthly_mean)
# t_m_a = load_and_prepare_dataset(
#     "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc"
# )['ANOMALY_ML_TEMPERATURE']
s_m_a = s_m_a.drop_vars('MONTH')

s_sub = load_and_prepare_dataset(
    "datasets/Sub_Layer_Salinity-(2004-2018).nc"
)['SUB_SALINITY']

h = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-(2004-2018).nc"
)['MLD']

w_e = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Entrainment_Velocity-(2004-2018).nc"
)['w_e']
q_entrainment = RHO_O * w_e * (s_sub - s_m)
if not ENTRAINMENT:
    q_entrainment = q_entrainment - q_entrainment
    w_e = w_e - w_e

ds_m_dt = (
    q_surface
    + q_ekman
    + q_geostrophic
) / (RHO_O * h)

_lambda = LAMBDA_A / (RHO_O * h) + w_e / h

integrate_factor = xr.where(
    _lambda == 0,
    SECONDS_MONTH,
    (1 - np.exp(-_lambda * SECONDS_MONTH)) / _lambda
)

s_m_simulated_list = []

for month_num in s_m_a['TIME'].values:
    if month_num == 0.5:
        s_m_simulated_da = s_m.sel(TIME=month_num)
        temp = s_m_simulated_da
    else:
        s_m_simulated_da = (
            temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
            + (
                (
                    s_sub.sel(TIME=month_num-1) * w_e.sel(TIME=month_num) / h.sel(TIME=month_num)
                    + ds_m_dt.sel(TIME=month_num-1)
                ) * integrate_factor.sel(TIME=month_num-1)
            )
        )
        temp = s_m_simulated_da
    s_m_simulated_da = s_m_simulated_da.expand_dims(TIME=[month_num])
    s_m_simulated_list.append(s_m_simulated_da)

s_m_simulated = xr.concat(
    s_m_simulated_list,
    dim="TIME",
    coords="minimal"
)
print(s_m_simulated)

s_m_simulated_monthly_mean = get_monthly_mean(s_m_simulated)
s_m_a_simulated = get_anomaly(s_m_simulated, s_m_simulated_monthly_mean)
s_m_a_simulated = s_m_a_simulated.drop_vars('MONTH')

print("simulated (max, min, mean, abs mean):")
print(s_m_a_simulated.max().item(), s_m_a_simulated.min().item())
print(s_m_a_simulated.mean().item())
print(abs(s_m_a_simulated).mean().item())

print("observed (max, min, mean, abs mean):")
print(s_m_a.max().item(), s_m_a.min().item())
print(s_m_a.mean().item())
print(abs(s_m_a).mean().item())
print('-----')

rms_difference = np.sqrt(((s_m_a - s_m_a_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((s_m_a_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((s_m_a ** 2).mean(dim=['TIME']))

print("rms simulated", rms_difference.mean().item())
rms_simulated.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()
print("rms observed", rms_observed.mean().item())
rms_observed.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()
rmse = rms_difference / rms_observed
print("normalised rmse", rmse.mean().item())
rmse.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()

corr = xr.corr(s_m_a, s_m_a_simulated, dim='TIME')
print("corr", corr.mean().item())
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()

# make_movie(s_m_a_simulated, -2, 2)
# make_movie(s_m_a_reynolds, -2, 2)
make_movie_2(
    s_m_a, s_m_a_simulated,
    vmin=-3, vmax=3,
    # save_path="compare_Liu.mp4"
)

print("simulated (max, min, mean, abs mean):")
print(s_m_simulated.max().item(), s_m_simulated.min().item())
print(s_m_simulated.mean().item())
print(abs(s_m_simulated).mean().item())

print("observed (max, min, mean, abs mean):")
print(s_m.max().item(), s_m.min().item())
print(s_m.mean().item())
print(abs(s_m).mean().item())
print('-----')

rms_difference = np.sqrt(((s_m - s_m_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((s_m_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((s_m ** 2).mean(dim=['TIME']))

print("rms simulated", rms_difference.mean().item())
rms_simulated.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=30, vmax=40)
plt.show()

print("rms observed", rms_observed.mean().item())
rms_observed.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=30, vmax=40)
plt.show()

rmse = rms_difference / rms_observed
print("normalised rmse", rmse.mean().item())
rmse.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()

corr = xr.corr(s_m, s_m_simulated, dim='TIME')
print("corr", corr.mean().item())
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()

# make_movie(s_m_a_simulated, -2, 2)
# make_movie(s_m_a_reynolds, -2, 2)
make_movie_2(
    s_m, s_m_simulated,
    vmin=30, vmax=40,
    # save_path="compare_Liu.mp4"
)

# s_m_a_simulated.name = 'SA_SIMULATED'
# s_m_a_simulated.attrs['units'] = 'PSU'
# s_m_a_simulated.to_netcdf("datasets/Simulation-SA.nc")

surface_fraction = []
entrainment_fraction = []
ekman_fraction = []
geo_fraction = []
total = (
    abs(q_surface)
    + abs(q_entrainment)
    + abs(q_ekman)
    + abs(q_geostrophic)
)
for month_num in s_m_a['TIME'].values:
    surface_fraction.append(
        (abs(q_surface.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    entrainment_fraction.append(
        (abs(q_entrainment.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    ekman_fraction.append(
        (abs(q_ekman.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    geo_fraction.append(
        (abs(q_geostrophic.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )

plt.plot(s_m_a['TIME'], surface_fraction, label='Surface')
plt.plot(s_m_a['TIME'], entrainment_fraction, label='Entrainment')
plt.plot(s_m_a['TIME'], ekman_fraction, label='Ekman')
plt.plot(s_m_a['TIME'], geo_fraction, label='Geostrophic')
plt.legend()
# plt.ylim(0, 1)
plt.show()
