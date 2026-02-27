"""
Simulate Sea Surface Temperature (SST) then Sea Surface Temperature Anomalies (SSTA).

Chengyun Zhu
2026-2-23
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
LAMBDA_A = 15

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


q_surface_ds = load_and_prepare_dataset(
    "datasets/Surface_Heat_Flux-(2004-2018).nc"
)
q_surface = (
    q_surface_ds['avg_slhtf']
    + q_surface_ds['avg_ishf']
    + q_surface_ds['avg_snswrf']
    + q_surface_ds['avg_snlwrf']
)
if not SURFACE:
    q_surface = q_surface - q_surface

q_ekman = load_and_prepare_dataset(
    "datasets/Simulation-Ekman_Heat_Flux_new-(2004-2018).nc"
)['EKMAN_HEAT_FLUX']
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

t_m = load_and_prepare_dataset(
    "datasets/.test-ml.nc"
)['MIXED_LAYER_TEMP']
t_m_monthly_mean = get_monthly_mean(t_m)
t_m_a = get_anomaly(t_m, t_m_monthly_mean)
# t_m_a = load_and_prepare_dataset(
#     "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc"
# )['ANOMALY_ML_TEMPERATURE']
t_m_a = t_m_a.drop_vars('MONTH')

t_m_monthly_mean_reynolds = load_and_prepare_dataset(
    "datasets/Reynolds/sst_ltm.nc"
)['sst']
t_m_a_reynolds = load_and_prepare_dataset(
    "datasets/Reynolds/sst_anomalies-(2004-2018).nc"
)['anom']
t_m_monthly_mean_reynolds = xr.concat(
    [t_m_monthly_mean_reynolds] * 15, dim='MONTH'
).reset_coords(drop=True)
t_m_monthly_mean_reynolds = t_m_monthly_mean_reynolds.rename({'MONTH': 'TIME'})
t_m_monthly_mean_reynolds['TIME'] = t_m_a.TIME
t_m_reynolds = t_m_a_reynolds + t_m_monthly_mean_reynolds

t_sub = load_and_prepare_dataset(
    "datasets/Sub_Layer_Temperature-(2004-2018).nc"
)['SUB_TEMPERATURE']

h = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-(2004-2018).nc"
)['MLD']

w_e = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Entrainment_Velocity-(2004-2018).nc"
)['w_e']
q_entrainment = RHO_O * C_O * w_e * (t_sub - t_m)
if not ENTRAINMENT:
    q_entrainment = q_entrainment - q_entrainment
    w_e = w_e - w_e

dt_m_a_dt = (
    q_surface
    + q_ekman
    + q_geostrophic
) / (RHO_O * C_O * h)

_lambda = LAMBDA_A / (RHO_O * C_O * h) + w_e / h

t_m_simulated_list = []

for month_num in t_m_a['TIME'].values:
    if month_num == 0.5:
        t_m_simulated_da = t_m_reynolds.sel(TIME=month_num)
        temp = t_m_simulated_da
    else:
        t_m_simulated_da = (
            temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
            + (
                (
                    t_sub.sel(TIME=month_num-1) * w_e.sel(TIME=month_num) / h.sel(TIME=month_num)
                    + dt_m_a_dt.sel(TIME=month_num-1)
                )
                / _lambda.sel(TIME=month_num-1) * (1 - np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH))
            )
        )
        temp = t_m_simulated_da
    t_m_simulated_da = t_m_simulated_da.expand_dims(TIME=[month_num])
    t_m_simulated_list.append(t_m_simulated_da)

t_m_simulated = xr.concat(
    t_m_simulated_list,
    dim="TIME",
    coords="minimal"
)
print(t_m_simulated)

t_m_simulated_monthly_mean = get_monthly_mean(t_m_simulated)
t_m_a_simulated = get_anomaly(t_m_simulated, t_m_simulated_monthly_mean)
t_m_a_simulated = t_m_a_simulated.drop_vars('MONTH')

print("simulated (max, min, mean, abs mean):")
print(t_m_a_simulated.max().item(), t_m_a_simulated.min().item())
print(t_m_a_simulated.mean().item())
print(abs(t_m_a_simulated).mean().item())

print("observed (max, min, mean, abs mean):")
print(t_m_a_reynolds.max().item(), t_m_a_reynolds.min().item())
print(t_m_a_reynolds.mean().item())
print(abs(t_m_a_reynolds).mean().item())
print('-----')

rms_difference = np.sqrt(((t_m_a_reynolds - t_m_a_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((t_m_a_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((t_m_a_reynolds ** 2).mean(dim=['TIME']))

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

corr = xr.corr(t_m_a_reynolds, t_m_a_simulated, dim='TIME')
print("corr", corr.mean().item())
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()

# make_movie(t_m_a_simulated, -2, 2)
# make_movie(t_m_a_reynolds, -2, 2)
make_movie_2(
    t_m_a_reynolds, t_m_a_simulated,
    vmin=-3, vmax=3,
    # save_path="compare_Liu.mp4"
)

print("simulated (max, min, mean, abs mean):")
print(t_m_simulated.max().item(), t_m_simulated.min().item())
print(t_m_simulated.mean().item())
print(abs(t_m_simulated).mean().item())

print("observed (max, min, mean, abs mean):")
print(t_m_reynolds.max().item(), t_m_reynolds.min().item())
print(t_m_reynolds.mean().item())
print(abs(t_m_reynolds).mean().item())
print('-----')

rms_difference = np.sqrt(((t_m_a_reynolds - t_m_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((t_m_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((t_m_reynolds ** 2).mean(dim=['TIME']))

print("rms simulated", rms_difference.mean().item())
rms_simulated.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=20)
plt.show()

print("rms observed", rms_observed.mean().item())
rms_observed.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=20)
plt.show()

rmse = rms_difference / rms_observed
print("normalised rmse", rmse.mean().item())
rmse.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()

corr = xr.corr(t_m_reynolds, t_m_simulated, dim='TIME')
print("corr", corr.mean().item())
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()

# make_movie(t_m_a_simulated, -2, 2)
# make_movie(t_m_a_reynolds, -2, 2)
make_movie_2(
    t_m_reynolds, t_m_simulated,
    vmin=-20, vmax=20,
    # save_path="compare_Liu.mp4"
)

# t_m_a_simulated.name = 'TA_SIMULATED'
# t_m_a_simulated.attrs['units'] = 'K'
# t_m_a_simulated.to_netcdf("datasets/Simulation-TA.nc")

q_surface_rad = (
    q_surface_ds['avg_snswrf']
    + q_surface_ds['avg_snlwrf']
)
q_surface_turb = (
    q_surface_ds['avg_slhtf']
    + q_surface_ds['avg_ishf']
)
surface_rad_fraction = []
surface_turb_fraction = []
entrainment_fraction = []
ekman_fraction = []
geo_fraction = []
total = (
    abs(q_surface_rad)
    + abs(q_surface_turb)
    + abs(q_entrainment)
    + abs(q_ekman)
    + abs(q_geostrophic)
)
for month_num in t_m_a['TIME'].values:
    surface_rad_fraction.append(
        (abs(q_surface_rad.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
    )
    surface_turb_fraction.append(
        (abs(q_surface_turb.sel(TIME=month_num)) / total.sel(TIME=month_num)).mean().item()
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

plt.plot(t_m_a['TIME'], surface_rad_fraction, label='Surface Radiative')
plt.plot(t_m_a['TIME'], surface_turb_fraction, label='Surface Turbulent')
plt.plot(t_m_a['TIME'], entrainment_fraction, label='Entrainment')
plt.plot(t_m_a['TIME'], ekman_fraction, label='Ekman')
plt.plot(t_m_a['TIME'], geo_fraction, label='Geostrophic')
plt.legend()
# plt.ylim(0, 1)
plt.show()
