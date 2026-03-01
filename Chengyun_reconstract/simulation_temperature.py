"""
Simulate Sea Surface Temperature Anomalies (SSTA).

Chengyun Zhu
2026-1-18
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file
# from utilities_plot import make_movie_2, make_movie

# matplotlib.use('TkAgg')

SURFACE = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = True
LAMBDA_A = 15

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


q_surface_ds = load_and_prepare_dataset(
    "datasets/Simulation-Surface_Heat_Flux-(2004-2018).nc"
)
q_surface = (
    q_surface_ds['ANOMALY_avg_slhtf']
    + q_surface_ds['ANOMALY_avg_ishf']
    + q_surface_ds['ANOMALY_avg_snswrf']
    + q_surface_ds['ANOMALY_avg_snlwrf']
)
q_surface = q_surface.drop_vars('MONTH')
q_surface.name = 'ANOMALY_SURFACE_HEAT_FLUX'
q_surface_rad = (
    q_surface_ds['ANOMALY_avg_snswrf']
    + q_surface_ds['ANOMALY_avg_snlwrf']
)
q_surface_rad.drop_vars('MONTH')
q_surface_turb = (
    q_surface_ds['ANOMALY_avg_slhtf']
    + q_surface_ds['ANOMALY_avg_ishf']
)
q_surface_turb.drop_vars('MONTH')
if not SURFACE:
    q_surface = q_surface - q_surface
    q_surface_rad = q_surface_rad - q_surface_rad
    q_surface_turb = q_surface_turb - q_surface_turb

q_ekman = load_and_prepare_dataset(
    "datasets/.test-Simulation-Ekman_Heat_Flux-(2004-2018).nc"
)['ANOMALY_EKMAN_HEAT_FLUX']
q_ekman = q_ekman.where(
    (q_ekman['LATITUDE'] > 5) | (q_ekman['LATITUDE'] < -5), 0
)
if not EKMAN:
    q_ekman = q_ekman - q_ekman

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

q_entrainment = load_and_prepare_dataset(
    "datasets/Simulation-Entrainment_Heat_Flux-(2004-2018).nc"
)['ANOMALY_ENTRAINMENT_HEAT_FLUX']
w_e_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc"
)['MONTHLY_MEAN_w_e']
if not ENTRAINMENT:
    q_entrainment = q_entrainment - q_entrainment
    w_e_monthly_mean = w_e_monthly_mean - w_e_monthly_mean

t_m = load_and_prepare_dataset(
    "datasets/.test-ml.nc"
)['MIXED_LAYER_TEMP']
t_m_monthly_mean = get_monthly_mean(t_m)
t_m_a = get_anomaly(t_m, t_m_monthly_mean)
# t_m_a = load_and_prepare_dataset(
#     "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc"
# )['ANOMALY_ML_TEMPERATURE']
t_m_a = t_m_a.drop_vars('MONTH')

t_m_a_reynolds = load_and_prepare_dataset(
    "datasets/Reynolds/sst_anomalies-(2004-2018).nc"
)['anom']

t_sub_a = load_and_prepare_dataset(
    "datasets/Sub_Layer_Temperature_Anomalies-(2004-2018).nc"
)['ANOMALY_SUB_TEMPERATURE']
t_sub_a = t_sub_a.drop_vars('MONTH')

h_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-Seasonal_Mean.nc"
)['MONTHLY_MEAN_MLD']

h_monthly_mean = xr.concat([h_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
h_monthly_mean['TIME'] = t_m_a.TIME

w_e_monthly_mean = xr.concat([w_e_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
w_e_monthly_mean = w_e_monthly_mean.rename({'MONTH': 'TIME'})
w_e_monthly_mean['TIME'] = t_m_a.TIME

dt_m_a_dt = (
    q_surface
    + q_ekman
    + q_geostrophic
) / (RHO_O * C_O * h_monthly_mean)

_lambda = LAMBDA_A / (RHO_O * C_O * h_monthly_mean) + w_e_monthly_mean / h_monthly_mean

t_m_a_simulated_list = []

for month_num in t_m_a['TIME'].values:
    if month_num == 0.5:
        t_m_a_simulated_da = t_m_a_reynolds.sel(TIME=month_num)
        temp = t_m_a_simulated_da
    else:
        t_m_a_simulated_da = (
            # t_m_a.sel(TIME=month_num-1)
            # + dt_m_a_dt.sel(TIME=month_num-1) * SECONDS_MONTH
            temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
            + (
                (
                    # t_sub_a.sel(TIME=month_num-1) * np.log(h_monthly_mean.sel(TIME=month_num)/h_monthly_mean.sel(TIME=month_num-1)) / SECONDS_MONTH
                    t_sub_a.sel(TIME=month_num-1) * w_e_monthly_mean.sel(TIME=month_num) / h_monthly_mean.sel(TIME=month_num)
                    + dt_m_a_dt.sel(TIME=month_num-1)
                )
                / _lambda.sel(TIME=month_num-1) * (1 - np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH))
            )
        )
        temp = t_m_a_simulated_da
    t_m_a_simulated_da = t_m_a_simulated_da.expand_dims(TIME=[month_num])
    t_m_a_simulated_list.append(t_m_a_simulated_da)

t_m_a_simulated = xr.concat(
    t_m_a_simulated_list,
    dim="TIME",
    coords="minimal"
)

t_m_a_simulated_monthly_mean = get_monthly_mean(t_m_a_simulated)
t_m_a_simulated = get_anomaly(t_m_a_simulated, t_m_a_simulated_monthly_mean)
t_m_a_simulated = t_m_a_simulated.drop_vars('MONTH')

# make_movie(t_m_a_simulated, -2, 2)
# make_movie(t_m_a_reynolds, -2, 2)
# make_movie_2(
#     t_m_a_reynolds, t_m_a_simulated,
#     vmin=-3, vmax=3,
#     save_path="compare_Liu.mp4"
# )

# t_m_a_simulated.name = 'TA_SIMULATED'
# t_m_a_simulated.attrs['units'] = 'K'
# t_m_a_simulated.to_netcdf("datasets/Simulation-TA.nc")

# ----------------------------------------------------------------------------
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
rmse = rms_difference / rms_observed

normalised_simulated = t_m_a_simulated / rms_simulated
normalised_observed = t_m_a_reynolds / rms_observed
normalised_rms_difference = np.sqrt(
    ((normalised_observed - normalised_simulated) ** 2).mean(dim=['TIME'])
)

print("rms simulated", rms_difference.mean().item())
rms_simulated.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()
print("rms observed", rms_observed.mean().item())
rms_observed.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()
print("normalised rmse", rmse.mean().item())
rmse.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()

print("normalised rmse", normalised_rms_difference.mean().item())
normalised_rms_difference.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()


# corr plot
corr = xr.corr(t_m_a_reynolds, t_m_a_simulated, dim='TIME')
print("corr", corr.mean().item())
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()

# fraction plot
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

# spatial mean plot
t_m_a_simulated = t_m_a_simulated.where(
    (t_m_a_simulated['LATITUDE'] > 20) | (t_m_a_simulated['LATITUDE'] < -20), 0
)
t_m_a_simulated_spatial_mean = t_m_a_simulated.mean(dim=['LONGITUDE', 'LATITUDE'])
t_m_a_reynolds = t_m_a_reynolds.where(
    (t_m_a_reynolds['LATITUDE'] > 20) | (t_m_a_reynolds['LATITUDE'] < -20), 0
)
t_m_a_reynolds_spatial_mean = t_m_a_reynolds.mean(dim=['LONGITUDE', 'LATITUDE'])
plt.plot(t_m_a_simulated_spatial_mean['TIME'], t_m_a_simulated_spatial_mean, label='Simulated')
plt.plot(t_m_a_reynolds_spatial_mean['TIME'], t_m_a_reynolds_spatial_mean, label='Observed')
plt.legend()
plt.show()

# QQ plot
# t_m_a_simulated = t_m_a_simulated.where(
#     (t_m_a_simulated['LATITUDE'] > 20) & (t_m_a_simulated['LATITUDE'] < 60), 0
# )
# t_m_a_simulated = t_m_a_simulated.where(
#     (t_m_a_simulated['LONGITUDE'] > -100) & (t_m_a_simulated['LONGITUDE'] < 0), 0
# )
# t_m_a_reynolds = t_m_a_reynolds.where(
#     (t_m_a_reynolds['LATITUDE'] > 20) & (t_m_a_reynolds['LATITUDE'] < 60), 0
# )
# t_m_a_reynolds = t_m_a_reynolds.where(
#     (t_m_a_reynolds['LONGITUDE'] > -100) & (t_m_a_reynolds['LONGITUDE'] < 0), 0
# )

# t_m_a_simulated = t_m_a_simulated.where(
#     ((t_m_a_simulated['LATITUDE'] < -20) & (t_m_a_simulated['LATITUDE'] > -60)), 0
# )
# t_m_a_simulated = t_m_a_simulated.where(
#     ((t_m_a_simulated['LONGITUDE'] > -180) & (t_m_a_simulated['LONGITUDE'] < -55)), 0
# )
# t_m_a_reynolds = t_m_a_reynolds.where(
#     ((t_m_a_reynolds['LATITUDE'] < -20) & (t_m_a_reynolds['LATITUDE'] > -60)), 0
# )
# t_m_a_reynolds = t_m_a_reynolds.where(
#     ((t_m_a_reynolds['LONGITUDE'] > -180) & (t_m_a_reynolds['LONGITUDE'] < -55)), 0
# )

t_m_a_simulated.sel(TIME=0.5).plot()
plt.figure(figsize=(6, 6))
for lon, lat in zip(t_m_a_simulated['LONGITUDE'], t_m_a_simulated['LATITUDE']):
    plt.plot(
        t_m_a_reynolds.sel(LONGITUDE=lon, LATITUDE=lat).values,
        t_m_a_simulated.sel(LONGITUDE=lon, LATITUDE=lat).values,
        ','
    )
x = np.linspace(-5, 5, 100)
plt.plot(x, x, 'r--')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
# plt.yscale('log', base=2)
# plt.xscale('log', base=2)
plt.show()
