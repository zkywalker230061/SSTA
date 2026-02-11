"""
Simulate Sea Surface Salinity Anomalies (SSSA).

Chengyun Zhu
2026-1-18
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.animation import FuncAnimation

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file

# matplotlib.use('TkAgg')


SURFACE = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = True

RHO_O = 1025  # kg / m^3
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


# def make_movie(dataset, vmin, vmax, colorbar_label=None, ENSO_ds=None):
#     times = dataset.TIME.values

#     fig, ax = plt.subplots()
#     # ax = plt.axes(projection=ccrs.PlateCarree())
#     # ax.coastlines()
#     pcolormesh = ax.pcolormesh(dataset.LONGITUDE.values, dataset.LATITUDE.values,
#                                dataset.isel(TIME=0), cmap='RdBu_r')
#     pcolormesh.set_clim(vmin=vmin, vmax=vmax)
#     title = ax.set_title(f'Time = {times[0]}')

#     cbar = plt.colorbar(pcolormesh, ax=ax, label=colorbar_label)
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')

#     def update(frame):
#         month = int((times[frame] + 0.5) % 12)
#         if month == 0:
#             month = 12
#         year = 2004 + int((times[frame]) / 12)
#         pcolormesh.set_array(dataset.isel(TIME=frame).values.ravel())
#         # pcolormesh.set_clim(vmin=float(model_anomaly_ds.isel(TIME=frame).min()), vmax=float(model_anomaly_ds.isel(TIME=frame).max()))
#         pcolormesh.set_clim(vmin=vmin, vmax=vmax)
#         cbar.update_normal(pcolormesh)
#         if (ENSO_ds is not None):
#             enso_index = ENSO_ds.isel(time=frame).value.values.item()
#             title.set_text(f'Year: {year}; Month: {month}; ENSO index: {round(enso_index, 4)}')
#         else:
#             title.set_text(f'Year: {year}; Month: {month}')
#         return [pcolormesh, title]

#     animation = FuncAnimation(fig, update, frames=len(times), interval=500, blit=False)
#     plt.show()


q_surface_ds = load_and_prepare_dataset(
    "datasets/Simulation-Surface_Water_Rate-(2004-2018).nc"
)
q_surface = (
    q_surface_ds['ANOMALY_avg_ie']
    + q_surface_ds['ANOMALY_avg_tprate']
)
q_surface = q_surface.drop_vars('MONTH')
q_surface.name = 'ANOMALY_SURFACE_WATER_RATE'
if not SURFACE:
    q_surface = q_surface - q_surface

q_ekman = load_and_prepare_dataset(
    "datasets/.test-Simulation-Ekman_Water_Rate-(2004-2018).nc"
)['ANOMALY_EKMAN_WATER_RATE']
q_ekman = q_ekman.where(
    (q_ekman['LATITUDE'] > 5) | (q_ekman['LATITUDE'] < -5), 0
)
if not EKMAN:
    q_ekman = q_ekman - q_ekman

q_geostrophic = load_and_prepare_dataset(
    "datasets/Simulation-Geostrophic_Water_Rate-(2004-2018).nc"
)['ANOMALY_GEOSTROPHIC_WATER_RATE']
q_geostrophic = q_geostrophic.where(
    (q_geostrophic['LATITUDE'] > 5) | (q_geostrophic['LATITUDE'] < -5), 0
)
if not GEOSTROPHIC:
    q_geostrophic = q_geostrophic - q_geostrophic

q_entrainment = load_and_prepare_dataset(
    "datasets/Simulation-Entrainment_Water_Rate-(2004-2018).nc"
)['ANOMALY_ENTRAINMENT_WATER_RATE']
w_e_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc"
)['MONTHLY_MEAN_w_e']
if not ENTRAINMENT:
    q_entrainment = q_entrainment - q_entrainment
    w_e_monthly_mean = w_e_monthly_mean - w_e_monthly_mean

s_m = load_and_prepare_dataset(
    'datasets/.test-ml.nc'
)['MIXED_LAYER_SALINITY']
s_m_monthly_mean = get_monthly_mean(s_m)
s_m_a = get_anomaly(s_m, s_m_monthly_mean)
# s_m_a = load_and_prepare_dataset(
#     "datasets/Mixed_Layer_Salinity_Anomalies-(2004-2018).nc"
# )['ANOMALY_ML_SALINITY']
s_m_a = s_m_a.drop_vars('MONTH')

s_sub_a = load_and_prepare_dataset(
    "datasets/Sub_Layer_Salinity_Anomalies-(2004-2018).nc"
)['ANOMALY_SUB_SALINITY']
s_sub_a = s_sub_a.drop_vars('MONTH')

h_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-Seasonal_Mean.nc"
)['MONTHLY_MEAN_MLD']

h_monthly_mean = xr.concat([h_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
h_monthly_mean['TIME'] = s_m_a.TIME

w_e_monthly_mean = xr.concat([w_e_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
w_e_monthly_mean = w_e_monthly_mean.rename({'MONTH': 'TIME'})
w_e_monthly_mean['TIME'] = s_m_a.TIME

ds_m_a_dt = (
    q_surface
    + q_ekman
    + q_geostrophic
) / (RHO_O * h_monthly_mean)

_lambda = w_e_monthly_mean / h_monthly_mean + 1e-20

s_m_a_simulated_list = []

for month_num in s_m_a['TIME'].values:
    if month_num == 0.5:
        s_m_a_simulated_da = s_m_a.sel(TIME=month_num)
        temp = s_m_a_simulated_da
    else:
        s_m_a_simulated_da = (
            # s_m_a.sel(TIME=month_num-1)
            # + ds_m_a_dt.sel(TIME=month_num-1) * SECONDS_MONTH
            temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
            + (
                (
                    s_sub_a.sel(TIME=month_num-1) * np.log(h_monthly_mean.sel(TIME=month_num)/h_monthly_mean.sel(TIME=month_num-1)) / SECONDS_MONTH
                    + ds_m_a_dt.sel(TIME=month_num-1)
                )
                / _lambda.sel(TIME=month_num-1) * (1 - np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH))
            )
        )
        temp = s_m_a_simulated_da
    s_m_a_simulated_da = s_m_a_simulated_da.expand_dims(TIME=[month_num])
    s_m_a_simulated_list.append(s_m_a_simulated_da)

s_m_a_simulated = xr.concat(
    s_m_a_simulated_list,
    dim="TIME",
    coords="minimal"
)

s_m_a_simulated_monthly_mean = get_monthly_mean(s_m_a_simulated)
s_m_a_simulated = get_anomaly(s_m_a_simulated, s_m_a_simulated_monthly_mean)
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

print("rms simulated", rms_simulated.mean().item())
rms_simulated.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=0.5)
plt.show()

print("rms observed", rms_observed.mean().item())
rms_observed.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=0.5)
plt.show()

rmse = rms_difference / rms_observed
print("normalised rmse", rmse.mean().item())
rmse.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=0, vmax=3)
plt.show()

corr = xr.corr(s_m_a, s_m_a_simulated, dim='TIME')
print("corr", corr.mean().item())
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()


# make_movie(s_m_a_simulated, -0.5, 0.5)
# make_movie(s_m_a, -0.5, 0.5)

surface_fraction = []
entrainment_fraction = []
ekman_fraction = []
geo_fraction = []
surface_contribution = []
entrainment_contribution = []
ekman_contribution = []
geo_contribution = []
total = abs(q_surface) + abs(q_entrainment) + abs(q_ekman) + abs(q_geostrophic)
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
    surface_contribution.append(q_surface.sel(TIME=month_num).mean().item())
    entrainment_contribution.append(q_entrainment.sel(TIME=month_num).mean().item())
    ekman_contribution.append(q_ekman.sel(TIME=month_num).mean().item())
    geo_contribution.append(q_geostrophic.sel(TIME=month_num).mean().item())

plt.plot(s_m_a['TIME'], surface_fraction, label='Surface')
plt.plot(s_m_a['TIME'], entrainment_fraction, label='Entrainment')
plt.plot(s_m_a['TIME'], ekman_fraction, label='Ekman')
plt.plot(s_m_a['TIME'], geo_fraction, label='Geostrophic')
# plt.plot(s_m_a['TIME'], surface_contribution, label='Surface')
# plt.plot(s_m_a['TIME'], entrainment_contribution, label='Entrainment')
# plt.plot(s_m_a['TIME'], ekman_contribution, label='Ekman')
# plt.plot(s_m_a['TIME'], geo_contribution, label='Geostrophic')
plt.legend()
# plt.ylim(0, 1)
plt.show()

t_m_a_simulated = load_and_prepare_dataset(
    "datasets/Simulation-TA.nc"
)['TA_SIMULATED']

corr_ssta_sssa = xr.corr(s_m_a_simulated, t_m_a_simulated, dim='TIME')
print("corr ssta sssa", corr_ssta_sssa.mean().item())
corr_ssta_sssa.plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=-1, vmax=1)
plt.show()
