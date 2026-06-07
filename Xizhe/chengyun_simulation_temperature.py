"""
Simulate Sea Surface Temperature Anomalies (SSTA).

Chengyun Zhu
2026-1-18
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.animation import FuncAnimation

from utils_read_nc import load_and_prepare_dataset
from utils_read_nc import get_monthly_mean, get_anomaly
# from utilities import save_file

# matplotlib.use('TkAgg')

SURFACE = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = True
LAMBDA_A = 15

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
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
if not SURFACE:
    q_surface = q_surface - q_surface

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
        t_m_a_simulated_da = t_m_a.sel(TIME=month_num)
        temp = t_m_a_simulated_da
    else:
        t_m_a_simulated_da = (
            # t_m_a.sel(TIME=month_num-1)
            # + dt_m_a_dt.sel(TIME=month_num-1) * SECONDS_MONTH
            temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
            + (
                (
                    t_sub_a.sel(TIME=month_num-1) * np.log(h_monthly_mean.sel(TIME=month_num)/h_monthly_mean.sel(TIME=month_num-1)) / SECONDS_MONTH
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

surface_fraction = []
entrainment_fraction = []
ekman_fraction = []
geo_fraction = []
surface_contribution = []
entrainment_contribution = []
ekman_contribution = []
geo_contribution = []
total = abs(q_surface) + abs(q_entrainment) + abs(q_ekman) + abs(q_geostrophic)
for month_num in t_m_a['TIME'].values:
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

plt.plot(t_m_a['TIME'], surface_fraction, label='Surface')
plt.plot(t_m_a['TIME'], entrainment_fraction, label='Entrainment')
plt.plot(t_m_a['TIME'], ekman_fraction, label='Ekman')
plt.plot(t_m_a['TIME'], geo_fraction, label='Geostrophic')
# plt.plot(t_m_a['TIME'], surface_contribution, label='Surface')
# plt.plot(t_m_a['TIME'], entrainment_contribution, label='Entrainment')
# plt.plot(t_m_a['TIME'], ekman_contribution, label='Ekman')
# plt.plot(t_m_a['TIME'], geo_contribution, label='Geostrophic')
plt.legend()
# plt.ylim(0, 1)
plt.show()

# t_m_a_simulated.name = 'TA_SIMULATED'
# t_m_a_simulated.attrs['units'] = 'K'
# t_m_a_simulated.to_netcdf("datasets/Simulation-TA.nc")
