"""
Simulate Sea Surface Salinity Anomalies (SSSA).

Chengyun Zhu
2026-3-4
"""

import xarray as xr
import numpy as np
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
import regionmask

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file
# from utilities_plot import make_movie_2, make_movie


SURFACE = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = True
GAMMA = 0.004

RHO_O = 1025  # kg / m^3
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month

q_surface_ds = load_and_prepare_dataset(
    "datasets/Simulation-Surface_Water_Rate-(2004-2025).nc"
)  # negative for evaporation, positive for precipitation

s_m_monthly_mean = load_and_prepare_dataset(
    'datasets/Mixed_Layer_Salinity-Clim_Mean.nc'
)['MONTHLY_MEAN_ML_SALINITY']

s_m_monthly_mean = xr.concat([s_m_monthly_mean] * 22, dim='MONTH').reset_coords(drop=True)
s_m_monthly_mean = s_m_monthly_mean.rename({'MONTH': 'TIME'})
s_m_monthly_mean['TIME'] = q_surface_ds.TIME

q_surface = (
    - q_surface_ds['ANOMALY_avg_ie']  # negative sign to retrive E'
    - q_surface_ds['ANOMALY_avg_tprate']  # negative to subtract P'
) * s_m_monthly_mean  # (E' - P') * S_bar

q_surface = q_surface.drop_vars('MONTH')
q_surface.name = 'ANOMALY_SURFACE_WATER_RATE'
if not SURFACE:
    q_surface = q_surface - q_surface

q_ekman = load_and_prepare_dataset(
    "datasets/Simulation-Ekman_Water_Rate-(2004-2025).nc"
)['ANOMALY_EKMAN_WATER_RATE']
q_ekman = q_ekman.where(
    (q_ekman['LATITUDE'] > 5) | (q_ekman['LATITUDE'] < -5), 0
)
if not EKMAN:
    q_ekman = q_ekman - q_ekman

q_geostrophic = load_and_prepare_dataset(
    "datasets/Simulation-Geostrophic_Water_Rate-(2004-2025).nc"
)['ANOMALY_GEOSTROPHIC_WATER_RATE']
q_geostrophic = q_geostrophic.where(
    (q_geostrophic['LATITUDE'] > 5) | (q_geostrophic['LATITUDE'] < -5), 0
)
if not GEOSTROPHIC:
    q_geostrophic = q_geostrophic - q_geostrophic

q_entrainment = load_and_prepare_dataset(
    "datasets/Simulation-Entrainment_Water_Rate-(2004-2025).nc"
)['ANOMALY_ENTRAINMENT_WATER_RATE']
w_e_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Entrainment_Velocity-Clim_Mean.nc"
)['MONTHLY_MEAN_w_e']
if not ENTRAINMENT:
    q_entrainment = q_entrainment - q_entrainment
    w_e_monthly_mean = w_e_monthly_mean - w_e_monthly_mean

s_m_a = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Salinity_Anomalies-(2004-2025).nc"
)['ANOMALY_ML_SALINITY']
s_m_a = s_m_a.drop_vars('MONTH')

s_sub_a = load_and_prepare_dataset(
    "datasets/Sub_Layer_Salinity_Anomalies-(2004-2025).nc"
)['ANOMALY_SUB_SALINITY']
s_sub_a = s_sub_a.drop_vars('MONTH')

h_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-Clim_Mean.nc"
)['MONTHLY_MEAN_MLD']

h_monthly_mean = xr.concat([h_monthly_mean] * 22, dim='MONTH').reset_coords(drop=True)
h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
h_monthly_mean['TIME'] = s_m_a.TIME

w_e_monthly_mean = xr.concat([w_e_monthly_mean] * 22, dim='MONTH').reset_coords(drop=True)
w_e_monthly_mean = w_e_monthly_mean.rename({'MONTH': 'TIME'})
w_e_monthly_mean['TIME'] = s_m_a.TIME

ds_m_a_dt = (
    q_surface
    + q_ekman
    + q_geostrophic
) / (RHO_O * h_monthly_mean)

_lambda = w_e_monthly_mean / h_monthly_mean + GAMMA / (RHO_O * h_monthly_mean)

integrate_factor = xr.where(
    _lambda == 0,
    SECONDS_MONTH,
    (1 - np.exp(-_lambda * SECONDS_MONTH)) / _lambda
)

s_m_a_simulated_list = []

for month_num in s_m_a['TIME'].values:
    if month_num == 0.5:
        s_m_a_simulated_da = s_m_a.sel(TIME=month_num)
        temp = s_m_a_simulated_da
    else:
        s_m_a_simulated_da = (
            temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
            + (
                (
                    (s_sub_a.sel(TIME=month_num-1)
                     * w_e_monthly_mean.sel(TIME=month_num-1)
                     / h_monthly_mean.sel(TIME=month_num-1))
                    + ds_m_a_dt.sel(TIME=month_num-1)
                ) * integrate_factor.sel(TIME=month_num-1)
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

s_m_a_simulated.name = 'SA_SIMULATED'
s_m_a_simulated.attrs['units'] = 'PSU'

s_m_a_simulated_monthly_mean = get_monthly_mean(
    s_m_a_simulated.where(s_m_a_simulated.TIME >= 12.5, drop=True)
)
s_m_a_simulated = get_anomaly(s_m_a_simulated, s_m_a_simulated_monthly_mean)
s_m_a_simulated = s_m_a_simulated.drop_vars('MONTH')

# make_movie(s_m_a_simulated, -0.5, 0.5)
# make_movie(s_m_a, -0.5, 0.5)
# make_movie_2(
#     s_m_a, s_m_a_simulated,
#     vmin=-0.5, vmax=0.5,
#     cmap='RdBu_r', unit='PSU',
#     title=['Observed SSSA', 'Simulated SSSA'],
#     save_path="SSSA-compare.mp4"
# )

# s_m_a_simulated.to_netcdf("datasets/Simulation-SA.nc")

# rms plots
# ----------------------------------------------------------------------------
print("simulated (max, min, mean, abs mean):")
print(f"Max: {s_m_a_simulated.max().item():.2f} PSU, Min: {s_m_a_simulated.min().item():.2f} PSU")
print(
    f"Mean: {s_m_a_simulated.mean().item():.2f} PSU, \
        Abs Mean: {abs(s_m_a_simulated).mean().item():.2f} PSU"
)

print("observed (max, min, mean, abs mean):")
print(f"Max: {s_m_a.max().item():.2f} PSU, Min: {s_m_a.min().item():.2f} PSU")
print(
    f"Mean: {s_m_a.mean().item():.2f} PSU, \
        Abs Mean: {abs(s_m_a).mean().item():.2f} PSU"
)
print('-----')

rms_difference = np.sqrt(((s_m_a - s_m_a_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((s_m_a_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((s_m_a ** 2).mean(dim=['TIME']))
rmse = rms_difference / rms_observed

fig, axes = plt.subplots(
    nrows=1, ncols=2,
    figsize=(15, 10), dpi=600,
    subplot_kw={'projection': ccrs.PlateCarree()}
)

axes[0].pcolormesh(
    rms_simulated['LONGITUDE'], rms_simulated['LATITUDE'], rms_simulated,
    cmap='nipy_spectral',
    vmin=-0, vmax=0.5
)
axes[0].coastlines()

axes[0].set_xlim(-180, 180)
axes[0].set_ylim(-90, 90)

gl = axes[0].gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=2, color='gray', alpha=0.5, linestyle='--'
)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.xlines = False
gl.ylines = False
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.ylabel_style = {'size': 15, 'color': 'gray'}
gl.xlabel_style = {'size': 15, 'color': 'gray'}

axes[0].set_title(
    f"RMS Simulated: Spatial Mean {rms_simulated.mean().item():.2f} PSU", fontsize=20, loc='left'
)

axes[1].pcolormesh(
    rms_observed['LONGITUDE'], rms_observed['LATITUDE'], rms_observed,
    cmap='nipy_spectral',
    vmin=-0, vmax=0.5
)
axes[1].coastlines()

axes[1].set_xlim(-180, 180)
axes[1].set_ylim(-90, 90)

gl = axes[1].gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=2, color='gray', alpha=0.5, linestyle='--'
)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.xlines = False
gl.ylines = False
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.ylabel_style = {'size': 15, 'color': 'gray'}
gl.xlabel_style = {'size': 15, 'color': 'gray'}
axes[1].set_title(
    f"RMS Observed: Spatial Mean {rms_observed.mean().item():.2f} PSU", fontsize=20, loc='left'
)

fig.colorbar(
    axes[0].collections[0], ax=axes.ravel().tolist(), label="PSU",
    orientation='horizontal', aspect=40, pad=0.05
)

plt.show()

plt.figure(figsize=(10, 5), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    rmse['LONGITUDE'], rmse['LATITUDE'], rmse,
    cmap='nipy_spectral',
    vmin=-0, vmax=3
)
ax.coastlines()

plt.xlim(-180, 180)
plt.ylim(-90, 90)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=2, color='gray', alpha=0.5, linestyle='--'
)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.xlines = False
gl.ylines = False
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.ylabel_style = {'size': 15, 'color': 'gray'}
gl.xlabel_style = {'size': 15, 'color': 'gray'}

plt.title(f"NRMSE: Spatial Mean {rmse.mean().item():.2f}", fontsize=20, loc='left')

plt.colorbar(
    shrink=0.75, orientation='horizontal', aspect=40, pad=0.1
)

plt.show()

# corr plot
# ----------------------------------------------------------------------------
corr = xr.corr(s_m_a, s_m_a_simulated, dim='TIME')

plt.figure(figsize=(10, 5), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    corr['LONGITUDE'], corr['LATITUDE'], corr,
    cmap='nipy_spectral',
    vmin=-1, vmax=1
)
ax.coastlines()

plt.xlim(-180, 180)
plt.ylim(-90, 90)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=2, color='gray', alpha=0.5, linestyle='--'
)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.xlines = False
gl.ylines = False
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.ylabel_style = {'size': 15, 'color': 'gray'}
gl.xlabel_style = {'size': 15, 'color': 'gray'}

plt.title(f"Correlation: Spatial Mean {corr.mean().item():.2f}", fontsize=20, loc='left')

plt.colorbar(
    shrink=0.75, orientation='horizontal', aspect=40, pad=0.1
)

plt.show()

# fraction plot
# ----------------------------------------------------------------------------
surface_fraction = []
entrainment_fraction = []
ekman_fraction = []
geo_fraction = []
wind_fraction = []

total = (
    abs(q_surface)
    + abs(q_entrainment)
    + abs(q_ekman)
    + abs(q_geostrophic)
)

weights = np.cos(np.deg2rad(s_m_a['LATITUDE']))
basins = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
mask = basins.mask(s_m_a['LONGITUDE'], s_m_a['LATITUDE'])
mask_numer = 2

for month_num in s_m_a['TIME'].values:
    surface_fraction.append(
        (
            abs(q_surface.sel(TIME=month_num)).where(mask == mask_numer)
            / total.sel(TIME=month_num).where(mask == mask_numer)
        ).weighted(weights).mean().item()
    )
    entrainment_fraction.append(
        (
            abs(q_entrainment.sel(TIME=month_num)).where(mask == mask_numer)
            / total.sel(TIME=month_num).where(mask == mask_numer)
        ).weighted(weights).mean().item()
    )
    ekman_fraction.append(
        (
            abs(q_ekman.sel(TIME=month_num)).where(mask == mask_numer)
            / total.sel(TIME=month_num).where(mask == mask_numer)
        ).weighted(weights).mean().item()
    )
    geo_fraction.append(
        (
            abs(q_geostrophic.sel(TIME=month_num)).where(mask == mask_numer)
            / total.sel(TIME=month_num).where(mask == mask_numer)
        ).weighted(weights).mean().item()
    )


plt.figure(figsize=(5, 5), dpi=600)
plt.plot(
    s_m_a['TIME'], surface_fraction,
    label='Surface', color='#d8031c', alpha=0.8
)
plt.plot(
    s_m_a['TIME'], entrainment_fraction,
    label='Entrainment', color='#66CCFF', alpha=0.8
)
plt.plot(
    s_m_a['TIME'], ekman_fraction,
    label='Ekman', color='#39C5BB', alpha=0.8
)
plt.plot(
    s_m_a['TIME'], geo_fraction,
    label='Geostrophic', color='#ffb703', alpha=0.8
)
plt.xlim(0.5, 264.5)
plt.xticks(np.arange(0.5, 264.5, 36), np.arange(2004, 2026, 3))
plt.xlabel('Year', loc='right')
plt.ylim(0, 0.5)
plt.ylabel('Fractional Contribution')
plt.ylim(0)
plt.legend(frameon=False, ncols=2, fontsize=8)
plt.title(r'$\gamma = 0.004$, Spatially Weighted', fontsize=15, loc='left')
plt.show()

surface_fraction_monthly_mean = [np.mean(surface_fraction[i::12]) for i in range(12)]
entrainment_fraction_monthly_mean = [np.mean(entrainment_fraction[i::12]) for i in range(12)]
ekman_fraction_monthly_mean = [np.mean(ekman_fraction[i::12]) for i in range(12)]
geo_fraction_monthly_mean = [np.mean(geo_fraction[i::12]) for i in range(12)]
wind_fraction_monthly_mean = [np.mean(wind_fraction[i::12]) for i in range(12)]

plt.figure(figsize=(5, 5), dpi=600)
months = np.arange(1, 13)
plt.plot(
    months, surface_fraction_monthly_mean,
    label='Surface', color='#d8031c', alpha=0.8
)
plt.plot(
    months, entrainment_fraction_monthly_mean,
    label='Entrainment', color='#66CCFF', alpha=0.8
)
plt.plot(
    months, ekman_fraction_monthly_mean,
    label='Ekman', color='#39C5BB', alpha=0.8
)
plt.plot(
    months, geo_fraction_monthly_mean,
    label='Geostrophic', color='#ffb703', alpha=0.8
)
plt.xlabel('Month', loc='right')
plt.ylim(0, 0.5)
plt.ylabel('Fractional Contribution')
plt.legend(frameon=False, ncols=2, fontsize=8)
plt.title('North Atlantic', fontsize=15, loc='left')
plt.show()

# # spatial mean plot
# s_m_a_simulated = s_m_a_simulated.where(
#     (s_m_a_simulated['LATITUDE'] > 20) | (s_m_a_simulated['LATITUDE'] < -20), 0
# )
# s_m_a_simulated_spatial_mean = s_m_a_simulated.mean(dim=['LONGITUDE', 'LATITUDE'])
# s_m_a = s_m_a.where(
#     (s_m_a['LATITUDE'] > 20) | (s_m_a['LATITUDE'] < -20), 0
# )
# s_m_a_spatial_mean = s_m_a.mean(dim=['LONGITUDE', 'LATITUDE'])
# plt.plot(s_m_a_simulated_spatial_mean['TIME'], s_m_a_simulated_spatial_mean, label='Simulated')
# plt.plot(s_m_a_spatial_mean['TIME'], s_m_a_spatial_mean, label='Observed')
# plt.legend()
# plt.show()

# # QQ plot
# # s_m_a_simulated = s_m_a_simulated.where(
# #     (s_m_a_simulated['LATITUDE'] > 20) & (s_m_a_simulated['LATITUDE'] < 70), 0
# # )
# # s_m_a_simulated = s_m_a_simulated.where(
# #     (s_m_a_simulated['LONGITUDE'] > -100) & (s_m_a_simulated['LONGITUDE'] < 0), 0
# # )
# # s_m_a = s_m_a.where(
# #     (s_m_a['LATITUDE'] > 20) & (s_m_a['LATITUDE'] < 70), 0
# # )
# # s_m_a = s_m_a.where(
# #     (s_m_a['LONGITUDE'] > -100) & (s_m_a['LONGITUDE'] < 0), 0
# # )

# # s_m_a_simulated = s_m_a_simulated.where(
# #     ((s_m_a_simulated['LATITUDE'] < -20) & (s_m_a_simulated['LATITUDE'] > -60)), 0
# # )
# # s_m_a_simulated = s_m_a_simulated.where(
# #     ((s_m_a_simulated['LONGITUDE'] > -180) & (s_m_a_simulated['LONGITUDE'] < -55)), 0
# # )
# # s_m_a = s_m_a.where(
# #     ((s_m_a['LATITUDE'] < -20) & (s_m_a['LATITUDE'] > -60)), 0
# # )
# # s_m_a = s_m_a.where(
# #     ((s_m_a['LONGITUDE'] > -180) & (s_m_a['LONGITUDE'] < -55)), 0
# # )

# # s_m_a_simulated.sel(TIME=0.5).plot()
# plt.figure(figsize=(6, 6))
# for lon, lat in zip(s_m_a_simulated['LONGITUDE'], s_m_a_simulated['LATITUDE']):
#     plt.plot(
#         s_m_a.sel(LONGITUDE=lon, LATITUDE=lat).values,
#         s_m_a_simulated.sel(LONGITUDE=lon, LATITUDE=lat).values,
#         ','
#     )
# x = np.linspace(-1, 1, 100)
# plt.plot(x, x, 'r--')
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# # plt.yscale('log', base=2)
# # plt.xscale('log', base=2)
# plt.show()

# autocorrelation plot
# ----------------------------------------------------------------------------
s_m_a_simulated = s_m_a_simulated.where(
    (s_m_a_simulated['LATITUDE'] > 15) | (s_m_a_simulated['LATITUDE'] < -15), 0
)
s_m_a_ = s_m_a.where(
    (s_m_a['LATITUDE'] > 15) | (s_m_a['LATITUDE'] < -15), 0
)

basins = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
mask = basins.mask(s_m_a['LONGITUDE'], s_m_a['LATITUDE'])
mask_numer = 2
test_region = s_m_a_simulated.where(mask == mask_numer)

autocorr_points_simulated = []
autocorr_points_observed = []
# for lon, lat in zip(test_region['LONGITUDE'], test_region['LATITUDE']):
for lon, lat in zip(s_m_a_simulated['LONGITUDE'], s_m_a_simulated['LATITUDE']):
    autocorr_point_simulated = acf(s_m_a_simulated.sel(LONGITUDE=lon, LATITUDE=lat), nlags=25)
    autocorr_point_observed = acf(s_m_a_.sel(LONGITUDE=lon, LATITUDE=lat), nlags=25)
    if not np.isnan(autocorr_point_simulated).all():
        autocorr_points_simulated.append(autocorr_point_simulated)
    if not np.isnan(autocorr_point_observed).all():
        autocorr_points_observed.append(autocorr_point_observed)
autocorr_points_simulated = np.array(autocorr_points_simulated)
autocorr_points_observed = np.array(autocorr_points_observed)

plt.figure(figsize=(5, 5), dpi=600)
plt.plot(
    autocorr_points_simulated.mean(axis=0),
    label='Simulated (2005-2025 LTM)',
    color='#EE0000', alpha=0.8
)
plt.plot(
    autocorr_points_observed.mean(axis=0),
    label='Observed (2005-2025 LTM)',
    color='#66CCFF', alpha=0.8
)

plt.xlim(0, 25)
plt.xlabel('lag (months)', loc='right')
plt.ylim(-0.2, 1)

plt.legend(frameon=False)
plt.title(r'$\gamma$'+f' = {GAMMA}, 15°N-15°S Excluded', fontsize=15, loc='left')

plt.show()


# something todo
# precipitation_a = -q_surface_ds['ANOMALY_avg_tprate'].where(
#     ~np.isnan(s_m_monthly_mean)
# )
# evaporation_a = -q_surface_ds['ANOMALY_avg_ie'].where(
#     ~np.isnan(s_m_monthly_mean)
# )
# plt.plot(precipitation_a.mean(dim=['LONGITUDE', 'LATITUDE']), label='Precipitation Anomaly')
# plt.plot(evaporation_a.mean(dim=['LONGITUDE', 'LATITUDE']), label='Evaporation Anomaly')
# plt.legend()
# plt.show()

# plt.plot(
#     evaporation_a.mean(dim=['LONGITUDE', 'LATITUDE']) + precipitation_a.mean(dim=['LONGITUDE', 'LATITUDE']),
#     label="E'-P'"
# )
# plt.legend()
# plt.show()

# # for each month, plot the range of SSSA: max-min
# s_m_a_simulated_range = s_m_a_simulated.max(dim=['LONGITUDE', 'LATITUDE']) - s_m_a_simulated.min(dim=['LONGITUDE', 'LATITUDE'])
# s_m_a_range = s_m_a.max(dim=['LONGITUDE', 'LATITUDE']) - s_m_a.min(dim=['LONGITUDE', 'LATITUDE'])
# plt.plot(s_m_a_simulated_range, label='Simulated SSSA Range')
# plt.plot(s_m_a_range, label='Observed SSSA Range')
# plt.legend()
# plt.show()
