"""
Simulate Sea Surface Temperature Anomalies (SSTA).

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


SURFACE_RAD = True
SURFACE_TURB = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = True
LAMBDA_A = 15
MAX_GRAD_TSUB = True

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


q_surface_ds = load_and_prepare_dataset(
    "datasets/Simulation-Surface_Heat_Flux-(2004-2025).nc"
)
q_surface_rad = (
    q_surface_ds['ANOMALY_avg_snswrf']
    + q_surface_ds['ANOMALY_avg_snlwrf']
)
q_surface_rad = q_surface_rad.drop_vars('MONTH')
q_surface_rad.name = 'ANOMALY_SURFACE_RADIATIVE_HEAT_FLUX'
q_surface_turb = (
    q_surface_ds['ANOMALY_avg_slhtf']
    + q_surface_ds['ANOMALY_avg_ishf']
)
q_surface_turb = q_surface_turb.drop_vars('MONTH')
q_surface_turb.name = 'ANOMALY_SURFACE_TURBULENT_HEAT_FLUX'
if not SURFACE_RAD:
    q_surface_rad = q_surface_rad - q_surface_rad
if not SURFACE_TURB:
    q_surface_turb = q_surface_turb - q_surface_turb

q_ekman = load_and_prepare_dataset(
    "datasets/Simulation-Ekman_Heat_Flux-(2004-2025).nc"
)['ANOMALY_EKMAN_HEAT_FLUX']
q_ekman = q_ekman.where(
    (q_ekman['LATITUDE'] > 5) | (q_ekman['LATITUDE'] < -5), 0
)
if not EKMAN:
    q_ekman = q_ekman - q_ekman

q_geostrophic = load_and_prepare_dataset(
    "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2025).nc"
)['ANOMALY_GEOSTROPHIC_HEAT_FLUX']
q_geostrophic = q_geostrophic.where(
    (q_geostrophic['LATITUDE'] > 5) | (q_geostrophic['LATITUDE'] < -5), 0
)
if not GEOSTROPHIC:
    q_geostrophic = q_geostrophic - q_geostrophic

if MAX_GRAD_TSUB:
    q_entrainment = load_and_prepare_dataset(
        "datasets/Simulation-Entrainment_Heat_Flux_Max_Gradient-(2004-2025).nc"
    )['ANOMALY_ENTRAINMENT_HEAT_FLUX']
else:
    q_entrainment = load_and_prepare_dataset(
        "datasets/Simulation-Entrainment_Heat_Flux-(2004-2025).nc"
    )['ANOMALY_ENTRAINMENT_HEAT_FLUX']
w_e_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Entrainment_Velocity-Clim_Mean.nc"
)['MONTHLY_MEAN_w_e']
if not ENTRAINMENT:
    q_entrainment = q_entrainment - q_entrainment
    w_e_monthly_mean = w_e_monthly_mean - w_e_monthly_mean

t_m_a = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2025).nc"
)['ANOMALY_ML_TEMPERATURE']
t_m_a = t_m_a.drop_vars('MONTH')

t_m_a_reynolds = load_and_prepare_dataset(
    "datasets/reynolds_sst_Anomalies-(2004-2025)_no_2004.nc"
)['ANOMALY_SST']
t_m_a_reynolds = t_m_a_reynolds.drop_vars('MONTH')

if MAX_GRAD_TSUB:
    t_sub_a = load_and_prepare_dataset(
        "datasets/Sub_Layer_Temperature_Max_Gradient_Method_Anomalies-(2004-2025).nc"
    )['ANOMALY_SUB_TEMPERATURE']
else:
    t_sub_a = load_and_prepare_dataset(
        "datasets/Sub_Layer_Temperature_Anomalies-(2004-2025).nc"
    )['ANOMALY_SUB_TEMPERATURE']
t_sub_a = t_sub_a.drop_vars('MONTH')

h_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-Clim_Mean.nc"
)['MONTHLY_MEAN_MLD']

h_monthly_mean = xr.concat([h_monthly_mean] * 22, dim='MONTH').reset_coords(drop=True)
h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
h_monthly_mean['TIME'] = t_m_a.TIME

w_e_monthly_mean = xr.concat([w_e_monthly_mean] * 22, dim='MONTH').reset_coords(drop=True)
w_e_monthly_mean = w_e_monthly_mean.rename({'MONTH': 'TIME'})
w_e_monthly_mean['TIME'] = t_m_a.TIME

dt_m_a_dt = (
    q_surface_rad
    + q_surface_turb
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
            temp * np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH)
            + (
                (
                    (t_sub_a.sel(TIME=month_num-1)
                     * w_e_monthly_mean.sel(TIME=month_num-1)
                     / h_monthly_mean.sel(TIME=month_num-1))
                    + dt_m_a_dt.sel(TIME=month_num-1)
                )
                * (1 - np.exp(-_lambda.sel(TIME=month_num-1) * SECONDS_MONTH))
                / _lambda.sel(TIME=month_num-1)
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

t_m_a_simulated.name = 'TA_SIMULATED'
t_m_a_simulated.attrs['units'] = 'K'

t_m_a_simulated_monthly_mean = get_monthly_mean(
    t_m_a_simulated.where(t_m_a.TIME >= 12.5, drop=True)
)
t_m_a_simulated = get_anomaly(t_m_a_simulated, t_m_a_simulated_monthly_mean)
t_m_a_simulated = t_m_a_simulated.drop_vars('MONTH')

# make_movie(t_m_a_simulated, -2, 2)
# make_movie(t_m_a_reynolds, -2, 2)
# make_movie_2(
#     t_m_a_reynolds, t_m_a_simulated,
#     vmin=-3, vmax=3,
#     cmap='RdBu_r', unit='°C',
#     title=['Observed SSTA', 'Simulated SSTA'],
#     save_path="SSTA-compare.mp4"
# )

# t_m_a_simulated.to_netcdf("datasets/Simulation-TA.nc")

# rms plots
# ----------------------------------------------------------------------------
print("simulated (max, min, mean, abs mean):")
print(f"Max: {t_m_a_simulated.max().item():.2f} °C, Min: {t_m_a_simulated.min().item():.2f} °C")
print(
    f"Mean: {t_m_a_simulated.mean().item():.2f} °C, \
        Abs Mean: {abs(t_m_a_simulated).mean().item():.2f} °C"
)

print("observed (max, min, mean, abs mean):")
print(f"Max: {t_m_a_reynolds.max().item():.2f} °C, Min: {t_m_a_reynolds.min().item():.2f} °C")
print(
    f"Mean: {t_m_a_reynolds.mean().item():.2f} °C, \
        Abs Mean: {abs(t_m_a_reynolds).mean().item():.2f} °C"
)
print('-----')

rms_difference = np.sqrt(((t_m_a_reynolds - t_m_a_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((t_m_a_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((t_m_a_reynolds ** 2).mean(dim=['TIME']))
rmse = rms_difference / rms_observed

fig, axes = plt.subplots(
    nrows=1, ncols=2,
    figsize=(15, 10), dpi=600,
    subplot_kw={'projection': ccrs.PlateCarree()}
)

axes[0].pcolormesh(
    rms_simulated['LONGITUDE'], rms_simulated['LATITUDE'], rms_simulated,
    cmap='nipy_spectral',
    vmin=-0, vmax=3
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
    f"RMS Simulated: Spatial Mean {rms_simulated.mean().item():.2f} °C", fontsize=20, loc='left'
)

axes[1].pcolormesh(
    rms_observed['LONGITUDE'], rms_observed['LATITUDE'], rms_observed,
    cmap='nipy_spectral',
    vmin=-0, vmax=3
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
    f"RMS Observed: Spatial Mean {rms_observed.mean().item():.2f} °C", fontsize=20, loc='left'
)

fig.colorbar(
    axes[0].collections[0], ax=axes.ravel().tolist(), label="°C",
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
corr = xr.corr(t_m_a_reynolds, t_m_a_simulated, dim='TIME')

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
wind = True
surface_rad_fraction = []
surface_turb_fraction = []
entrainment_fraction = []
ekman_fraction = []
geo_fraction = []
wind_fraction = []
if wind:
    total = (
        abs(q_surface_rad)
        + abs(q_entrainment)
        + abs(q_geostrophic)
        + abs(q_surface_turb + q_ekman)
    )
else:
    total = (
        abs(q_surface_rad)
        + abs(q_surface_turb)
        + abs(q_entrainment)
        + abs(q_ekman)
        + abs(q_geostrophic)
    )

weights = np.cos(np.deg2rad(t_m_a['LATITUDE']))
basins = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
mask = basins.mask(t_m_a['LONGITUDE'], t_m_a['LATITUDE'])
mask_numer = 2

for month_num in t_m_a['TIME'].values:
    surface_rad_fraction.append(
        (
            abs(q_surface_rad.sel(TIME=month_num)).where(mask == mask_numer)
            / total.sel(TIME=month_num).where(mask == mask_numer)
        ).weighted(weights).mean().item()
    )
    surface_turb_fraction.append(
        (
            abs(q_surface_turb.sel(TIME=month_num)).where(mask == mask_numer)
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
    wind_fraction.append(
        (
            abs(q_surface_turb.sel(TIME=month_num) +
                q_ekman.sel(TIME=month_num)).where(mask == mask_numer)
            / total.sel(TIME=month_num).where(mask == mask_numer)
        ).weighted(weights).mean().item()
    )


plt.figure(figsize=(5, 5), dpi=600)
plt.plot(
    t_m_a['TIME'], surface_rad_fraction,
    label='Radiation', color='#d8031c', alpha=0.8
)
plt.plot(
    t_m_a['TIME'], entrainment_fraction,
    label='Entrainment', color='#66CCFF', alpha=0.8
)
plt.plot(
    t_m_a['TIME'], geo_fraction,
    label='Geostrophic', color='#ffb703', alpha=0.8
)
if wind:
    plt.plot(
        t_m_a['TIME'], wind_fraction,
        label='Wind (Turbulence + Ekman)', color='#006400', alpha=0.8
    )
else:
    plt.plot(
        t_m_a['TIME'], surface_turb_fraction,
        label='Turbulence', color='#006400', alpha=0.8
    )
    plt.plot(
        t_m_a['TIME'], ekman_fraction,
        label='Ekman', color='#39C5BB', alpha=0.8
    )
plt.xlim(0.5, 264.5)
plt.xticks(np.arange(0.5, 264.5, 36), np.arange(2004, 2026, 3))
plt.xlabel('Year', loc='right')
plt.ylim(0, 0.5)
plt.ylabel('Fractional Contribution')
plt.ylim(0)
plt.legend(frameon=False, ncols=(3-int(wind)), fontsize=8)
plt.title(r'$\gamma$'+f' = {LAMBDA_A}, Spatially Weighted', fontsize=15, loc='left')
plt.show()

surface_rad_fraction_monthly_mean = [np.mean(surface_rad_fraction[i::12]) for i in range(12)]
surface_turb_fraction_monthly_mean = [np.mean(surface_turb_fraction[i::12]) for i in range(12)]
entrainment_fraction_monthly_mean = [np.mean(entrainment_fraction[i::12]) for i in range(12)]
ekman_fraction_monthly_mean = [np.mean(ekman_fraction[i::12]) for i in range(12)]
geo_fraction_monthly_mean = [np.mean(geo_fraction[i::12]) for i in range(12)]
wind_fraction_monthly_mean = [np.mean(wind_fraction[i::12]) for i in range(12)]

plt.figure(figsize=(5, 5), dpi=600)
months = np.arange(1, 13)
plt.plot(
    months, surface_rad_fraction_monthly_mean,
    label='Radiation', color='#d8031c', alpha=0.8
)
plt.plot(
    months, entrainment_fraction_monthly_mean,
    label='Entrainment', color='#66CCFF', alpha=0.8
)
plt.plot(
    months, geo_fraction_monthly_mean,
    label='Geostrophic', color='#ffb703', alpha=0.8
)
if wind:
    plt.plot(
        months, wind_fraction_monthly_mean,
        label='Wind (Turbulence + Ekman)', color='#006400', alpha=0.8
    )
else:
    plt.plot(
        months, surface_turb_fraction_monthly_mean,
        label='Turbulence', color='#006400', alpha=0.8
    )
    plt.plot(
        months, ekman_fraction_monthly_mean,
        label='Ekman', color='#39C5BB', alpha=0.8
    )
plt.xlabel('Month', loc='right')
plt.ylim(0, 0.7)
plt.ylabel('Fractional Contribution')
plt.legend(frameon=False, ncols=(3-int(wind)), fontsize=8)
plt.title('North Atlantic', fontsize=15, loc='left')
plt.show()


# # spatial mean plot
# t_m_a_simulated = t_m_a_simulated.where(
#     (t_m_a_simulated['LATITUDE'] > 15) | (t_m_a_simulated['LATITUDE'] < -15), 0
# )
# t_m_a_simulated_spatial_mean = t_m_a_simulated.mean(dim=['LONGITUDE', 'LATITUDE'])
# t_m_a_reynolds = t_m_a_reynolds.where(
#     (t_m_a_reynolds['LATITUDE'] > 15) | (t_m_a_reynolds['LATITUDE'] < -15), 0
# )
# t_m_a_reynolds_spatial_mean = t_m_a_reynolds.mean(dim=['LONGITUDE', 'LATITUDE'])
# plt.plot(t_m_a_simulated_spatial_mean['TIME'], t_m_a_simulated_spatial_mean, label='Simulated')
# plt.plot(t_m_a_reynolds_spatial_mean['TIME'], t_m_a_reynolds_spatial_mean, label='Observed')
# plt.legend()
# plt.show()

# # QQ plot
# # t_m_a_simulated = t_m_a_simulated.where(
# #     (t_m_a_simulated['LATITUDE'] > 20) & (t_m_a_simulated['LATITUDE'] < 60), 0
# # )
# # t_m_a_simulated = t_m_a_simulated.where(
# #     (t_m_a_simulated['LONGITUDE'] > -100) & (t_m_a_simulated['LONGITUDE'] < 0), 0
# # )
# # t_m_a_reynolds = t_m_a_reynolds.where(
# #     (t_m_a_reynolds['LATITUDE'] > 20) & (t_m_a_reynolds['LATITUDE'] < 60), 0
# # )
# # t_m_a_reynolds = t_m_a_reynolds.where(
# #     (t_m_a_reynolds['LONGITUDE'] > -100) & (t_m_a_reynolds['LONGITUDE'] < 0), 0
# # )

# # t_m_a_simulated = t_m_a_simulated.where(
# #     ((t_m_a_simulated['LATITUDE'] < -20) & (t_m_a_simulated['LATITUDE'] > -60)), 0
# # )
# # t_m_a_simulated = t_m_a_simulated.where(
# #     ((t_m_a_simulated['LONGITUDE'] > -180) & (t_m_a_simulated['LONGITUDE'] < -55)), 0
# # )
# # t_m_a_reynolds = t_m_a_reynolds.where(
# #     ((t_m_a_reynolds['LATITUDE'] < -20) & (t_m_a_reynolds['LATITUDE'] > -60)), 0
# # )
# # t_m_a_reynolds = t_m_a_reynolds.where(
# #     ((t_m_a_reynolds['LONGITUDE'] > -180) & (t_m_a_reynolds['LONGITUDE'] < -55)), 0
# # )

# t_m_a_simulated.sel(TIME=0.5).plot()
# plt.figure(figsize=(6, 6))
# for lon, lat in zip(t_m_a_simulated['LONGITUDE'], t_m_a_simulated['LATITUDE']):
#     plt.plot(
#         t_m_a_reynolds.sel(LONGITUDE=lon, LATITUDE=lat).values,
#         t_m_a_simulated.sel(LONGITUDE=lon, LATITUDE=lat).values,
#         ','
#     )
# x = np.linspace(-5, 5, 100)
# plt.plot(x, x, 'r--')
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
# # plt.yscale('log', base=2)
# # plt.xscale('log', base=2)
# plt.show()

# autocorrelation plot
# ----------------------------------------------------------------------------
reynolds_raw = load_and_prepare_dataset(
    "datasets/Reynolds/sst.mon.anom.2004-2025.nc"
)['anom']

t_m_a_simulated = t_m_a_simulated.where(
    (t_m_a_simulated['LATITUDE'] > 15) | (t_m_a_simulated['LATITUDE'] < -15), 0
)
t_m_a_reynolds = t_m_a_reynolds.where(
    (t_m_a_reynolds['LATITUDE'] > 15) | (t_m_a_reynolds['LATITUDE'] < -15), 0
)
reynolds_raw = reynolds_raw.where(
    (reynolds_raw['LATITUDE'] > 15) | (reynolds_raw['LATITUDE'] < -15), 0
)

basins = regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50
mask = basins.mask(t_m_a['LONGITUDE'], t_m_a['LATITUDE'])
mask_numer = 2
test_region = t_m_a_simulated.where(mask == mask_numer)

autocorr_points_simulated = []
autocorr_points_observed = []
autocorr_points_observed_raw = []
# for lon, lat in zip(test_region['LONGITUDE'], test_region['LATITUDE']):
for lon, lat in zip(t_m_a_simulated['LONGITUDE'], t_m_a_simulated['LATITUDE']):
    autocorr_point_simulated = acf(t_m_a_simulated.sel(LONGITUDE=lon, LATITUDE=lat), nlags=25)
    autocorr_point_observed = acf(t_m_a_reynolds.sel(LONGITUDE=lon, LATITUDE=lat), nlags=25)
    autocorr_point_observed_raw = acf(reynolds_raw.sel(LONGITUDE=lon, LATITUDE=lat), nlags=25)
    if not np.isnan(autocorr_point_simulated).all():
        autocorr_points_simulated.append(autocorr_point_simulated)
    if not np.isnan(autocorr_point_observed).all():
        autocorr_points_observed.append(autocorr_point_observed)
    if not np.isnan(autocorr_point_observed_raw).all():
        autocorr_points_observed_raw.append(autocorr_point_observed_raw)
autocorr_points_simulated = np.array(autocorr_points_simulated)
autocorr_points_observed = np.array(autocorr_points_observed)
autocorr_points_observed_raw = np.array(autocorr_points_observed_raw)

plt.figure(figsize=(5, 5), dpi=600)
plt.plot(
    autocorr_points_simulated.mean(axis=0),
    label='Simulated (2005-2025 LTM)',
    color='#EE0000', alpha=0.8
)
plt.plot(
    autocorr_points_observed.mean(axis=0),
    label='Observed - Reynolds SSTA (2005-2025 LTM)',
    color='#66CCFF', alpha=0.8
)
plt.plot(
    autocorr_points_observed_raw.mean(axis=0),
    label='Observed - Reynolds SSTA (1971-2020 LTM)',
    color='#66CCFF', alpha=0.5, linestyle='--'
)

plt.xlim(0, 25)
plt.xlabel('lag (months)', loc='right')
plt.ylim(-0.2, 1)

plt.legend(frameon=False)
plt.title(r'$\gamma$'+f' = {LAMBDA_A}, 15°N-15°S Excluded', fontsize=15, loc='left')

plt.show()
