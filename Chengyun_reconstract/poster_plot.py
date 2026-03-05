# %%
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from utilities import load_and_prepare_dataset
# from utilities import get_monthly_mean

font = {'size': 30}
matplotlib.rc('font', **font)

# %%
h = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-(2004-2018).nc"
)['MLD']
plt.figure(figsize=(25, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    h['LONGITUDE'], h['LATITUDE'], h.sel(TIME=0.5),
    cmap='Blues',
    # levels=200,
    vmin=0,
    vmax=500
)
ax.coastlines()

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
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

plt.colorbar(label=h.attrs.get('units'))
plt.show()

# %%
ssh = load_and_prepare_dataset(
    "datasets/Sea_Surface_Height-(2004-2018).nc"
)['ssh']
plt.figure(figsize=(25, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    ssh['LONGITUDE'], ssh['LATITUDE'], ssh.sel(TIME=0.5),
    cmap='RdBu_r',
    # levels=200,
    vmin=-3,
    vmax=3
)
ax.coastlines()

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
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

plt.colorbar(label=ssh.attrs.get('units'))
plt.show()

# %%
simulation_full = load_and_prepare_dataset(
    "datasets/Simulation-Implicit/1111_geostrophiccurrent_ekmanmeanadv_maxgrad_gamma15.0.nc"
)["IMPLICIT"]
# simulation_without_surface = load_and_prepare_dataset(
#     "datasets/Simulation-Implicit/0111_geostrophiccurrent_ekmanmeanadv_maxgrad_gamma15.0.nc"
# )["IMPLICIT"]
# simulation_without_entrainment = load_and_prepare_dataset(
#     "datasets/Simulation-Implicit/1101_geostrophiccurrent_ekmanmeanadv_maxgrad_gamma15.0.nc"
# )["IMPLICIT"]
# simulation_without_ekman = load_and_prepare_dataset(
#     "datasets/Simulation-Implicit/1011_geostrophiccurrent_maxgrad_gamma15.0.nc"
# )["IMPLICIT"]
# simulation_without_geo = load_and_prepare_dataset(
#     "datasets/Simulation-Implicit/1110_ekmanmeanadv_maxgrad_gamma15.0.nc"
# )["IMPLICIT"]

# corr_surface = xr.corr(simulation_full, simulation_without_surface, dim='TIME')
# corr_ent = xr.corr(simulation_full, simulation_without_entrainment, dim='TIME')
# corr_ekman = xr.corr(simulation_full, simulation_without_ekman, dim='TIME')
# corr_geo = xr.corr(simulation_full, simulation_without_geo, dim='TIME')

# # plt.figure(figsize=(20, 10))
# fig, axes = plt.subplots(
#     nrows=2, ncols=2,
#     figsize=(20, 10),
#     subplot_kw={'projection': ccrs.PlateCarree()}
# )
# plt.subplots_adjust(wspace=0.1, hspace=0)

# ax1 = axes[0, 0]
# plot = ax1.pcolormesh(
#     corr_surface['LONGITUDE'], corr_surface['LATITUDE'], corr_surface,
#     cmap='nipy_spectral',
#     # levels=200,
#     vmin=0,
#     vmax=1,
# )
# ax1.coastlines()

# ax1.set_xlim(-180, 180)
# ax1.set_ylim(-90, 90)
# ax1.set_title('Without Surface')

# ax2 = axes[0, 1]
# plot = ax2.pcolormesh(
#     corr_ent['LONGITUDE'], corr_ent['LATITUDE'], corr_ent,
#     cmap='nipy_spectral',
#     # levels=200,
#     vmin=0,
#     vmax=1,
# )
# ax2.coastlines()

# ax2.set_xlim(-180, 180)
# ax2.set_ylim(-90, 90)
# ax2.set_title('Without Entrainment')

# ax3 = axes[1, 0]
# plot = ax3.pcolormesh(
#     corr_ekman['LONGITUDE'], corr_ekman['LATITUDE'], corr_ekman,
#     cmap='nipy_spectral',
#     # levels=200,
#     vmin=0,
#     vmax=1,
# )
# ax3.coastlines()

# ax3.set_xlim(-180, 180)
# ax3.set_ylim(-90, 90)
# ax3.set_title('Without Ekman')

# ax4 = axes[1, 1]
# plot = ax4.pcolormesh(
#     corr_geo['LONGITUDE'], corr_geo['LATITUDE'], corr_geo,
#     cmap='nipy_spectral',
#     # levels=200,
#     vmin=0,
#     vmax=1,
# )
# ax4.coastlines()

# ax4.set_xlim(-180, 180)
# ax4.set_ylim(-90, 90)
# ax4.set_title('Without Geostrophic')

# fig.colorbar(plot, ax=axes.ravel().tolist())

# plt.show()

# %%
t_m_a_reynolds = load_and_prepare_dataset(
    "datasets/Reynolds/sst_anomalies-(2004-2018).nc"
)['anom']
t_m_a_simulated = simulation_full

rms_difference = np.sqrt(((t_m_a_reynolds - t_m_a_simulated) ** 2).mean(dim=['TIME']))
rms_simulated = np.sqrt((t_m_a_simulated ** 2).mean(dim=['TIME']))
rms_observed = np.sqrt((t_m_a_reynolds ** 2).mean(dim=['TIME']))

print("rms simulated", rms_difference.mean().item())
plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    rms_simulated['LONGITUDE'], rms_simulated['LATITUDE'], rms_simulated,
    cmap='nipy_spectral',
    # levels=200,
    vmin=0,
    vmax=3
)
ax.coastlines()

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
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

plt.colorbar(label=rms_simulated.attrs.get('units'))
plt.show()

print("rms observed", rms_observed.mean().item())
plt.figure(figsize=(20, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    rms_observed['LONGITUDE'], rms_observed['LATITUDE'], rms_observed,
    cmap='nipy_spectral',
    # levels=200,
    vmin=0,
    vmax=3
)
ax.coastlines()

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
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

plt.colorbar(label=rms_observed.attrs.get('units'))
plt.show()

rmse = rms_difference / rms_observed
print("normalised rmse", rmse.mean().item())
plt.figure(figsize=(25, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    rmse['LONGITUDE'], rmse['LATITUDE'], rmse,
    cmap='nipy_spectral',
    # levels=200,
    vmin=0,
    vmax=3
)
ax.coastlines()

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
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

plt.colorbar(label=rmse.attrs.get('units'))
plt.show()

corr = xr.corr(t_m_a_reynolds, t_m_a_simulated, dim='TIME')
print("corr", corr.mean().item())
plt.figure(figsize=(25, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    corr['LONGITUDE'], corr['LATITUDE'], corr,
    cmap='nipy_spectral',
    # levels=200,
    vmin=-1,
    vmax=1
)
ax.coastlines()

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
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

plt.colorbar(label=corr.attrs.get('units'))
plt.show()

# %%
time_to_see = 125.5
fig, axes = plt.subplots(
    nrows=2, ncols=1,
    figsize=(20, 16),
    subplot_kw={'projection': ccrs.PlateCarree()}
)
plt.subplots_adjust(hspace=0.1)

ax1 = axes[0]
plot = ax1.pcolormesh(
    t_m_a_reynolds['LONGITUDE'], t_m_a_reynolds['LATITUDE'], t_m_a_reynolds.sel(TIME=time_to_see),
    cmap='RdBu_r',
    # levels=200,
    vmin=-2,
    vmax=2,
)
ax1.coastlines()

ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)
# ax1.set_title(f'Reynolds SSTA, Time = {time_to_see}')

ax2 = axes[1]
plot = ax2.pcolormesh(
    t_m_a_simulated['LONGITUDE'], t_m_a_simulated['LATITUDE'], t_m_a_simulated.sel(TIME=time_to_see),
    cmap='RdBu_r',
    # levels=200,
    vmin=-2,
    vmax=2,
)
ax2.coastlines()

ax2.set_xlim(-180, 180)
ax2.set_ylim(-90, 90)
# ax2.set_title(f'Simulated SSTA, Time = {time_to_see}')

fig.colorbar(plot, ax=axes.ravel().tolist(), label="SSTA (°C)")

plt.show()

# %%
from utilities_plot import make_movie_2


make_movie_2(
    t_m_a_reynolds, t_m_a_simulated,
    vmin=-3, vmax=3,
    save_path="compare_Implicit.mp4"
)

# %%
