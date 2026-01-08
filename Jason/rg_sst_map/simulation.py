"""
Simulation adapted from code made by Chris and Chengyun
Adapted an Implicit scheme and various other schemes
"""

#%%
import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
#import esmpy as ESMF
from read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
matplotlib.use('TkAgg')

#HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "../datasets/heat_flux_interpolated_all_contributions.nc"
#HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
MLD_TEMP_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature-(2004-2018).nc"
MLD_DEPTH_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
#TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
HEAT_FLUX_DATA_PATH = r"C:\Users\jason\MSciProject\ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
EK_DATA_PATH = r"C:\Users\jason\MSciProject\Ekman_Current_Anomaly.nc"

mld_temperature_ds = xr.open_dataset(MLD_TEMP_PATH, decode_times=False)
mld_depth_ds = xr.open_dataset(MLD_DEPTH_PATH, decode_times=False)
heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_DATA_PATH)
ekman_ds = load_and_prepare_dataset(EK_DATA_PATH)

temperature = mld_temperature_ds['__xarray_dataarray_variable__']
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

mld_depth_ds = mld_depth_ds.rename({'MONTH': 'TIME'})

heat_flux = (heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_ishf'] + heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_snlwrf'])
heat_flux.attrs.update(units='W m**-2', long_name='Net Surface Heat Flux')
heat_flux.name = 'NET_HEAT_FLUX'
heat_flux_monthly_mean = get_monthly_mean(heat_flux)
heat_flux_anomaly = get_anomaly(heat_flux, heat_flux_monthly_mean)
heat_flux_anomaly = heat_flux_anomaly.drop_vars(['MONTH'])

ekman_anomaly = ekman_ds['Q_Ek_anom']

# print(heat_flux_anomaly)
# print('temperature_anomaly: \n',temperature_anomaly)
# print('heat_flux_ds: \n', heat_flux_ds)
# print('mld_ds:\n',mld_depth_ds )


#%%
RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # average seconds in a month
GAMMA = 10



MONTH = 0.5
times = temperature.TIME.values

COMPUTE_IMPLICIT = True
COMPUTE_EXPLICIT = False

if COMPUTE_IMPLICIT == True:
    for i, month in enumerate(times):
        if month == MONTH:
            model_anomaly_ds = (
                temperature_anomaly.sel(TIME=MONTH, method='nearest', tolerance=0.51)
                - temperature_anomaly.sel(TIME=MONTH, method='nearest', tolerance=0.51)
            ).expand_dims(TIME=[MONTH])
        else:
            prev = model_anomaly_ds.sel(TIME=times[i-1], method='nearest', tolerance=0.51)

            # month-of-year mid-point 
            moy = ((month - 0.5) % 12) + 0.5

            denominator = (
                RHO_O * C_O *
                mld_depth_ds
                .sel(TIME=moy, method='nearest', tolerance=0.51)['MONTHLY_MEAN_MLD_PRESSURE']
            )
            damp_factor  = GAMMA / denominator

            forcing_term = (
                heat_flux_anomaly.sel(TIME=month, method='nearest', tolerance=0.51) +
                ekman_anomaly.sel(TIME=month, method='nearest', tolerance=0.51)
            ) / denominator

            next_ = (prev + SECONDS_MONTH * forcing_term) / (1 + SECONDS_MONTH * damp_factor)
            model_anomaly_ds = xr.concat([model_anomaly_ds, next_.expand_dims(TIME=[month])], dim='TIME')

if COMPUTE_EXPLICIT == True:
    for i, month in enumerate(times):
        if month == MONTH:
            model_anomaly_ds = (
                temperature_anomaly.sel(TIME=MONTH, method='nearest', tolerance=0.51)
                - temperature_anomaly.sel(TIME=MONTH, method='nearest', tolerance=0.51)
            ).expand_dims(TIME=[MONTH])
        else:
            prev_time = times[i-1]
            prev = model_anomaly_ds.sel(TIME=prev_time, method='nearest', tolerance=0.51)

            moy = ((prev_time - 0.5) % 12) + 0.5
            denominator = (
                RHO_O * C_O *
                mld_depth_ds
                .sel(TIME=moy, method='nearest', tolerance=0.51)['MONTHLY_MEAN_MLD_PRESSURE']
            )

            damp_factor = GAMMA / denominator
            forcing_term = (
                heat_flux_anomaly.sel(TIME=moy, method='nearest', tolerance=0.51)
                + ekman_anomaly.sel(TIME=moy, method='nearest', tolerance=0.51)
            ) / denominator
            
            cur = (1 - SECONDS_MONTH * damp_factor) * prev + SECONDS_MONTH * forcing_term
            cur = cur.expand_dims(TIME=[month])
            model_anomaly_ds = xr.concat([model_anomaly_ds, cur], dim='TIME')

# model_anomaly_ds.sel(TIME=100.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r')
# plt.show()

# make a movie
times = model_anomaly_ds.TIME.values

fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.axes(projection=ccrs.PlateCarree())

pcolormesh = ax.pcolormesh(
    model_anomaly_ds.LONGITUDE.values,
    model_anomaly_ds.LATITUDE.values,
    model_anomaly_ds.isel(TIME=0),
    cmap='RdBu_r',
    vmin=-20, vmax=20
)
# contourf = ax.contourf(
#     model_anomaly_ds.LONGITUDE.values,
#     model_anomaly_ds.LATITUDE.values,
#     model_anomaly_ds.isel(TIME=0),
#     cmap='RdBu_r',
#     levels=200,
#     vmin=-20, vmax=20
# )
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

cbar = plt.colorbar(pcolormesh, ax=ax, label=model_anomaly_ds.attrs.get('units'))
# cbar = plt.colorbar(contourf, ax=ax, label=model_anomaly_ds.attrs.get('units'))

title = ax.set_title(f'Time = {times[0]}')


def update(frame):
    pcolormesh.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
    cbar.update_normal(pcolormesh)
    title.set_text(
        f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}'
    )
    return [pcolormesh, title]
    # contourf.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
    # cbar.update_normal(contourf)
    # contourf = ax.contourf(
    #     model_anomaly_ds.LONGITUDE.values,
    #     model_anomaly_ds.LATITUDE.values,
    #     model_anomaly_ds.isel(TIME=frame),
    #     cmap='RdBu_r',
    #     levels=200,
    #     vmin=-20, vmax=20
    # )
    # title.set_text(
    #     f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}'
    # )
    # return [contourf, title]


animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
plt.show()



#%%
#--Chris Code----------------------------------------------
# model_anomalies = []
# added_baseline = False
# for month in heat_flux_anomaly_ds.TIME.values:
#     if not added_baseline:
#         base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
#         base = base.expand_dims(TIME=[month])  # <-- keep its time
#         model_anomalies.append(base)
#         added_baseline = True
#     else:
#         prev = model_anomalies[-1].isel(TIME=-1)
#         cur = prev + (30.4375 * 24 * 60 * 60) * heat_flux_anomaly_ds.sel(TIME=month)['NET_HEAT_FLUX_ANOMALY'] / (mld_ds.sel(TIME=month)['MLD_PRESSURE'] * 1025 * 4100)
#         cur = cur.expand_dims(TIME=[month])
#         model_anomalies.append(cur)
# model_anomaly_ds = xr.concat(model_anomalies, 'TIME')
# model_anomaly_ds = model_anomaly_ds.drop_vars(["PRESSURE"])
# print(model_anomaly_ds)

# times = model_anomaly_ds.TIME.values

# fig, ax = plt.subplots()
# pcolormesh = ax.pcolormesh(model_anomaly_ds.LONGITUDE.values, model_anomaly_ds.LATITUDE.values, model_anomaly_ds.isel(TIME=0), cmap='RdBu_r')
# title = ax.set_title(f'Time = {times[0]}')

# cbar = plt.colorbar(pcolormesh, ax=ax, label='Modelled anomaly from surface heat flux')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')


# def update(frame):
#     pcolormesh.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
#     #pcolormesh.set_clim(vmin=float(model_anomaly_ds.isel(TIME=frame).min()), vmax=float(model_anomaly_ds.isel(TIME=frame).max()))
#     pcolormesh.set_clim(vmin=-20, vmax=20)
#     cbar.update_normal(pcolormesh)
#     title.set_text(f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}')
#     return [pcolormesh, title]

# animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
# plt.show()

