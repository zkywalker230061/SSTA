"""
Simulation of ocean temperature anomalies.

Chris O.S., tweaked by Chengyun.
2025-10-30
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from h_analysis import get_anomaly

matplotlib.use('TkAgg')


TEMP_DATA_PATH = "../datasets/Mixed_Layer_Temperature-(2004-2018).nc"
MLD_DATA_PATH = "../datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = "../datasets/Sub_Layer_Temperature-(2004-2018).nc"
ENT_VELOC_DATA_PATH = "../datasets/Entrainment_Velocity-(2004-2018).nc"

HEAT_FLUX_DATA_PATH = "../datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
ENT_FLUX_DATA_PATH = "../datasets/Entrainment_Heat_Flux_Anomaly-(2004-2018).nc"
GEO_DATA_PATH = ""
EK_DATA_PATH = "../datasets/Ekman_Current_Anomaly.nc"

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # average seconds in a month

temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
# display(temperature_ds)
temperature = temperature_ds['MLD_TEMPERATURE']
temperature_monthly_mean = get_monthly_mean(temperature)
# display(temperature_monthly_mean)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)
# display(temperature_anomaly)

mld_ds = load_and_prepare_dataset(MLD_DATA_PATH)
# mld_ds = mld_ds.rename({'MONTH': 'TIME'})
# display(mld_ds)
mld = mld_ds['MONTHLY_MEAN_MLD_PRESSURE']

t_sub_ds = load_and_prepare_dataset(T_SUB_DATA_PATH)
# display(t_sub_ds)
t_sub = t_sub_ds['SUB_TEMPERATURE']
t_sub_monthly_mean = get_monthly_mean(t_sub)
# display(t_sub_monthly_mean)
t_sub_anomaly = get_anomaly(t_sub, t_sub_monthly_mean)
# display(t_sub_anomaly)

w_e_ds = load_and_prepare_dataset(ENT_VELOC_DATA_PATH)
# display(w_e_ds)
w_e = w_e_ds['ENTRAINMENT_VELOCITY']
w_e_monthly_mean = get_monthly_mean(w_e)
# display(w_e_monthly_mean)

heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_DATA_PATH)
heat_flux = (
    heat_flux_ds['avg_slhtf']
    + heat_flux_ds['avg_ishf']
    + heat_flux_ds['avg_snswrf']
    + heat_flux_ds['avg_snlwrf']
)
heat_flux.attrs.update(
    units='W m**-2', long_name='Net Surface Heat Flux'
)
heat_flux.name = 'NET_HEAT_FLUX'
# display(heat_flux)
heat_flux_monthly_mean = get_monthly_mean(heat_flux)
# display(heat_flux_monthly_mean)
heat_flux_anomaly = get_anomaly(heat_flux, heat_flux_monthly_mean)
heat_flux_anomaly = heat_flux_anomaly.drop_vars(['MONTH'])
# display(heat_flux_anomaly)

ent = load_and_prepare_dataset(ENT_FLUX_DATA_PATH)
# display(ent)
ent_anomaly = ent['ENTRAINMENT_ANOMALY']

ekman = load_and_prepare_dataset(EK_DATA_PATH)
# display(ekman)
ekman_anomaly = ekman['Q_Ek_anom']


MONTH = 0.5
model_list = []
for month in temperature.TIME.values:
    # if month == MONTH:
    #     model_anomaly_ds = (
    #         temperature_anomaly.sel(TIME=MONTH)
    #         - temperature_anomaly.sel(TIME=MONTH)
    #     )
    #     model_anomaly_ds = model_anomaly_ds.expand_dims(TIME=[MONTH])
    # else:
    #     prev = model_anomaly_ds.sel(TIME=month-1)
    #     cur = (
    #         prev
    #         + SECONDS_MONTH * (
    #             heat_flux_anomaly.sel(TIME=month)
    #             + ent_anomaly.sel(TIME=month)
    #             # + GEO
    #             + ekman_anomaly.sel(TIME=month)
    #         ) / (RHO_O * C_O * mld.sel(MONTH=(month % 12 + 0.5)))
    #     )
    #     cur = cur.expand_dims(TIME=[month])
    #     model_anomaly_ds = xr.concat([model_anomaly_ds, cur], dim='TIME')
    temp = (
        (t_sub_anomaly.sel(TIME=month) + (heat_flux_anomaly.sel(TIME=month) + ekman_anomaly.sel(TIME=month))/(RHO_O * C_O * w_e_monthly_mean.sel(MONTH=(month % 12 + 0.5))))
        * (1 - np.exp(- (w_e_monthly_mean.sel(MONTH=(month % 12 + 0.5))*SECONDS_MONTH*month)/mld.sel(MONTH=(month % 12 + 0.5))))
    )
    model_list.append(temp.expand_dims(TIME=[month]))
model_anomaly_ds = xr.concat(model_list, dim='TIME')
display(model_anomaly_ds)

model_anomaly_ds.to_netcdf("../datasets/Simulated_SSTA-(2004-2018).nc")

# difference_from_obs = model_anomaly_ds - temperature_anomaly
# display(difference_from_obs)
# model_anomaly_ds = difference_from_obs  # to visualize the difference

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
