"""
Simulation of observed anomalies.

Chengyun Zhu
2025-11-05
"""

# from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from utils_read_nc import load_and_prepare_dataset
from chris_utils import get_monthly_mean, get_anomaly

matplotlib.use('TkAgg')

observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"

observed_temp_ds = xr.open_dataset(observed_path, decode_times=False)

observed_temperature_monthly_average = get_monthly_mean(observed_temp_ds['__xarray_dataarray_variable__'])
observed_temperature_anomaly = get_anomaly(observed_temp_ds, '__xarray_dataarray_variable__', observed_temperature_monthly_average)
observed_temperature_anomaly = observed_temperature_anomaly['__xarray_dataarray_variable___ANOMALY']


# display(t_anomaly)
print(observed_temperature_anomaly.max().item(), observed_temperature_anomaly.min().item())

# make a movie
times = observed_temperature_anomaly.TIME.values

fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.axes(projection=ccrs.PlateCarree())

pcolormesh = ax.pcolormesh(
    observed_temperature_anomaly.LONGITUDE.values,
    observed_temperature_anomaly.LATITUDE.values,
    observed_temperature_anomaly.isel(TIME=0),
    cmap='RdBu_r',
    vmin=-2, vmax=2
)
# contourf = ax.contourf(
#     t_anomaly.LONGITUDE.values,
#     t_anomaly.LATITUDE.values,
#     t_anomaly.isel(TIME=0),
#     cmap='RdBu_r',
#     levels=200,
#     vmin=-5, vmax=5,
#     # extend='min'
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

cbar = plt.colorbar(pcolormesh, ax=ax, label=observed_temperature_anomaly.attrs.get('units'))
# cbar = plt.colorbar(contourf, ax=ax, label=t_anomaly.attrs.get('units'))

title = ax.set_title(f'Time = {times[0]}')


def update(frame):
    pcolormesh.set_array(observed_temperature_anomaly.isel(TIME=frame).values.ravel())
    cbar.update_normal(pcolormesh)
    title.set_text(
        f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}'
    )
    return [pcolormesh, title]

    # contourf.set_array(t_anomaly.isel(TIME=frame).values.ravel())
    # cbar.update_normal(contourf)

    # contourf = ax.contourf(
    #     t_anomaly.LONGITUDE.values,
    #     t_anomaly.LATITUDE.values,
    #     t_anomaly.isel(TIME=frame),
    #     cmap='RdBu_r',
    #     levels=200,
    #     vmin=-5, vmax=5,
    #     # extend='min'
    # )
    # title.set_text(
    #     f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}'
    # )
    # return [contourf, title]


ani = animation.FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)

# writer = animation.FFMpegWriter(
#     fps=15, bitrate=1800
# )
# ani.save("movie.mp4", writer=writer)
# ani.save("SSTA.gif", writer='pillow', fps=5)

plt.show()





