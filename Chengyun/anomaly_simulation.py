"""
Simulation of observed anomalies.

Chengyun Zhu
2025-11-05
"""

# from IPython.display import display

# import xarray as xr
# import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from h_analysis import get_anomaly

matplotlib.use('TkAgg')


t = load_and_prepare_dataset(
    "../datasets/Mixed_Layer_Temperature-(2004-2018).nc",
)
# display(t)
t_monthly_mean = get_monthly_mean(t['MLD_TEMPERATURE'])
# display(t_monthly_mean)
t_anomaly = get_anomaly(t['MLD_TEMPERATURE'], t_monthly_mean)
# display(t_anomaly)
print(t_anomaly.max().item(), t_anomaly.min().item())

# make a movie
times = t_anomaly.TIME.values

fig, ax = plt.subplots(figsize=(12, 6))

ax = plt.axes(projection=ccrs.PlateCarree())

pcolormesh = ax.pcolormesh(
    t_anomaly.LONGITUDE.values,
    t_anomaly.LATITUDE.values,
    t_anomaly.isel(TIME=0),
    cmap='RdBu_r',
    vmin=-5, vmax=5
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

cbar = plt.colorbar(pcolormesh, ax=ax, label=t_anomaly.attrs.get('units'))
# cbar = plt.colorbar(contourf, ax=ax, label=t_anomaly.attrs.get('units'))

title = ax.set_title(f'Time = {times[0]}')


def update(frame):
    pcolormesh.set_array(t_anomaly.isel(TIME=frame).values.ravel())
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
ani.save("SSTA.gif", writer='pillow', fps=5)

plt.show()
