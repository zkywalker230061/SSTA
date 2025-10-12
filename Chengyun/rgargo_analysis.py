"""
RG-ARGO Data Analysis

Chengyun Zhu
2025-10-12
"""

from IPython.display import display

import xarray as xr
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs

from rgargo_read import add_netcdf_time


if __name__ == "__main__":

    with xr.open_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc", decode_times=False
    ) as ds_temp:
        add_netcdf_time(ds_temp, mid_month=False)
        display(ds_temp)
        print(ds_temp.keys())
        # plot temperature at surface at the start time; test
        tm2d = ds_temp['ARGO_TEMPERATURE_MEAN'].isel(PRESSURE=0)
        # display(tm2d)
        # print(tm2d.min().item(), tm2d.max().item())
        tm2d.plot(
            figsize=(10, 5),
            ylim=(-90, 90),
            cmap='coolwarm',
            vmin=-2, vmax=31
        )
