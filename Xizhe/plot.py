"""
RG-ARGO Data Analysis

Chengyun Zhu
2025-10-12
"""

from IPython.display import display

# import xarray as xr
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs

from read import load_and_prepare_dataset


if __name__ == "__main__":

    ds_temp = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )
    display(ds_temp)

    # meant_0: Mean Temperature for 15 years at surface
    meant_0 = ds_temp['ARGO_TEMPERATURE_MEAN'].isel(PRESSURE=0)
    display(meant_0)
    # print(meant_0.min().item(), meant_0.max().item())
    meant_0.plot(
        figsize=(10, 5),
        xlim=(-180, 180),
        ylim=(-90, 90),
        # cmap='coolwarm',
        cmap='RdBu_r',
        vmin=-2, vmax=31
    )

    ta_0_2004jan = ds_temp['ARGO_TEMPERATURE_ANOMALY'].sel(TIME=0.5).isel(PRESSURE=0)
    display(ta_0_2004jan)
    # print(ta_0_2004jan.min().item(), ta_0_2004jan.max().item())
    ta_0_2004jan.plot(
        figsize=(10, 5),
        xlim=(-180, 180),
        ylim=(-90, 90),
        cmap='RdBu_r',
        vmin=-8, vmax=8
    )
