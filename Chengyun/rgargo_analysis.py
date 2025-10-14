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

from rgargo_read import load_and_prepare_dataset
from rgargo_plot import visualise_dataset


def main():
    """Main function for rgargo_analysis.py."""

    ds_temp = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )
    display(ds_temp)

    # meant_0: Mean Temperature for 15 years at surface
    meant_0 = ds_temp['ARGO_TEMPERATURE_MEAN'].isel(PRESSURE=0)
    # print(meant_0.min().item(), meant_0.max().item())
    visualise_dataset(meant_0, vmin=-2, vmax=31)

    # ta_0_2004jan: Temperature Anomaly at surface in 2004-01
    ta_0_2004jan = ds_temp['ARGO_TEMPERATURE_ANOMALY'].sel(TIME=0.5).isel(PRESSURE=0)
    # print(ta_0_2004jan.min().item(), ta_0_2004jan.max().item())
    visualise_dataset(ta_0_2004jan, vmin=-8, vmax=8)

    # ta_all_2024jan_e0n0: Temperature Anomaly at all depths in 2024-01 at (0°E, 0°N)
    ta_all_2004jan_e0n0 = ds_temp['ARGO_TEMPERATURE_ANOMALY'].sel(TIME=0.5).sel(
        LONGITUDE=0, LATITUDE=0, method='nearest'
    )
    # print(ta_all_2004jan_e0n0.min().item(), ta_all_2004jan_e0n0.max().item())
    visualise_dataset(ta_all_2004jan_e0n0)


if __name__ == "__main__":
    main()
