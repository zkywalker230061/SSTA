"""
ERA5 dataset analysis. TEMPORARY FILE.

Chengyun Zhu
2025/10/29
"""

from IPython.display import display

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
# from rgargo_plot import visualise_dataset


def main():
    """Main function for era5_analysis.py"""
    ds_era5 = load_and_prepare_dataset(
        "../datasets/era5_interpolated.nc",
    )
    ds_era5_uninterpolated = xr.open_dataset(
        "../datasets/ERA5_Mean_Turbulent_Surface_Stress.nc"
    )
    # update attributes
    ds_era5['avg_iews'].attrs = ds_era5_uninterpolated['avg_iews'].attrs
    ds_era5['avg_inss'].attrs = ds_era5_uninterpolated['avg_inss'].attrs
    ds_era5['TIME'].attrs.update({
        'units': 'months since 2004-01-01 00:00:00',
        'time_origin': '01-JAN-2004 00:00:00',
        'axis': 'T'
    })
    display(ds_era5)
    ew = ds_era5['avg_iews']
    ns = ds_era5['avg_inss']
    ew_monthly_mean = get_monthly_mean(ew)
    ns_monthly_mean = get_monthly_mean(ns)
    # display(ew_monthly_mean)
    # display(ns_monthly_mean)

    MONTH = 1
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    x = ds_era5['LONGITUDE'].values
    y = ds_era5['LATITUDE'].values
    u = ew_monthly_mean.sel(
        MONTH=MONTH
    ).values
    v = ns_monthly_mean.sel(
        MONTH=MONTH
    ).values
    magnitude = np.hypot(u, v)
    ax.streamplot(
        x, y, u, v,
        transform=ccrs.PlateCarree(),
        linewidth=1, density=1, color=magnitude
        )
    plt.show()

    ds_monthly_mean = ew_monthly_mean.to_dataset(name='MONTHLY_MEAN_avg_iews')
    ds_monthly_mean['MONTHLY_MEAN_avg_inss'] = ns_monthly_mean
    # display(ds_monthly_mean)
    ds_monthly_mean.to_netcdf(
        "../datasets/ERA5_Mean_Turbulent_Surface_Stress-Seasonal_Cycle_Mean.nc"
    )


if __name__ == "__main__":
    main()
