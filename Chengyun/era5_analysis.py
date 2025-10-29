"""
ERA5 dataset analysis.

Chengyun Zhu
2025/10/29
"""

from IPython.display import display

import numpy as np
# import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
# from rgargo_plot import visualise_dataset


def save_monthly_mean_wind_stress():
    """Save the monthly mean of the ERA5 wind stress dataset."""

    ds_era5 = load_and_prepare_dataset(
        "../datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress.nc",
    )
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
    display(ds_monthly_mean)
    ds_monthly_mean.to_netcdf(
        "../datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress-Seasonal_Cycle_Mean.nc"
    )


def save_monthly_mean_heat_flux():
    """Save the monthly mean of the ERA5 heat flux dataset."""

    ds_era5 = load_and_prepare_dataset(
        "../datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc",
    )
    lh = ds_era5['avg_slhtf']
    sh = ds_era5['avg_ishf']
    sw = ds_era5['avg_snswrf']
    lw = ds_era5['avg_snlwrf']
    lh_monthly_mean = get_monthly_mean(lh)
    sh_monthly_mean = get_monthly_mean(sh)
    sw_monthly_mean = get_monthly_mean(sw)
    lw_monthly_mean = get_monthly_mean(lw)

    ds_monthly_mean = lh_monthly_mean.to_dataset(name='MONTHLY_MEAN_avg_slhtf')
    ds_monthly_mean['MONTHLY_MEAN_avg_ishf'] = sh_monthly_mean
    ds_monthly_mean['MONTHLY_MEAN_avg_snswrf'] = sw_monthly_mean
    ds_monthly_mean['MONTHLY_MEAN_avg_snlwrf'] = lw_monthly_mean
    display(ds_monthly_mean)
    ds_monthly_mean.to_netcdf(
        "../datasets/ERA5-ARGO_Mean_Surface_Heat_Flux-Seasonal_Cycle_Mean.nc"
    )


def main():
    """Main function for era5_analysis.py"""

    save_monthly_mean_wind_stress()
    save_monthly_mean_heat_flux()


if __name__ == "__main__":
    main()
