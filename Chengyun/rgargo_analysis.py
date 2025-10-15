"""
RG-ARGO Data Analysis

Chengyun Zhu
2025-10-12
"""

# from IPython.display import display

# import xarray as xr
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs

import xarray as xr

from rgargo_read import load_and_prepare_dataset
# from rgargo_plot import visualise_dataset


MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3,
    'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9,
    'Oct': 10, 'Nov': 11, 'Dec': 12
}


def get_monthly_mean(
    da: xr.DataArray,
) -> xr.DataArray:
    """
    Get the monthly mean of the DataArray.

    Parameters
    ----------
    da: xarray.DataArray
        The DataArray to process.

    Returns
    -------
    xarray.DataArray
        The monthly mean DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """
    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    monthly_means = []
    for _, month_num in MONTHS.items():
        monthly_means.append(
            da.sel(TIME=da['TIME'][month_num-1::12]).mean(dim='TIME')
        )
    monthly_mean_da = xr.concat(monthly_means, dim='MONTH')
    monthly_mean_da = monthly_mean_da.assign_coords(MONTH=list(MONTHS.values()))
    monthly_mean_da['MONTH'].attrs['units'] = 'month'
    monthly_mean_da['MONTH'].attrs['axis'] = 'M'
    monthly_mean_da.attrs['units'] = da.attrs.get('units')
    monthly_mean_da.attrs['long_name'] = f"Monthly Mean of {da.attrs.get('long_name')}"
    monthly_mean_da.name = f"MONTHLY_MEAN_{da.name}"
    return monthly_mean_da


def save_monthly_mean_anomalies():
    """Save the monthly mean of the anomaly dataset."""

    ds_temp = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )
    # display(ds_temp)
    ta = ds_temp['ARGO_TEMPERATURE_ANOMALY']
    # display(ta)
    ta_monthly_mean = get_monthly_mean(ta)
    # display(ta_monthly_mean)
    # visualise_dataset(
    #     ta_monthly_mean.sel(PRESSURE=0, MONTH=1, method='nearest')
    # )
    ta_monthly_mean.to_netcdf("../datasets/ARGO_TEMPERATURE_ANOMALY_Monthly_Mean.nc")

    ds_salt = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Salinity_2019.nc",
    )
    # display(ds_salt)
    sa = ds_salt['ARGO_SALINITY_ANOMALY']
    # display(sa)
    sa_monthly_mean = get_monthly_mean(sa)
    # display(sa_monthly_mean)
    # visualise_dataset(
    #     sa_monthly_mean.sel(PRESSURE=0, MONTH=1, method='nearest')
    # )
    sa_monthly_mean.to_netcdf("../datasets/ARGO_SALINITY_ANOMALY_Monthly_Mean.nc")


def save_monthly_mean_temperature():
    """Save the monthly mean temperature dataset."""

    ta_monthly_mean = load_and_prepare_dataset(
        "../datasets/ARGO_TEMPERATURE_ANOMALY_Monthly_Mean.nc",
    )['MONTHLY_MEAN_ARGO_TEMPERATURE_ANOMALY']
    # display(ta_monthly_mean)
    # visualise_dataset(
    #     ta_monthly_mean.sel(PRESSURE=0, MONTH=1, method='nearest')
    # )
    t_15years_mean = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )['ARGO_TEMPERATURE_MEAN']
    # display(t_15years_mean)
    # visualise_dataset(
    #     t_15years_mean.sel(PRESSURE=0, method='nearest')
    # )

    # add two toghether, monthly mean have dimension MONTH, 15 years mean not
    t_15years_mean = t_15years_mean.expand_dims(MONTH=12)
    t_monthly_mean = t_15years_mean + ta_monthly_mean

    t_monthly_mean.attrs['units'] = ta_monthly_mean.attrs.get('units')
    t_monthly_mean.attrs['long_name'] = (
        "Monthly Mean of Temperature Jan 2004 - Dec 2018 (15.0 year)"
    )
    t_monthly_mean.name = "MONTHLY_MEAN_TEMPERATURE"
    # display(t_monthly_mean)
    # visualise_dataset(
    #     t_monthly_mean.sel(PRESSURE=0, MONTH=1, method='nearest')
    # )
    # visualise_dataset(
    #     t_monthly_mean.sel(MONTH=1, LONGITUDE=0, LATITUDE=0, method='nearest')
    # )
    t_monthly_mean.to_netcdf("../datasets/Temperature_Monthly_Mean.nc")


def save_monthly_mean_salinity():
    """Save the monthly mean salinity dataset."""

    sa_monthly_mean = load_and_prepare_dataset(
        "../datasets/ARGO_SALINITY_ANOMALY_Monthly_Mean.nc",
    )['MONTHLY_MEAN_ARGO_SALINITY_ANOMALY']
    # display(sa_monthly_mean)
    # visualise_dataset(
    #     sa_monthly_mean.sel(PRESSURE=0, MONTH=1, method='nearest')
    # )
    s_15years_mean = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Salinity_2019.nc",
    )['ARGO_SALINITY_MEAN']
    # display(s_15years_mean)
    # visualise_dataset(
    #     s_15years_mean.sel(PRESSURE=0, method='nearest')
    # )

    # add two toghether, monthly mean have dimension MONTH, 15 years mean not
    s_15years_mean = s_15years_mean.expand_dims(MONTH=12)
    s_monthly_mean = s_15years_mean + sa_monthly_mean

    s_monthly_mean.attrs['units'] = sa_monthly_mean.attrs.get('units')
    s_monthly_mean.attrs['long_name'] = (
        "Monthly Mean of Salinity Jan 2004 - Dec 2018 (15.0 year)"
    )
    s_monthly_mean.name = "MONTHLY_MEAN_SALINITY"
    # display(s_monthly_mean)
    # visualise_dataset(
    #     s_monthly_mean.sel(PRESSURE=0, MONTH=1, method='nearest')
    # )
    # visualise_dataset(
    #     s_monthly_mean.sel(MONTH=1, LONGITUDE=0, LATITUDE=0, method='nearest')
    # )
    s_monthly_mean.to_netcdf("../datasets/Salinity_Monthly_Mean.nc")


def main():
    """Main function for rgargo_analysis.py."""
    # save_monthly_mean_anomalies()

    # save_monthly_mean_temperature()
    # save_monthly_mean_salinity()


if __name__ == "__main__":
    main()
