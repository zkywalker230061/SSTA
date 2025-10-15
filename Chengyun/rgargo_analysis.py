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
    monthly_mean_da.name = f"Monthly Mean of {da.name}"
    return monthly_mean_da


def main():
    """Main function for rgargo_analysis.py."""

    # ds_temp = load_and_prepare_dataset(
    #     "../datasets/RG_ArgoClim_Temperature_2019.nc",
    # )
    # display(ds_temp)
    # visualise_dataset(ds_temp)  # Call Kal'tsit.

    # ta = ds_temp['ARGO_TEMPERATURE_ANOMALY']
    # display(ta)
    # ta_monthly_mean = get_monthly_mean(ta)
    # display(ta_monthly_mean)
    # visualise_dataset(ta_monthly_mean.sel(
    #     PRESSURE=0, MONTH=1, method='nearest')
    # )
    # ta_monthly_mean.to_netcdf("../datasets/ARGO_TEMPERATURE_ANOMALY_Monthly_Mean.nc")

    ta_monthly_mean = load_and_prepare_dataset(
        "../datasets/ARGO_TEMPERATURE_ANOMALY_Monthly_Mean.nc",
    )
    display(ta_monthly_mean)


if __name__ == "__main__":
    main()
