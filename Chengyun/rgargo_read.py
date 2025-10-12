"""
Read RG-ARGO data file.

Chengyun Zhu
2025-10-10
"""

from IPython.display import display

import xarray as xr
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


def add_netcdf_time(ds, mid_month=False):
    """
    Add a standard netCDF time coordinate to the dataset.
    Original:
        'TIME' - axis T
    New:
        'time' - axis t

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset with time variable to decode.
    mid_month: bool, default True
        If True, set time to the middle of the month.
        If False, set time to the start of the month.

    Raises
    ------
    ValueError
        If mid_month is not True or False.
    """
    _, reference_date = ds['TIME'].attrs['units'].split('since')
    if mid_month is False:
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['TIME'], freq='MS')
    elif mid_month is True:
        ds['time'] = (pd.date_range(start=reference_date, periods=ds.sizes['TIME'], freq='MS')
                      + pd.DateOffset(days=14))
    else:
        raise ValueError("mid_month must be True or False.")
    ds['time'].attrs['calendar'] = '360_day'
    ds['time'].attrs['axis'] = 't'


if __name__ == "__main__":

    with xr.open_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc", decode_times=False
    ) as ds_temp:
        add_netcdf_time(ds_temp, mid_month=False)
        display(ds_temp)
        print(ds_temp.keys())

    with xr.open_dataset(
        "../datasets/RG_ArgoClim_Salinity_2019.nc", decode_times=False
    ) as ds_salt:
        add_netcdf_time(ds_salt, mid_month=False)
        display(ds_salt)
        print(ds_salt.keys())
