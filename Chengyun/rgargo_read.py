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


def _time_standard(ds: xr.Dataset, mid_month: bool = False) -> xr.Dataset:
    """
    Add a standard netCDF time coordinate to the dataset.
    Original:
        'TIME' - axis T
    New:
        'time' - axis t, calendar 360_day

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset with time variable to decode.
    mid_month: bool
        Default is False.
        If True, set time to the middle of the month.
        If False, set time to the start of the month.

    Returns
    -------
    xarray.Dataset
        Dataset with new 'time' coordinate.

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
    return ds


def _longitude_180(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert longitude from [0, 360] to [-180, 180].

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset with 'LONGITUDE' coordinate.

    Returns
    -------
    xarray.Dataset
        Dataset with 'LONGITUDE' in [-180, 180] and sorted.
    """
    lon_atrib = ds.coords['LONGITUDE'].attrs
    ds['LONGITUDE'] = ((ds['LONGITUDE'] + 180) % 360) - 180
    ds = ds.sortby(ds['LONGITUDE'])
    ds['LONGITUDE'].attrs.update(lon_atrib)
    ds['LONGITUDE'].attrs['modulo'] = 180
    return ds


def load_and_prepare_dataset(
        filepath: str,
        time_standard: bool = False,
        time_standard_mid_month: bool = False,
        longitude_180: bool = True
) -> xr.Dataset | None:
    """
    Load, standardize time, and convert longitude for RG-ARGO dataset.

    Parameters
    ----------
    filepath: str
        Path to the RG-ARGO netCDF file.
    time_standard: bool
        Default is False.
        If True, standardize the time coordinate.
    time_standard_mid_month: bool
        Default is False.
        If True, standardize the time coordinate to the middle of the month.
    longitude_180: bool
        Default is True.
        If True, convert longitude to [-180, 180].

    Returns
    -------
    xarray.Dataset or None
        The processed dataset, or None if loading fails.
    """
    try:
        with xr.open_dataset(filepath, decode_times=False) as ds:
            if time_standard:
                ds = _time_standard(ds)
            if time_standard_mid_month:
                ds = _time_standard(ds, mid_month=True)
            if longitude_180:
                ds = _longitude_180(ds)
            return ds
    except (OSError, ValueError, KeyError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


if __name__ == "__main__":

    ds_temp = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )
    display(ds_temp)

    ds_salt = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Salinity_2019.nc",
    )
    display(ds_salt)

    ds_202509 = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_202509_2019.nc",
    )
    display(ds_202509)

    display(dir(ds_temp))
