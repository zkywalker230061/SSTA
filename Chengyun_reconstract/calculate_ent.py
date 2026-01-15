"""
Calculate Entrainment term.

Chengyun Zhu
2026-1-13
"""

import xarray as xr
import numpy as np

from utilities import load_and_prepare_dataset
# from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


def _find_half_quantity(depth, quantity, mld):
    """
    Find the quantity at the mixed layer depth (hbar) from a temperature or salinity profile.

    Parameters
    ----------
    depth : np.ndarray
        1D array of depth values.
    quantity : np.ndarray
        1D array of quantity values corresponding to the depth profile.
    mld : float
        The mixed layer depth.

    Returns
    -------
    float
        The quantity at the mixed layer depth (hbar).
    """

    above_mld = np.where(depth <= mld)[0]
    below_mld = np.where(depth > mld)[0]

    # catch case where no MLD (should be only over land)
    if len(above_mld) == 0 or len(below_mld) == 0:
        return np.nan

    above_mld_index = above_mld[-1]
    below_mld_index = below_mld[0]
    mld_quantity = np.interp(
        mld,
        [depth[above_mld_index], depth[below_mld_index]],
        [quantity[above_mld_index], quantity[below_mld_index]]
    )
    return mld_quantity


def get_monthly_sub_temperature(t, h, month):
    """
    Get sub-layer temperature for a given month.

    Parameters
    ----------
    t : xarray.Dataset
        Dataset containing temperature profiles.
    h : xarray.Dataset
        Dataset containing mixed layer depth profiles.
    month : np.datetime64
        The month to process.

    Returns
    -------
    xarray.DataArray
        DataArray of sub-layer temperature.
    """

    t = t.sel(TIME=month)
    h = h.sel(TIME=month)

    slt = xr.apply_ufunc(
        _find_half_quantity,
        t['PRESSURE'], t, h,
        input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True
    )

    return slt


def save_sub_temperature_dataset():
    """Save the sub-layer temperature dataset to a NetCDF file."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Temperature-(2004-2018).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset('datasets/Temperature-(2004-2018).nc')['TEMPERATURE']
    h = load_and_prepare_dataset('datasets/Mixed_Layer_Depth-(2004-2018).nc')['MLD']

    monthly_datasets = []
    for month in t.TIME.values:
        monthly_datasets.append(
            get_monthly_sub_temperature(t, h, month)
        )
    t_sub = xr.concat(monthly_datasets, dim="TIME",
                      coords="different", compat='equals')

    t_sub['LATITUDE'].attrs = t['LATITUDE'].attrs
    t_sub['LONGITUDE'].attrs = t['LONGITUDE'].attrs
    t_sub.attrs['units'] = t.attrs['units']
    t_sub.attrs['long_name'] = (
        'Monthly Sub Layer Temperature Jan 2004 - Dec 2018 (15.0 year)'
    )
    t_sub.name = 'SUB_TEMPERATURE'

    save_file(
        t_sub,
        'datasets/Sub_Layer_Temperature-(2004-2018).nc'
    )


def get_monthly_sub_salinity(s, h, month):
    """
    Get sub-layer salinity for a given month.

    Parameters
    ----------
    s : xarray.Dataset
        Dataset containing salinity profiles.
    h : xarray.Dataset
        Dataset containing mixed layer depth profiles.
    month : np.datetime64
        The month to process.

    Returns
    -------
    xarray.DataArray
        DataArray of sub-layer salinity.
    """

    s = s.sel(TIME=month)
    h = h.sel(TIME=month)

    sls = xr.apply_ufunc(
        _find_half_quantity,
        s['PRESSURE'], s, h,
        input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True
    )

    return sls


def save_sub_salinity_dataset():
    """Save the sub-layer salinity dataset to a NetCDF file."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Salinity-(2004-2018).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset('datasets/Salinity-(2004-2018).nc')['SALINITY']
    h = load_and_prepare_dataset('datasets/Mixed_Layer_Depth-(2004-2018).nc')['MLD']

    monthly_datasets = []
    for month in s.TIME.values:
        monthly_datasets.append(
            get_monthly_sub_salinity(s, h, month)
        )
    s_sub = xr.concat(monthly_datasets, dim="TIME",
                      coords="different", compat='equals')

    s_sub['LATITUDE'].attrs = s['LATITUDE'].attrs
    s_sub['LONGITUDE'].attrs = s['LONGITUDE'].attrs
    s_sub.attrs['units'] = s.attrs['units']
    s_sub.attrs['long_name'] = (
        'Monthly Sub Layer Salinity Jan 2004 - Dec 2018 (15.0 year)'
    )
    s_sub.name = 'SUB_SALINITY'

    save_file(
        s_sub,
        'datasets/Sub_Layer_Salinity-(2004-2018).nc'
    )


def main():
    """Main function to calcuate entrainment term."""

    save_sub_temperature_dataset()
    save_sub_salinity_dataset()


if __name__ == "__main__":
    main()
