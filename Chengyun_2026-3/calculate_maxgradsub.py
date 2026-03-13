"""
Module to calculate max gradient quantity for the sub layer.

Chengyun
2026-3-5
"""

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


def _get_monthly_sub_temperature(t, h, month):
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


def save_sub_temperature():
    """Save the sub-layer temperature dataset to a NetCDF file."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Temperature-(2004-2025).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset('datasets/Temperature-(2004-2025).nc')['TEMPERATURE']
    h = load_and_prepare_dataset('datasets/Mixed_Layer_Depth-(2004-2025).nc')['MLD']

    monthly_datasets = []
    for month in t.TIME.values:
        monthly_datasets.append(
            _get_monthly_sub_temperature(t, h, month)
        )
    t_sub = xr.concat(monthly_datasets, dim="TIME",
                      coords="different", compat='equals')

    t_sub['LATITUDE'].attrs = t['LATITUDE'].attrs
    t_sub['LONGITUDE'].attrs = t['LONGITUDE'].attrs
    t_sub.attrs['units'] = t.attrs['units']
    t_sub.attrs['long_name'] = (
        'Monthly Sub Layer Temperature Jan 2004 - Dec 2025 (22.0 year)'
    )
    t_sub.name = 'SUB_TEMPERATURE'

    save_file(
        t_sub,
        'datasets/Sub_Layer_Temperature-(2004-2025).nc'
    )


def _get_monthly_sub_salinity(s, h, month):
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


def save_sub_salinity():
    """Save the sub-layer salinity dataset to a NetCDF file."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Salinity-(2004-2025).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset('datasets/Salinity-(2004-2025).nc')['SALINITY']
    h = load_and_prepare_dataset('datasets/Mixed_Layer_Depth-(2004-2025).nc')['MLD']

    monthly_datasets = []
    for month in s.TIME.values:
        monthly_datasets.append(
            _get_monthly_sub_salinity(s, h, month)
        )
    s_sub = xr.concat(monthly_datasets, dim="TIME",
                      coords="different", compat='equals')

    s_sub['LATITUDE'].attrs = s['LATITUDE'].attrs
    s_sub['LONGITUDE'].attrs = s['LONGITUDE'].attrs
    s_sub.attrs['units'] = s.attrs['units']
    s_sub.attrs['long_name'] = (
        'Monthly Sub Layer Salinity Jan 2004 - Dec 2025 (22.0 year)'
    )
    s_sub.name = 'SUB_SALINITY'

    save_file(
        s_sub,
        'datasets/Sub_Layer_Salinity-(2004-2025).nc'
    )


def main():
    """Main function to save sub-layer temperature and salinity for max-gradient method."""

    pass


if __name__ == "__main__":
    main()
