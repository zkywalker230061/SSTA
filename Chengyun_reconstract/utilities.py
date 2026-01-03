"""
Useful utility functions for Chengyun's reconstruction code.

Chengyun Zhu
2025-11-27
"""

import time

import xarray as xr


MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3,
    'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9,
    'Oct': 10, 'Nov': 11, 'Dec': 12
}


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
        longitude_180: bool = True
) -> xr.Dataset | None:
    """
    Load, standardize time, and convert longitude for RG-ARGO dataset.

    Parameters
    ----------
    filepath: str
        Path to the RG-ARGO netCDF file.
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
            if longitude_180:
                ds = _longitude_180(ds)
            return ds
    except (OSError, ValueError, KeyError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


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
    monthly_mean_da.attrs['long_name'] = f"Seasonal Cycle Mean of {da.attrs.get('long_name')}"
    monthly_mean_da.name = f"MONTHLY_MEAN_{da.name}"
    return monthly_mean_da


def get_anomaly(
    da: xr.DataArray,
    monthly_mean_da: xr.DataArray
) -> xr.DataArray:
    """
    Get the anomaly of the DataArray based on the provided monthly mean.

    Parameters
    ----------
    da: xarray.DataArray
        Input DataArray with 'TIME' coordinate.
    monthly_mean_da: xarray.DataArray
        Monthly mean DataArray with 'MONTH' coordinate.

    Returns
    -------
    xarray.DataArray
        Anomaly DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """

    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    anomalies = []
    for month_num in da.coords['TIME']:
        month_num = month_num.values
        month_mean_num = int((month_num + 0.5) % 12)
        if month_mean_num == 0:
            month_mean_num = 12
        anomalies.append(
            da.sel(TIME=month_num) - monthly_mean_da.sel(MONTH=month_mean_num)
        )
    anomaly_da = xr.concat(anomalies, "TIME")
    anomaly_da.attrs['units'] = da.attrs.get('units')
    anomaly_da.attrs['long_name'] = f"Anomaly of {da.attrs.get('long_name')}"
    anomaly_da.name = f"ANOMALY_{da.name}"
    return anomaly_da


def save_file(
    data: xr.Dataset,
    filepath: str,
) -> None:
    """
    Save dataset to a netCDF file.

    Parameters
    ----------
    data: xarray.Dataset
        Dataset to save.
    filepath: str
        Path to save the netCDF file.
    """

    with open("logs.txt", "r+", encoding="utf-8") as logs:
        if filepath in logs.read():
            pass
        else:
            logs.write(filepath + "\n")
            logs.write(str(time.time()) + "\n")
            data.to_netcdf(filepath)


def main():
    """Main function for testing utilities."""

    pass


if __name__ == "__main__":
    main()
