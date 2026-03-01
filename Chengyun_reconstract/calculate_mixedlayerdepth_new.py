"""
Calculate Mixed Layer Depth (h) and (hbar) datasets - new method.

Chengyun Zhu
2026-1-29
"""

import xarray as xr
import numpy as np
import gsw

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file

HBAR_DDIFF = 0.03
MAX_DEPTH = float(2000)


def _find_half_depth(density_anomaly_profile, pressure, density_anomaly_surface_mean):
    """
    Find the mixed layer depth (hbar) from a density anomaly profile.
    - new method: not interpolating, but return the ARGO-pressure immediately above MLD.

    Parameters
    ----------
    density_anomaly_profile : np.ndarray
        1D array of density anomaly values at different pressures.
    pressure : np.ndarray
        1D array of pressure values corresponding to the density profile.
    density_anomaly_surface_mean : float
        Mean density anomaly near the surface.

    Returns
    -------
    float
        The pressure at the mixed layer depth (hbar).
        Returns MAX_DEPTH if not found, or -inf for land.
    """

    # sort pressure and temperature to be in order of increasing pressure
    indices_increasing_pressure = np.argsort(pressure)
    pressure = pressure[indices_increasing_pressure]
    density_anomaly_profile = density_anomaly_profile[indices_increasing_pressure]

    sigma_0 = density_anomaly_surface_mean  # density at surface == first reading

    # land
    if np.isnan(sigma_0):
        return -np.inf

    sigma_mld_min = sigma_0 - HBAR_DDIFF
    sigma_mld_max = sigma_0 + HBAR_DDIFF
    pressures_below_sigma_0 = np.where(density_anomaly_profile <= sigma_mld_min)[0]
    pressures_above_sigma_0 = np.where(density_anomaly_profile >= sigma_mld_max)[0]

    # only sigma below sigma_mld_min
    if len(pressures_below_sigma_0) != 0 and len(pressures_above_sigma_0) == 0:
        index2 = pressures_below_sigma_0[0]
        index1 = index2 - 1
        sigma_mld = sigma_mld_min
    # only sigma above sigma_mld_max
    elif len(pressures_below_sigma_0) == 0 and len(pressures_above_sigma_0) != 0:
        index2 = pressures_above_sigma_0[0]
        index1 = index2 - 1
        sigma_mld = sigma_mld_max
    # both found, check which is closer to surface
    elif len(pressures_below_sigma_0) != 0 and len(pressures_above_sigma_0) != 0:
        if pressures_above_sigma_0[0] < pressures_below_sigma_0[0]:
            index2 = pressures_above_sigma_0[0]
            index1 = index2 - 1
            sigma_mld = sigma_mld_max
        else:
            index2 = pressures_below_sigma_0[0]
            index1 = index2 - 1
            sigma_mld = sigma_mld_min
    else:
        return MAX_DEPTH

    mld = np.interp(
        sigma_mld,
        [density_anomaly_profile[index1], density_anomaly_profile[index2]],
        [pressure[index1], pressure[index2]]
    )
    if mld <= MAX_DEPTH:
        return min(pressure[index1], pressure[index2])
    return MAX_DEPTH


def _find_surface_density_anomaly_mean(ds: xr.Dataset) -> None:
    """
    Find the mean density anomaly near the surface (top 10 dbar).

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing density anomaly and pressure data.
    """

    ds_surface = ds.sel(PRESSURE=[2.5])
    sigma0_surface_mean = ds_surface.DENSITY_ANOMALY.mean(dim='PRESSURE')
    ds['SURFACE_DENSITY_ANOMALY_MEAN'] = sigma0_surface_mean


# def _find_depth_iteration(density_anomaly_profile, pressure):
#     """
#     Find the mixed layer depth (hbar) from a density anomaly profile using iterative method.

#     Parameters
#     ----------
#     density_anomaly_profile : np.ndarray
#         1D array of density anomaly values at different pressures.
#     pressure : np.ndarray
#         1D array of pressure values corresponding to the density profile.

#     Returns
#     -------
#     float
#         The pressure at the mixed layer depth (hbar).
#         Returns MAX_DEPTH if not found, or -inf for land.
#     """

#     # sort pressure and temperature to be in order of increasing pressure
#     indices_increasing_pressure = np.argsort(pressure)
#     pressure = pressure[indices_increasing_pressure]
#     density_anomaly_profile = density_anomaly_profile[indices_increasing_pressure]

#     sigma0_surface_mean = density_anomaly_profile[:1].mean()

#     # land
#     if np.isnan(sigma0_surface_mean):
#         return -np.inf

#     i = 1
#     while (
#         abs(density_anomaly_profile[i+1] - sigma0_surface_mean) < HBAR_DDIFF
#         and i <= len(pressure)-1
#     ):
#         sigma0_surface_mean = (
#             density_anomaly_profile[:i+1].mean()
#         )
#         i += 1
#     mld = pressure[i]
#     if mld <= MAX_DEPTH:
#         return mld
#     return MAX_DEPTH


def get_monthly_mld(
    ds: xr.Dataset,
    month: int | None = None
) -> xr.Dataset:
    """
    Get the monthly mixed layer depth (MLD) from the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing temperature and pressure data.
    month : int | None, optional
        The month to select (1-12). If None, use the entire dataset.

    Returns
    -------
    xr.Dataset
        The dataset with an added variable 'MLD' representing the mixed layer depth.
    """

    if month is not None:
        ds = ds.isel(TIME=month)
    sa = gsw.SA_from_SP(
        SP=ds.SALINITY, p=ds.PRESSURE, lon=ds.LONGITUDE, lat=ds.LATITUDE
    )
    ct = gsw.CT_from_t(
        SA=sa, t=ds.TEMPERATURE, p=ds.PRESSURE
    )
    sigma0 = gsw.sigma0(SA=sa, CT=ct)
    ds['DENSITY_ANOMALY'] = sigma0
    ds['DENSITY_ANOMALY'].attrs = {"units": "kg/m^3"}

    _find_surface_density_anomaly_mean(ds)
    mld = xr.apply_ufunc(
        _find_half_depth,
        ds['DENSITY_ANOMALY'], ds['PRESSURE'], ds['SURFACE_DENSITY_ANOMALY_MEAN'],
        input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True
    )

    # mld = xr.apply_ufunc(
    #     _find_depth_iteration,
    #     ds['DENSITY_ANOMALY'], ds['PRESSURE'],
    #     input_core_dims=[['PRESSURE'], ['PRESSURE']],
    #     vectorize=True
    # )

    ds.drop_vars(["PRESSURE"])  # don't need pressure anymore
    ds['MLD'] = mld   # save to dataset

    return ds


def save_mld():
    """Save monthly mixed layer depth dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Depth_new-(2004-2018).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2018).nc"
    )
    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2018).nc"
    )
    ds = xr.merge([t, s])

    monthly_datasets = []
    for month in range(0, 180):
        monthly_datasets.append(get_monthly_mld(ds, month))
    h_ds = xr.concat(monthly_datasets, dim="TIME",
                     coords="different", compat='equals')
    h = h_ds['MLD']

    # restore attributes
    h['LATITUDE'].attrs = ds['LATITUDE'].attrs
    h['LONGITUDE'].attrs = ds['LONGITUDE'].attrs
    h.attrs['units'] = 'dbar'
    h.attrs['long_name'] = (
        'Mixed Layer Depth (h)'
    )
    h.name = 'MLD'

    save_file(
        h,
        "datasets/Mixed_Layer_Depth_new-(2004-2018).nc"
    )


def save_monthly_mean_mld():
    """Save the monthly mean of the mixed layer depth dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Depth_new-Seasonal_Mean.nc" in logs_datasets.read():
            return

    h = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth_new-(2004-2018).nc",
    )['MLD']
    h_monthly_mean = get_monthly_mean(h)

    save_file(
        h_monthly_mean,
        "datasets/Mixed_Layer_Depth_new-Seasonal_Mean.nc"
    )


def _find_half_quantity(depth, quantity, mld):
    """
    Find the quantity at the mixed layer depth (hbar) from a temperature or salinity profile.
    -new method: not interpolating, but return the value immediately below MLD.

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
    return quantity[below_mld_index]


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
        if "datasets/Sub_Layer_Temperature_new-(2004-2018).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset('datasets/Temperature-(2004-2018).nc')['TEMPERATURE']
    h = load_and_prepare_dataset('datasets/Mixed_Layer_Depth-(2004-2018).nc')['MLD']

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
        'Monthly Sub Layer Temperature Jan 2004 - Dec 2018 (15.0 year)'
    )
    t_sub.name = 'SUB_TEMPERATURE'

    save_file(
        t_sub,
        'datasets/Sub_Layer_Temperature_new-(2004-2018).nc'
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
        if "datasets/Sub_Layer_Salinity_new-(2004-2018).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset('datasets/Salinity-(2004-2018).nc')['SALINITY']
    h = load_and_prepare_dataset('datasets/Mixed_Layer_Depth-(2004-2018).nc')['MLD']

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
        'Monthly Sub Layer Salinity Jan 2004 - Dec 2018 (15.0 year)'
    )
    s_sub.name = 'SUB_SALINITY'

    save_file(
        s_sub,
        'datasets/Sub_Layer_Salinity_new-(2004-2018).nc'
    )


def save_monthly_mean_sub_temperature():
    """Save the monthly mean sub-layer temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Temperature_new-Seasonal_Mean.nc" in logs_datasets.read():
            return

    t_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Temperature_new-(2004-2018).nc'
    )['SUB_TEMPERATURE']

    t_sub_monthly_mean = get_monthly_mean(t_sub)

    save_file(
        t_sub_monthly_mean,
        'datasets/Sub_Layer_Temperature_new-Seasonal_Mean.nc'
    )


def save_monthly_mean_sub_salinity():
    """Save the monthly mean sub-layer salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Salinity_new-Seasonal_Mean.nc" in logs_datasets.read():
            return

    s_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Salinity_new-(2004-2018).nc'
    )['SUB_SALINITY']

    s_sub_monthly_mean = get_monthly_mean(s_sub)

    save_file(
        s_sub_monthly_mean,
        'datasets/Sub_Layer_Salinity_new-Seasonal_Mean.nc'
    )


def save_sub_temperature_anomalies():
    """Save the sub-layer temperature anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Temperature_Anomalies_new-(2004-2018).nc" in logs_datasets.read():
            return

    t_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Temperature_new-(2004-2018).nc'
    )['SUB_TEMPERATURE']

    t_sub_monthly_mean = load_and_prepare_dataset(
        'datasets/Sub_Layer_Temperature_new-Seasonal_Mean.nc'
    )['MONTHLY_MEAN_SUB_TEMPERATURE']

    t_sub_a = get_anomaly(t_sub, t_sub_monthly_mean)

    save_file(
        t_sub_a,
        'datasets/Sub_Layer_Temperature_Anomalies_new-(2004-2018).nc'
    )


def save_sub_salinity_anomalies():
    """Save the sub-layer salinity anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Salinity_Anomalies_new-(2004-2018).nc" in logs_datasets.read():
            return

    s_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Salinity_new-(2004-2018).nc'
    )['SUB_SALINITY']

    s_sub_monthly_mean = load_and_prepare_dataset(
        'datasets/Sub_Layer_Salinity_new-Seasonal_Mean.nc'
    )['MONTHLY_MEAN_SUB_SALINITY']

    s_sub_a = get_anomaly(s_sub, s_sub_monthly_mean)

    save_file(
        s_sub_a,
        'datasets/Sub_Layer_Salinity_Anomalies_new-(2004-2018).nc'
    )


def main():
    """Main function to prepare datasets from RGARGO."""

    save_mld()
    save_monthly_mean_mld()

    save_sub_temperature()
    save_monthly_mean_sub_temperature()
    save_sub_temperature_anomalies()
    save_sub_salinity()
    save_monthly_mean_sub_salinity()
    save_sub_salinity_anomalies()


if __name__ == "__main__":
    main()
