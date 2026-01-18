"""
Calculate Entrainment term.

Chengyun Zhu
2026-1-13
"""

import xarray as xr
import numpy as np

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


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


def save_sub_temperature():
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


def save_sub_salinity():
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


def save_monthly_mean_sub_temperature():
    """Save the monthly mean sub-layer temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Temperature-Seasonal_Mean.nc" in logs_datasets.read():
            return

    t_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Temperature-(2004-2018).nc'
    )['SUB_TEMPERATURE']

    t_sub_monthly_mean = get_monthly_mean(t_sub)

    save_file(
        t_sub_monthly_mean,
        'datasets/Sub_Layer_Temperature-Seasonal_Mean.nc'
    )


def save_monthly_mean_sub_salinity():
    """Save the monthly mean sub-layer salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Salinity-Seasonal_Mean.nc" in logs_datasets.read():
            return

    s_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Salinity-(2004-2018).nc'
    )['SUB_SALINITY']

    s_sub_monthly_mean = get_monthly_mean(s_sub)

    save_file(
        s_sub_monthly_mean,
        'datasets/Sub_Layer_Salinity-Seasonal_Mean.nc'
    )


def save_sub_temperature_anomalies():
    """Save the sub-layer temperature anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Temperature_Anomaly-(2004-2018).nc" in logs_datasets.read():
            return

    t_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Temperature-(2004-2018).nc'
    )['SUB_TEMPERATURE']

    t_sub_monthly_mean = load_and_prepare_dataset(
        'datasets/Sub_Layer_Temperature-Seasonal_Mean.nc'
    )['MONTHLY_MEAN_SUB_TEMPERATURE']

    t_sub_anomaly = get_anomaly(t_sub, t_sub_monthly_mean)

    save_file(
        t_sub_anomaly,
        'datasets/Sub_Layer_Temperature_Anomaly-(2004-2018).nc'
    )


def save_sub_salinity_anomalies():
    """Save the sub-layer salinity anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sub_Layer_Salinity_Anomaly-(2004-2018).nc" in logs_datasets.read():
            return

    s_sub = load_and_prepare_dataset(
        'datasets/Sub_Layer_Salinity-(2004-2018).nc'
    )['SUB_SALINITY']

    s_sub_monthly_mean = load_and_prepare_dataset(
        'datasets/Sub_Layer_Salinity-Seasonal_Mean.nc'
    )['MONTHLY_MEAN_SUB_SALINITY']

    s_sub_anomaly = get_anomaly(s_sub, s_sub_monthly_mean)

    save_file(
        s_sub_anomaly,
        'datasets/Sub_Layer_Salinity_Anomaly-(2004-2018).nc'
    )


def save_entrainment_velocity():
    """Save the entrainment velocity (w_e) dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Entrainment_Velocity-(2004-2018).nc" in logs_datasets.read():
            return

    h = load_and_prepare_dataset(
        'datasets/Mixed_Layer_Depth-(2004-2018).nc'
    )['MLD']

    # Suppress warnings for NaN and -inf values in gradient computation
    with np.errstate(invalid='ignore'):
        w_e = (
            np.gradient(h, axis=h.get_axis_num('TIME'))
            / SECONDS_MONTH  # dbar/s
        )
    # NaN and -inf to NaN
    w_e = np.where(np.isnan(h), np.nan, w_e)
    w_e = np.where(h == -np.inf, np.nan, w_e)
    # set negative entrainment velocity to zero
    w_e = np.where(w_e < 0, 0, w_e)

    w_e_da = xr.DataArray(
        w_e,
        coords=h.coords,
        dims=h.dims,
        name='w_e'
    )
    w_e_da.attrs['units'] = 'dbar/s'
    w_e_da.attrs['long_name'] = (
        'Monthly Entrainment Velocity Jan 2004 - Dec 2018 (15.0 year)'
    )

    save_file(
        w_e_da,
        'datasets/Mixed_Layer_Entrainment_Velocity-(2004-2018).nc'
    )


def save_monthly_mean_entrainment_velocity():
    """Save the monthly mean entrainment velocity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc" in logs_datasets.read():
            return

    w_e = load_and_prepare_dataset(
        'datasets/Mixed_Layer_Entrainment_Velocity-(2004-2018).nc'
    )['w_e']

    w_e_monthly_mean = get_monthly_mean(w_e)

    save_file(
        w_e_monthly_mean,
        'datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc'
    )


def save_entrainment_anomaly_temperature():
    """Save the Q_Entrainment dataset for temperature."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Entrainment_Heat_Flux-(2004-2018).nc" in logs_datasets.read():
            return

    t_sub_a = load_and_prepare_dataset(
        "datasets/Sub_Layer_Temperature_Anomaly-(2004-2018).nc"
    )['ANOMALY_SUB_TEMPERATURE']
    t_m_a = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Temperature_Anomaly-(2004-2018).nc"
    )['ANOMALY_ML_TEMPERATURE']
    w_e_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_w_e']

    w_e_monthly_mean = xr.concat([w_e_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    w_e_monthly_mean = w_e_monthly_mean.rename({'MONTH': 'TIME'})
    w_e_monthly_mean['TIME'] = t_m_a.TIME

    q_entrainment_prime = RHO_O * C_O * w_e_monthly_mean * (t_sub_a - t_m_a)

    q_entrainment_prime = q_entrainment_prime.drop_vars('MONTH')

    q_entrainment_prime.attrs['units'] = 'W/m^2'
    q_entrainment_prime.attrs['long_name'] = (
        'Monthly Q_Entrainment Anomaly Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_entrainment_prime.name = 'ANOMALY_ENTRAINMENT_HEAT_FLUX'

    save_file(
        q_entrainment_prime,
        "datasets/Simulation-Entrainment_Heat_Flux-(2004-2018).nc"
    )


def save_entrainment_anomaly_salinity():
    """Save the Q_Entrainment dataset for salinity."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Entrainment_Water_Flux-(2004-2018).nc" in logs_datasets.read():
            return

    s_sub_a = load_and_prepare_dataset(
        "datasets/Sub_Layer_Salinity_Anomaly-(2004-2018).nc"
    )['ANOMALY_SUB_SALINITY']
    s_m_a = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Salinity_Anomaly-(2004-2018).nc"
    )['ANOMALY_ML_SALINITY']
    w_e_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_w_e']

    w_e_monthly_mean = xr.concat([w_e_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    w_e_monthly_mean = w_e_monthly_mean.rename({'MONTH': 'TIME'})
    w_e_monthly_mean['TIME'] = s_m_a.TIME

    q_entrainment_salt_prime = RHO_O * w_e_monthly_mean * (s_sub_a - s_m_a)

    q_entrainment_salt_prime = q_entrainment_salt_prime.drop_vars('MONTH')

    q_entrainment_salt_prime.attrs['units'] = 'kg/m^2/s'
    q_entrainment_salt_prime.attrs['long_name'] = (
        'Monthly Q_Entrainment Water Flux Anomaly Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_entrainment_salt_prime.name = 'ANOMALY_ENTRAINMENT_WATER_FLUX'

    save_file(
        q_entrainment_salt_prime,
        "datasets/Simulation-Entrainment_Water_Flux-(2004-2018).nc"
    )


def main():
    """Main function to calcuate entrainment term."""

    save_sub_temperature()
    save_sub_salinity()
    save_monthly_mean_sub_temperature()
    save_monthly_mean_sub_salinity()
    save_sub_temperature_anomalies()
    save_sub_salinity_anomalies()

    save_entrainment_velocity()
    save_monthly_mean_entrainment_velocity()

    save_entrainment_anomaly_temperature()
    save_entrainment_anomaly_salinity()


if __name__ == "__main__":
    main()
