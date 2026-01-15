"""
Prepare RGARGO dataset.

Chengyun Zhu
2026-1-3
"""

import xarray as xr
import numpy as np

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


def save_temperature():
    """Save temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature-(2004-2018).nc" in logs_datasets.read():
            return

    t_15years_mean = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )['ARGO_TEMPERATURE_MEAN']
    ta = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )['ARGO_TEMPERATURE_ANOMALY']

    t_15years_mean = t_15years_mean.expand_dims(TIME=180)
    t = t_15years_mean + ta

    t.attrs['units'] = ta.attrs.get('units')
    t.attrs['long_name'] = (
        "Monthly Temperature Jan 2004 - Dec 2018 (15.0 year)"
    )
    t.name = "TEMPERATURE"

    save_file(
        t,
        "datasets/Temperature-(2004-2018).nc"
    )


def save_salinity():
    """Save salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity-(2004-2018).nc" in logs_datasets.read():
            return

    s_15years_mean = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Salinity_2019.nc",
    )['ARGO_SALINITY_MEAN']
    sa = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Salinity_2019.nc",
    )['ARGO_SALINITY_ANOMALY']

    s_15years_mean = s_15years_mean.expand_dims(TIME=180)
    s = s_15years_mean + sa

    s.attrs['units'] = sa.attrs.get('units')
    s.attrs['long_name'] = (
        "Monthly Salinity Jan 2004 - Dec 2018 (15.0 year)"
    )
    s.name = "SALINITY"

    save_file(
        s,
        "datasets/Salinity-(2004-2018).nc"
    )


def save_monthly_mean_temperature():
    """Save the monthly mean temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature-Seasonal_Cycle_Mean.nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2018).nc",
    )['TEMPERATURE']
    t_monthly_mean = get_monthly_mean(t)
    save_file(
        t_monthly_mean,
        "datasets/Temperature-Seasonal_Cycle_Mean.nc"
    )


def save_monthly_mean_salinity():
    """Save the monthly mean salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity-Seasonal_Cycle_Mean.nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2018).nc",
    )['SALINITY']
    s_monthly_mean = get_monthly_mean(s)
    save_file(
        s_monthly_mean,
        "datasets/Salinity-Seasonal_Cycle_Mean.nc"
    )


def save_temperature_anomalies():
    """Save temperature anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature_Anomaly-(2004-2018).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2018).nc",
    )['TEMPERATURE']
    t_monthly_mean = load_and_prepare_dataset(
        "datasets/Temperature-Seasonal_Cycle_Mean.nc",
    )['MONTHLY_MEAN_TEMPERATURE']
    ta = get_anomaly(
        t,
        t_monthly_mean
    )
    save_file(ta, "datasets/Temperature_Anomaly-(2004-2018).nc")


def save_salinity_anomalies():
    """Save salinity anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity_Anomaly-(2004-2018).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2018).nc",
    )['SALINITY']
    s_monthly_mean = load_and_prepare_dataset(
        "datasets/Salinity-Seasonal_Cycle_Mean.nc",
    )['MONTHLY_MEAN_SALINITY']
    sa = get_anomaly(
        s,
        s_monthly_mean
    )
    save_file(sa, "datasets/Salinity_Anomaly-(2004-2018).nc")


def save_monthly_mean_temperature_anomalies():
    """Save monthly mean temperature anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature_Anomaly-Seasonal_Cycle_Mean.nc" in logs_datasets.read():
            return

    ta = load_and_prepare_dataset(
        "datasets/Temperature_Anomaly-(2004-2018).nc",
    )['ANOMALY_TEMPERATURE']
    ta_monthly_mean = get_monthly_mean(ta)
    save_file(
        ta_monthly_mean,
        "datasets/Temperature_Anomaly-Seasonal_Cycle_Mean.nc"
    )


def save_monthly_mean_salinity_anomalies():
    """Save monthly mean salinity anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity_Anomaly-Seasonal_Cycle_Mean.nc" in logs_datasets.read():
            return

    sa = load_and_prepare_dataset(
        "datasets/Salinity_Anomaly-(2004-2018).nc",
    )['ANOMALY_SALINITY']
    sa_monthly_mean = get_monthly_mean(sa)
    save_file(
        sa_monthly_mean,
        "datasets/Salinity_Anomaly-Seasonal_Cycle_Mean.nc"
    )


def _trapezoid_mean(quantity, depth, h_single):
    """
    Integrate temperature profile up to depth h_single.

    Parameters
    ----------
    quantity: np.ndarray
        Temperature profile array.
    depth: np.ndarray
        Depth profile array.
    h_single: float
        Mixed layer depth.

    Returns
    -------
    float
        Integrated temperature up to h_single.
    """

    mask_below = depth < h_single

    if not np.any(mask_below) or np.isnan(h_single):
        return np.nan

    valid = ~np.isnan(quantity)
    depth_valid = depth[valid]
    quantity_valid = quantity[valid]

    if len(depth_valid) < 2:
        return np.nan

    q_at_h = np.interp(h_single, depth_valid, quantity_valid)

    mask_below_valid = depth_valid < h_single
    p_below = depth_valid[mask_below_valid]
    q_below = quantity_valid[mask_below_valid]
    if len(p_below) < 1:
        return np.nan

    p_final = np.append(p_below, h_single)
    q_final = np.append(q_below, q_at_h)

    q_m = np.trapezoid(q_final, p_final) / h_single

    return q_m


def save_mixed_layer_temperature():
    """Save mixed layer temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Temperature-(2004-2018).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2018).nc",
    )['TEMPERATURE']
    h = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth-(2004-2018).nc",
    )['MLD']

    t_m_list = []
    for month in t.coords['TIME']:
        t_value = t.sel(TIME=month)
        h_value = h.sel(TIME=month)

        t_m_value = xr.apply_ufunc(
            _trapezoid_mean,
            t_value,
            t_value['PRESSURE'],
            h_value,
            input_core_dims=[['PRESSURE'], ['PRESSURE'], []],
            vectorize=True,
        )
        t_m_list.append(t_m_value)

    t_m = xr.concat(t_m_list, dim="TIME",
                    coords="different", compat='equals')

    t_m.attrs['units'] = t.attrs.get('units')
    t_m.attrs['long_name'] = 'Monthly Mixed Layer Temperature Jan 2004 - Dec 2018 (15.0 year)'
    t_m.name = "ML_TEMPERATURE"

    save_file(
        t_m,
        "datasets/Mixed_Layer_Temperature-(2004-2018).nc"
    )


def save_mixed_layer_salinity():
    """Save mixed layer salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Salinity-(2004-2018).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2018).nc",
    )['SALINITY']
    h = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth-(2004-2018).nc",
    )['MLD']

    s_m_list = []
    for month in s.coords['TIME']:
        s_value = s.sel(TIME=month)
        h_value = h.sel(TIME=month)

        s_m_value = xr.apply_ufunc(
            _trapezoid_mean,
            s_value,
            s_value['PRESSURE'],
            h_value,
            input_core_dims=[['PRESSURE'], ['PRESSURE'], []],
            vectorize=True,
        )
        s_m_list.append(s_m_value)

    s_m = xr.concat(s_m_list, dim="TIME",
                    coords="different", compat='equals')

    s_m.attrs['units'] = s.attrs.get('units')
    s_m.attrs['long_name'] = 'Monthly Mixed Layer Salinity Jan 2004 - Dec 2018 (15.0 year)'
    s_m.name = "ML_SALINITY"

    save_file(
        s_m,
        "datasets/Mixed_Layer_Salinity-(2004-2018).nc"
    )


def main():
    """Main function to prepare datasets from RGARGO."""

    save_temperature()
    save_salinity()
    save_monthly_mean_temperature()
    save_monthly_mean_salinity()
    save_temperature_anomalies()
    save_salinity_anomalies()
    save_monthly_mean_temperature_anomalies()
    save_monthly_mean_salinity_anomalies()
    save_mixed_layer_temperature()
    save_mixed_layer_salinity()


if __name__ == "__main__":
    main()
