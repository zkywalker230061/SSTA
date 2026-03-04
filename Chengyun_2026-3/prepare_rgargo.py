"""
Prepare RGARGO dataset.

Chengyun Zhu
2026-3-3
"""

import xarray as xr
import numpy as np

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file

import calculate_mixedlayerdepth

EXTENTION = [
    '201901', '201902', '201903', '201904', '201905', '201906',
    '201907', '201908', '201909', '201910', '201911', '201912',
    '202001', '202002', '202003', '202004', '202005', '202006',
    '202007', '202008', '202009', '202010', '202011', '202012',
    '202101', '202102', '202103', '202104', '202105', '202106',
    '202107', '202108', '202109', '202110', '202111', '202112',
    '202201', '202202', '202203', '202204', '202205', '202206',
    '202207', '202208', '202209', '202210', '202211', '202212',
    '202301', '202302', '202303', '202304', '202305', '202306',
    '202307', '202308', '202309', '202310', '202311', '202312',
    '202401', '202402', '202403', '202404', '202405', '202406',
    '202407', '202408', '202409', '202410', '202411', '202412',
    '202501', '202502', '202503', '202504', '202505', '202506',
    '202507', '202508', '202509', '202510', '202511', '202512',
]


def save_temperature():
    """Save temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature-(2004-2025).nc" in logs_datasets.read():
            return

    t_15years_mean = load_and_prepare_dataset(
        "datasets/RGARGO/RG_ArgoClim_Temperature_2019.nc",
    )['ARGO_TEMPERATURE_MEAN']
    ta = load_and_prepare_dataset(
        "datasets/RGARGO/RG_ArgoClim_Temperature_2019.nc",
    )['ARGO_TEMPERATURE_ANOMALY']

    for time in EXTENTION:
        ta_new = load_and_prepare_dataset(
            f"datasets/RGARGO/RG_ArgoClim_{time}_2019.nc",
        )['ARGO_TEMPERATURE_ANOMALY']
        ta = xr.concat(
            [ta, ta_new],
            dim="TIME", coords="different", compat='equals'
        )

    t_22years_mean = t_15years_mean.expand_dims(TIME=264)
    t = t_22years_mean + ta

    t.attrs['units'] = ta.attrs.get('units')
    t.attrs['long_name'] = (
        "Temperature Jan 2004 - Dec 2025 (22.0 year)"
    )
    t.name = "TEMPERATURE"

    save_file(
        t,
        "datasets/Temperature-(2004-2025).nc"
    )


def save_salinity():
    """Save salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity-(2004-2025).nc" in logs_datasets.read():
            return

    s_15years_mean = load_and_prepare_dataset(
        "datasets/RGARGO/RG_ArgoClim_Salinity_2019.nc",
    )['ARGO_SALINITY_MEAN']
    sa = load_and_prepare_dataset(
        "datasets/RGARGO/RG_ArgoClim_Salinity_2019.nc",
    )['ARGO_SALINITY_ANOMALY']

    for time in EXTENTION:
        sa_new = load_and_prepare_dataset(
            f"datasets/RGARGO/RG_ArgoClim_{time}_2019.nc",
        )['ARGO_SALINITY_ANOMALY']
        sa = xr.concat(
            [sa, sa_new],
            dim="TIME", coords="different", compat='equals'
        )

    s_22years_mean = s_15years_mean.expand_dims(TIME=264)
    s = s_22years_mean + sa

    s.attrs['units'] = sa.attrs.get('units')
    s.attrs['long_name'] = (
        "Monthly Salinity Jan 2004 - Dec 2025 (22.0 year)"
    )
    s.name = "SALINITY"

    save_file(
        s,
        "datasets/Salinity-(2004-2025).nc"
    )


def save_monthly_mean_temperature():
    """Save the monthly mean temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature-Clim_Mean.nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2025).nc",
    )['TEMPERATURE']
    t_monthly_mean = get_monthly_mean(t)
    save_file(
        t_monthly_mean,
        "datasets/Temperature-Clim_Mean.nc"
    )


def save_monthly_mean_salinity():
    """Save the monthly mean salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity-Clim_Mean.nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2025).nc",
    )['SALINITY']
    s_monthly_mean = get_monthly_mean(s)
    save_file(
        s_monthly_mean,
        "datasets/Salinity-Clim_Mean.nc"
    )


def save_temperature_anomalies():
    """Save temperature anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature_Anomalies-(2004-2025).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2025).nc",
    )['TEMPERATURE']
    t_monthly_mean = load_and_prepare_dataset(
        "datasets/Temperature-Clim_Mean.nc",
    )['MONTHLY_MEAN_TEMPERATURE']
    t_a = get_anomaly(
        t,
        t_monthly_mean
    )
    save_file(t_a, "datasets/Temperature_Anomalies-(2004-2025).nc")


def save_salinity_anomalies():
    """Save salinity anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity_Anomalies-(2004-2025).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2025).nc",
    )['SALINITY']
    s_monthly_mean = load_and_prepare_dataset(
        "datasets/Salinity-Clim_Mean.nc",
    )['MONTHLY_MEAN_SALINITY']
    s_a = get_anomaly(
        s,
        s_monthly_mean
    )
    save_file(s_a, "datasets/Salinity_Anomalies-(2004-2025).nc")


def _trapezoid_mean(quantity, depth, h_single):
    """
    Integrate quantity profile up to depth h_single.

    Parameters
    ----------
    quantity: np.ndarray
        Quantity profile array.
    depth: np.ndarray
        Depth profile array.
    h_single: float
        Mixed layer depth.

    Returns
    -------
    float
        Integrated quantity up to h_single.
    """

    if not np.isfinite(h_single) or h_single <= 0:
        return np.nan

    # clip h to available depth range
    h_eff = min(h_single, depth[-1])

    q_at_h = np.interp(h_eff, depth, quantity)
    q_at_0 = np.interp(0.0, depth, quantity)

    mask_mid = (depth > 0.0) & (depth < h_eff)
    p_final = np.concatenate(([0.0], depth[mask_mid], [h_eff]))
    q_final = np.concatenate(([q_at_0], quantity[mask_mid], [q_at_h]))

    return np.trapezoid(q_final, p_final) / h_eff


def save_mixed_layer_temperature():
    """Save mixed layer temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Temperature-(2004-2025).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2025).nc",
    )['TEMPERATURE']
    h = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth-(2004-2025).nc",
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
    t_m.attrs['long_name'] = 'Monthly Mixed Layer Temperature Jan 2004 - Dec 2025 (22.0 year)'
    t_m.name = "ML_TEMPERATURE"

    save_file(
        t_m,
        "datasets/Mixed_Layer_Temperature-(2004-2025).nc"
    )


def save_mixed_layer_salinity():
    """Save mixed layer salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Salinity-(2004-2025).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2025).nc",
    )['SALINITY']
    h = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth-(2004-2025).nc",
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
    s_m.attrs['long_name'] = 'Monthly Mixed Layer Salinity Jan 2004 - Dec 2025 (22.0 year)'
    s_m.name = "ML_SALINITY"

    save_file(
        s_m,
        "datasets/Mixed_Layer_Salinity-(2004-2025).nc"
    )


def save_monthly_mean_mixed_layer_temperature():
    """Save the monthly mean mixed layer temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Temperature-Clim_Mean.nc" in logs_datasets.read():
            return

    t_m = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Temperature-(2004-2025).nc",
    )['ML_TEMPERATURE']
    t_m_monthly_mean = get_monthly_mean(t_m)
    save_file(
        t_m_monthly_mean,
        "datasets/Mixed_Layer_Temperature-Clim_Mean.nc"
    )


def save_monthly_mean_mixed_layer_salinity():
    """Save the monthly mean mixed layer salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Salinity-Clim_Mean.nc" in logs_datasets.read():
            return

    s_m = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Salinity-(2004-2025).nc",
    )['ML_SALINITY']
    s_m_monthly_mean = get_monthly_mean(s_m)
    save_file(
        s_m_monthly_mean,
        "datasets/Mixed_Layer_Salinity-Clim_Mean.nc"
    )


def save_mixed_layer_temperature_anomalies():
    """Save mixed layer temperature anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2025).nc" in logs_datasets.read():
            return

    t_m = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Temperature-(2004-2025).nc",
    )['ML_TEMPERATURE']
    t_m_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Temperature-Clim_Mean.nc",
    )['MONTHLY_MEAN_ML_TEMPERATURE']
    t_m_a = get_anomaly(
        t_m,
        t_m_monthly_mean
    )
    save_file(
        t_m_a,
        "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2025).nc"
    )


def save_mixed_layer_salinity_anomalies():
    """Save mixed layer salinity anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Salinity_Anomalies-(2004-2025).nc" in logs_datasets.read():
            return

    s_m = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Salinity-(2004-2025).nc",
    )['ML_SALINITY']
    s_m_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Salinity-Clim_Mean.nc",
    )['MONTHLY_MEAN_ML_SALINITY']
    s_m_a = get_anomaly(
        s_m,
        s_m_monthly_mean
    )
    save_file(
        s_m_a,
        "datasets/Mixed_Layer_Salinity_Anomalies-(2004-2025).nc"
    )


def main():
    """Main function to prepare datasets from RGARGO."""

    save_temperature()
    save_salinity()
    save_monthly_mean_temperature()
    save_monthly_mean_salinity()
    save_temperature_anomalies()
    save_salinity_anomalies()

    calculate_mixedlayerdepth.main()

    save_mixed_layer_temperature()
    save_mixed_layer_salinity()
    save_monthly_mean_mixed_layer_temperature()
    save_monthly_mean_mixed_layer_salinity()
    save_mixed_layer_temperature_anomalies()
    save_mixed_layer_salinity_anomalies()


if __name__ == "__main__":
    main()
