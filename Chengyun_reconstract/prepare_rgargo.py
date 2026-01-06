"""
Prepare RGARGO dataset.

Chengyun Zhu
2026-1-3
"""

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
    )
    t_monthly_mean = get_monthly_mean(t['TEMPERATURE'])
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
    )
    s_monthly_mean = get_monthly_mean(s['SALINITY'])
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
    )
    t_monthly_mean = load_and_prepare_dataset(
        "datasets/Temperature-Seasonal_Cycle_Mean.nc",
    )
    ta = get_anomaly(
        t['TEMPERATURE'],
        t_monthly_mean['MONTHLY_MEAN_TEMPERATURE']
    )
    save_file(ta, "datasets/Temperature_Anomaly-(2004-2018).nc")


def save_salinity_anomalies():
    """Save salinity anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Salinity_Anomaly-(2004-2018).nc" in logs_datasets.read():
            return

    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2018).nc",
    )
    s_monthly_mean = load_and_prepare_dataset(
        "datasets/Salinity-Seasonal_Cycle_Mean.nc",
    )
    sa = get_anomaly(
        s['SALINITY'],
        s_monthly_mean['MONTHLY_MEAN_SALINITY']
    )
    save_file(sa, "datasets/Salinity_Anomaly-(2004-2018).nc")


def save_monthly_mean_temperature_anomalies():
    """Save monthly mean temperature anomaly dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Temperature_Anomaly-Seasonal_Cycle_Mean.nc" in logs_datasets.read():
            return

    ta = load_and_prepare_dataset(
        "datasets/Temperature_Anomaly-(2004-2018).nc",
    )
    ta_monthly_mean = get_monthly_mean(ta['ANOMALY_TEMPERATURE'])
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
    )
    sa_monthly_mean = get_monthly_mean(sa['ANOMALY_SALINITY'])
    save_file(
        sa_monthly_mean,
        "datasets/Salinity_Anomaly-Seasonal_Cycle_Mean.nc"
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


if __name__ == "__main__":
    main()
