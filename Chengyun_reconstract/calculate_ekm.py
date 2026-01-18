"""
Calculate Ekman term.

Chengyun Zhu
2026-1-18
"""

# import xarray as xr
# import numpy as np

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


def save_monthly_mean_turbulent_surface_stress():
    """Calculate and save monthly mean turbulent surface stress dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Turbulent_Surface_Stress-Seasonal_Mean.nc" in logs_datasets.read():
            return

    tao = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress-(2004-2018).nc"
    )

    ew = tao['avg_iews']
    ns = tao['avg_inss']

    ew_monthly_mean = get_monthly_mean(ew)
    ns_monthly_mean = get_monthly_mean(ns)

    tao_monthly_mean = ew_monthly_mean.to_dataset(name='MONTHLY_MEAN_avg_iews')
    tao_monthly_mean['MONTHLY_MEAN_avg_inss'] = ns_monthly_mean

    save_file(
        tao_monthly_mean,
        "datasets/Turbulent_Surface_Stress-Seasonal_Mean.nc"
    )


def save_turbulent_surface_stress_anomalies():
    """Calculate and save turbulent surface stress anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Turbulent_Surface_Stress_Anomalies-(2004-2018).nc" in logs_datasets.read():
            return

    tao = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress-(2004-2018).nc"
    )

    tao_monthly_mean = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress-Seasonal_Mean.nc"
    )

    ew_a = get_anomaly(tao['avg_iews'], tao_monthly_mean['MONTHLY_MEAN_avg_iews'])
    ns_a = get_anomaly(tao['avg_inss'], tao_monthly_mean['MONTHLY_MEAN_avg_inss'])

    tao_anomalies = ew_a.to_dataset(name='ANOMALY_avg_iews')
    tao_anomalies['ANOMALY_avg_inss'] = ns_a

    save_file(
        tao_anomalies,
        "datasets/Turbulent_Surface_Stress_Anomalies-(2004-2018).nc"
    )


def main():
    """Main function to calcuate Ekman term."""

    save_monthly_mean_turbulent_surface_stress()
    save_turbulent_surface_stress_anomalies()


if __name__ == "__main__":
    main()
