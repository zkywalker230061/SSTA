"""
Prepare RGARGO dataset.

Chengyun Zhu
2026-1-3
"""

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean  # , get_anomaly
from utilities import save_file


def save_monthly_mean_anomalies():
    """Save the monthly mean of the anomaly dataset."""

    ds_temp = load_and_prepare_dataset(
        "datasets/RG_ArgoClim_Temperature_2019.nc",
    )
    ta = ds_temp['ARGO_TEMPERATURE_ANOMALY']
    ta_monthly_mean = get_monthly_mean(ta)
    save_file(ta_monthly_mean, "datasets/Temperature_Anomaly-Seasonal_Cycle_Mean.nc")

    ds_salt = load_and_prepare_dataset(
        "datasets/RG_ArgoClim_Salinity_2019.nc",
    )
    sa = ds_salt['ARGO_SALINITY_ANOMALY']
    sa_monthly_mean = get_monthly_mean(sa)
    save_file(sa_monthly_mean, "datasets/Salinity_Anomaly-Seasonal_Cycle_Mean.nc")


def save_monthly_mean_temperature():
    """Save the monthly mean temperature dataset."""

    t_15years_mean = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )['ARGO_TEMPERATURE_MEAN']
    ta_monthly_mean = load_and_prepare_dataset(
        "datasets/Temperature_Anomaly-Seasonal_Cycle_Mean.nc",
    )['MONTHLY_MEAN_ARGO_TEMPERATURE_ANOMALY']

    t_15years_mean = t_15years_mean.expand_dims(MONTH=12)
    t_monthly_mean = t_15years_mean + ta_monthly_mean

    t_monthly_mean.attrs['units'] = ta_monthly_mean.attrs.get('units')
    t_monthly_mean.attrs['long_name'] = (
        "Seasonal Cycle Mean of Temperature Jan 2004 - Dec 2018 (15.0 year)"
    )
    t_monthly_mean.name = "MONTHLY_MEAN_TEMPERATURE"

    save_file(
        t_monthly_mean,
        "datasets/Temperature-Seasonal_Cycle_Mean.nc"
    )


def save_monthly_mean_salinity():
    """Save the monthly mean salinity dataset."""

    s_15years_mean = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Salinity_2019.nc",
    )['ARGO_SALINITY_MEAN']
    sa_monthly_mean = load_and_prepare_dataset(
        "../datasets/Salinity_Anomaly-Seasonal_Cycle_Mean.nc",
    )['MONTHLY_MEAN_ARGO_SALINITY_ANOMALY']

    s_15years_mean = s_15years_mean.expand_dims(MONTH=12)
    s_monthly_mean = s_15years_mean + sa_monthly_mean

    s_monthly_mean.attrs['units'] = sa_monthly_mean.attrs.get('units')
    s_monthly_mean.attrs['long_name'] = (
        "Seasonal Cycle Mean of Salinity Jan 2004 - Dec 2018 (15.0 year)"
    )
    s_monthly_mean.name = "MONTHLY_MEAN_SALINITY"

    save_file(
        s_monthly_mean,
        "datasets/Salinity-Seasonal_Cycle_Mean.nc"
    )


def save_temperature():
    """Save temperature dataset."""

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


def main():
    """Main function to prepare datasets from RGARGO."""

    save_monthly_mean_anomalies()
    save_monthly_mean_temperature()
    save_monthly_mean_salinity()
    save_temperature()
    save_salinity()


if __name__ == "__main__":
    main()
