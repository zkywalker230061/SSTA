"""
Calculate Ekman term.

Chengyun Zhu
2026-1-18
"""

import xarray as xr
import numpy as np

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


C_O = 4100  # J / (kg K)
R = 6.371e6  # m


def save_monthly_mean_turbulent_surface_stress():
    """Calculate and save monthly mean turbulent surface stress dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Turbulent_Surface_Stress-Seasonal_Mean.nc" in logs_datasets.read():
            return

    tao_ds = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress-(2004-2018).nc"
    )

    ew = tao_ds['avg_iews']
    ns = tao_ds['avg_inss']

    ew_monthly_mean = get_monthly_mean(ew)
    ns_monthly_mean = get_monthly_mean(ns)

    tao_monthly_mean_ds = ew_monthly_mean.to_dataset(name='MONTHLY_MEAN_avg_iews')
    tao_monthly_mean_ds['MONTHLY_MEAN_avg_inss'] = ns_monthly_mean

    save_file(
        tao_monthly_mean_ds,
        "datasets/Turbulent_Surface_Stress-Seasonal_Mean.nc"
    )


def save_turbulent_surface_stress_anomalies():
    """Calculate and save turbulent surface stress anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Turbulent_Surface_Stress_Anomalies-(2004-2018).nc" in logs_datasets.read():
            return

    tao_ds = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress-(2004-2018).nc"
    )

    tao_monthly_mean_ds = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress-Seasonal_Mean.nc"
    )

    ew_a = get_anomaly(tao_ds['avg_iews'],
                       tao_monthly_mean_ds['MONTHLY_MEAN_avg_iews'])
    ns_a = get_anomaly(tao_ds['avg_inss'],
                       tao_monthly_mean_ds['MONTHLY_MEAN_avg_inss'])

    tao_a_ds = ew_a.to_dataset(name='ANOMALY_avg_iews')
    tao_a_ds['ANOMALY_avg_inss'] = ns_a

    save_file(
        tao_a_ds,
        "datasets/Turbulent_Surface_Stress_Anomalies-(2004-2018).nc"
    )


def save_ekman_anomaly_temperature():
    """Calculate and save Ekman anomaly temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Ekman_Heat_Flux-(2004-2018).nc" in logs_datasets.read():
            return

    t_m_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Temperature-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_ML_TEMPERATURE']
    tao_a_ds = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress_Anomalies-(2004-2018).nc"
    )

    t_m_monthly_mean = xr.concat([t_m_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    t_m_monthly_mean = t_m_monthly_mean.rename({'MONTH': 'TIME'})
    t_m_monthly_mean['TIME'] = tao_a_ds.TIME

    dt_m_monthly_mean_dtheta = (
        np.gradient(t_m_monthly_mean, axis=t_m_monthly_mean.get_axis_num('LONGITUDE'))
    )
    dt_m_monthly_mean_dtheta_da = xr.DataArray(
        dt_m_monthly_mean_dtheta,
        coords=t_m_monthly_mean.coords,
        dims=t_m_monthly_mean.dims,
    )
    dtheta_dx = 1 / (R * np.sin(np.deg2rad(t_m_monthly_mean['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=t_m_monthly_mean['LONGITUDE'])
    dt_m_monthly_mean_dx_da = dt_m_monthly_mean_dtheta_da * dtheta_dx
    dt_m_monthly_mean_dx_da.name = 'dt_m_monthly_mean_dx'

    dt_m_monthly_mean_dphi = (
        np.gradient(t_m_monthly_mean, axis=t_m_monthly_mean.get_axis_num('LATITUDE'))
    )
    dt_m_monthly_mean_dphi_da = xr.DataArray(
        dt_m_monthly_mean_dphi,
        coords=t_m_monthly_mean.coords,
        dims=t_m_monthly_mean.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    dt_m_monthly_mean_dy_da = dt_m_monthly_mean_dphi_da * dphi_dy
    dt_m_monthly_mean_dy_da.name = 'dt_m_monthly_mean_dy'

    f = 2 * (2*np.pi/(24*60*60)) * np.sin(
        np.deg2rad(t_m_monthly_mean['LATITUDE'])
    )
    f = f.expand_dims(LONGITUDE=t_m_monthly_mean['LONGITUDE'])
    f = f.where(
        (t_m_monthly_mean['LATITUDE'] > 5) | (t_m_monthly_mean['LATITUDE'] < -5)
    )

    q_ekman_a = C_O / f * (
        tao_a_ds['ANOMALY_avg_iews'] * dt_m_monthly_mean_dy_da -
        tao_a_ds['ANOMALY_avg_inss'] * dt_m_monthly_mean_dx_da
    )

    q_ekman_a = q_ekman_a.drop_vars('MONTH')

    q_ekman_a.attrs['units'] = 'W/m^2'
    q_ekman_a.attrs['long_name'] = (
        'Monthly Q_Ekman Anomaly Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_ekman_a.name = 'ANOMALY_EKMAN_HEAT_FLUX'

    save_file(
        q_ekman_a,
        "datasets/Simulation-Ekman_Heat_Flux-(2004-2018).nc"
    )


def save_ekman_anomaly_salinity():
    """Calculate and save Ekman anomaly salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Ekman_Water_Rate-(2004-2018).nc" in logs_datasets.read():
            return

    s_m_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Salinity-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_ML_SALINITY']
    tao_a_ds = load_and_prepare_dataset(
        "datasets/Turbulent_Surface_Stress_Anomalies-(2004-2018).nc"
    )

    s_m_monthly_mean = xr.concat([s_m_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    s_m_monthly_mean = s_m_monthly_mean.rename({'MONTH': 'TIME'})
    s_m_monthly_mean['TIME'] = tao_a_ds.TIME

    dt_m_monthly_mean_dtheta = (
        np.gradient(s_m_monthly_mean, axis=s_m_monthly_mean.get_axis_num('LONGITUDE'))
    )
    dt_m_monthly_mean_dtheta_da = xr.DataArray(
        dt_m_monthly_mean_dtheta,
        coords=s_m_monthly_mean.coords,
        dims=s_m_monthly_mean.dims,
    )
    dtheta_dx = 1 / (R * np.sin(np.deg2rad(s_m_monthly_mean['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=s_m_monthly_mean['LONGITUDE'])
    dt_m_monthly_mean_dx_da = dt_m_monthly_mean_dtheta_da * dtheta_dx
    dt_m_monthly_mean_dx_da.name = 'dt_m_monthly_mean_dx'

    dt_m_monthly_mean_dphi = (
        np.gradient(s_m_monthly_mean, axis=s_m_monthly_mean.get_axis_num('LATITUDE'))
    )
    dt_m_monthly_mean_dphi_da = xr.DataArray(
        dt_m_monthly_mean_dphi,
        coords=s_m_monthly_mean.coords,
        dims=s_m_monthly_mean.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    dt_m_monthly_mean_dy_da = dt_m_monthly_mean_dphi_da * dphi_dy
    dt_m_monthly_mean_dy_da.name = 'dt_m_monthly_mean_dy'

    f = 2 * (2*np.pi/(24*60*60)) * np.sin(
        np.deg2rad(s_m_monthly_mean['LATITUDE'])
    )
    f = f.expand_dims(LONGITUDE=s_m_monthly_mean['LONGITUDE'])
    f = f.where(
        (s_m_monthly_mean['LATITUDE'] > 5) | (s_m_monthly_mean['LATITUDE'] < -5)
    )

    q_ekman_a = 1 / f * (
        tao_a_ds['ANOMALY_avg_iews'] * dt_m_monthly_mean_dy_da -
        tao_a_ds['ANOMALY_avg_inss'] * dt_m_monthly_mean_dx_da
    )

    q_ekman_a = q_ekman_a.drop_vars('MONTH')

    q_ekman_a.attrs['units'] = 'kg/m^2/s'
    q_ekman_a.attrs['long_name'] = (
        'Monthly Q_Ekman Anomaly Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_ekman_a.name = 'ANOMALY_EKMAN_WATER_RATE'

    save_file(
        q_ekman_a,
        "datasets/Simulation-Ekman_Water_Rate-(2004-2018).nc"
    )


def main():
    """Main function to calcuate Ekman term."""

    save_monthly_mean_turbulent_surface_stress()
    save_turbulent_surface_stress_anomalies()

    save_ekman_anomaly_temperature()
    save_ekman_anomaly_salinity()


if __name__ == "__main__":
    main()
