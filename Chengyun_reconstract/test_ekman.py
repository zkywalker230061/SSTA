"""
Calculate Ekman term - test.

Chengyun Zhu
2026-2-5
"""

import xarray as xr
import numpy as np

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


C_O = 4100  # J / (kg K)
R = 6.371e6  # m


def save_ekman_anomaly_temperature():
    """Calculate and save Ekman anomaly temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/.test-Simulation-Ekman_Heat_Flux-(2004-2018).nc" in logs_datasets.read():
            return

    t_m = load_and_prepare_dataset(
        "datasets/.test-ml.nc"
    )['MIXED_LAYER_TEMP']
    t_m_monthly_mean = get_monthly_mean(t_m)

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
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(t_m_monthly_mean['LATITUDE']))) / (np.pi/180)
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
        "datasets/.test-Simulation-Ekman_Heat_Flux-(2004-2018).nc"
    )


def save_ekman_anomaly_salinity():
    """Calculate and save Ekman anomaly salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/.test-Simulation-Ekman_Water_Rate-(2004-2018).nc" in logs_datasets.read():
            return

    s_m = load_and_prepare_dataset(
        "datasets/.test-ml.nc"
    )['MIXED_LAYER_SALINITY']
    s_m_monthly_mean = get_monthly_mean(s_m)

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
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(s_m_monthly_mean['LATITUDE']))) / (np.pi/180)
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
        "datasets/.test-Simulation-Ekman_Water_Rate-(2004-2018).nc"
    )


def main():
    """Main function to calcuate Ekman term."""

    save_ekman_anomaly_temperature()
    save_ekman_anomaly_salinity()


if __name__ == "__main__":
    main()
