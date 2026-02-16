"""
Calculate Geostrophic term.

Chengyun Zhu
2026-1-12
"""

import xarray as xr
import numpy as np
import gsw

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file

REF_PRESSURE = 950  # dbar (~m)
G = 9.81
RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
R = 6.371e6  # m


def _get_alpha(
    salinity, temperature, pressure, longitude, latitude
):
    """
    Calculate the thermal expansion coefficient alpha.

    Parameters
    ----------
    salinity: xarray.DataArray
        Salinity in PSU.
    temperature: xarray.DataArray
        Temperature in degrees Celsius.
    pressure: xarray.DataArray
        Pressure in dbar.
    longitude: xarray.DataArray
        Longitude in degrees.
    latitude: xarray.DataArray
        Latitude in degrees.

    Returns
    -------
    xarray.DataArray
        Thermal expansion coefficient alpha (1/°C).
    """
    sa = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    ct = gsw.CT_from_t(sa, temperature, pressure)
    return gsw.alpha(sa, ct, pressure)


def save_alpha():
    """Calculate and save thermal expansion coefficient alpha dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/alpha.nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset("datasets/Temperature-(2004-2018).nc")
    s = load_and_prepare_dataset("datasets/Salinity-(2004-2018).nc")

    alpha = xr.apply_ufunc(
        _get_alpha,
        s.SALINITY, t.TEMPERATURE, s.PRESSURE, s.LONGITUDE, s.LATITUDE,
        input_core_dims=[["PRESSURE"], ["PRESSURE"], ["PRESSURE"], [], []],
        output_core_dims=[["PRESSURE"]], vectorize=True
    )

    alpha.attrs['units'] = '1/K'
    alpha.attrs['long_name'] = (
        'Thermal Expansion Coefficient Jan 2004 - Dec 2018 (15.0 year)'
    )
    alpha.name = 'alpha'

    save_file(
        alpha,
        "datasets/alpha.nc"
    )


def save_sea_surface_height():
    """Calculate and save sea surface height dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sea_Surface_Height-(2004-2018).nc" in logs_datasets.read():
            return

    alpha = load_and_prepare_dataset("datasets/alpha.nc")['alpha']

    alpha_integrate = alpha.sel(PRESSURE=slice(0, REF_PRESSURE)).integrate("PRESSURE")

    ssh = (G * REF_PRESSURE + alpha_integrate) / G

    ssh.attrs['units'] = 'dbar'
    ssh.attrs['long_name'] = (
        'Monthly Sea Surface Height Jan 2004 - Dec 2018 (15.0 year)'
    )
    ssh.name = 'ssh'

    save_file(
        ssh,
        "datasets/Sea_Surface_Height-(2004-2018).nc"
    )


def save_monthly_mean_sea_surface_height():
    """Calculate and save monthly mean sea surface height dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sea_Surface_Height-Seasonal_Mean.nc" in logs_datasets.read():
            return

    ssh = load_and_prepare_dataset(
        "datasets/Sea_Surface_Height-(2004-2018).nc"
    )['ssh']

    ssh_monthly_mean = get_monthly_mean(ssh)

    save_file(
        ssh_monthly_mean,
        "datasets/Sea_Surface_Height-Seasonal_Mean.nc"
    )


def save_sea_surface_height_anomalies():
    """Calculate and save sea surface height anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sea_Surface_Height_Anomalies-(2004-2018).nc" in logs_datasets.read():
            return

    ssh = load_and_prepare_dataset(
        "datasets/Sea_Surface_Height-(2004-2018).nc"
    )['ssh']
    ssh_monthly_mean = load_and_prepare_dataset(
        "datasets/Sea_Surface_Height-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_ssh']

    ssh_a = get_anomaly(ssh, ssh_monthly_mean)

    save_file(
        ssh_a,
        "datasets/Sea_Surface_Height_Anomalies-(2004-2018).nc"
    )


def save_geostrophic_anomaly_temperature():
    """Calculate and save Geostrophic anomaly temperature dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2018).nc" in logs_datasets.read():
            return

    t_m_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Temperature-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_ML_TEMPERATURE']
    t_m_a = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc"
    )['ANOMALY_ML_TEMPERATURE']
    ssh_monthly_mean = load_and_prepare_dataset(
        "datasets/Sea_Surface_Height-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_ssh']
    ssh_a = load_and_prepare_dataset(
        "datasets/Sea_Surface_Height_Anomalies-(2004-2018).nc"
    )['ANOMALY_ssh']
    h_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_MLD']

    t_m_monthly_mean = xr.concat([t_m_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    t_m_monthly_mean = t_m_monthly_mean.rename({'MONTH': 'TIME'})
    t_m_monthly_mean['TIME'] = t_m_a.TIME
    ssh_monthly_mean = xr.concat([ssh_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    ssh_monthly_mean = ssh_monthly_mean.rename({'MONTH': 'TIME'})
    ssh_monthly_mean['TIME'] = ssh_a.TIME
    h_monthly_mean = xr.concat([h_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
    h_monthly_mean['TIME'] = t_m_a.TIME

    # dt_m_bar
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

    # dt_m_a
    dt_m_a_dtheta = (
        np.gradient(t_m_a, axis=t_m_a.get_axis_num('LONGITUDE'))
    )
    dt_m_a_dtheta_da = xr.DataArray(
        dt_m_a_dtheta,
        coords=t_m_a.coords,
        dims=t_m_a.dims,
    )
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(t_m_a['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=t_m_a['LONGITUDE'])
    dt_m_a_dx_da = dt_m_a_dtheta_da * dtheta_dx
    dt_m_a_dx_da.name = 'dt_m_a_dx'

    dt_m_a_dphi = (
        np.gradient(t_m_a, axis=t_m_a.get_axis_num('LATITUDE'))
    )
    dt_m_a_dphi_da = xr.DataArray(
        dt_m_a_dphi,
        coords=t_m_a.coords,
        dims=t_m_a.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    dt_m_a_dy_da = dt_m_a_dphi_da * dphi_dy
    dt_m_a_dy_da.name = 'dt_m_a_dy'

    # dssh_bar
    dssh_monthly_mean_dtheta = (
        np.gradient(ssh_monthly_mean, axis=ssh_monthly_mean.get_axis_num('LONGITUDE'))
    )
    dssh_monthly_mean_dtheta_da = xr.DataArray(
        dssh_monthly_mean_dtheta,
        coords=ssh_monthly_mean.coords,
        dims=ssh_monthly_mean.dims,
    )
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(ssh_monthly_mean['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=ssh_monthly_mean['LONGITUDE'])
    dssh_monthly_mean_dx_da = dssh_monthly_mean_dtheta_da * dtheta_dx
    dssh_monthly_mean_dx_da.name = 'dssh_monthly_mean_dx'

    dssh_monthly_mean_dphi = (
        np.gradient(ssh_monthly_mean, axis=ssh_monthly_mean.get_axis_num('LATITUDE'))
    )
    dssh_monthly_mean_dphi_da = xr.DataArray(
        dssh_monthly_mean_dphi,
        coords=ssh_monthly_mean.coords,
        dims=ssh_monthly_mean.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    dssh_monthly_mean_dy_da = dssh_monthly_mean_dphi_da * dphi_dy
    dssh_monthly_mean_dy_da.name = 'dssh_monthly_mean_dy'

    # dssh_a
    dssh_a_dtheta = (
        np.gradient(ssh_a, axis=ssh_a.get_axis_num('LONGITUDE'))
    )
    dssh_a_dtheta_da = xr.DataArray(
        dssh_a_dtheta,
        coords=ssh_a.coords,
        dims=ssh_a.dims,
    )
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(ssh_a['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=ssh_a['LONGITUDE'])
    dssh_a_dx_da = dssh_a_dtheta_da * dtheta_dx
    dssh_a_dx_da.name = 'dssh_a_dx'

    dssh_a_dphi = (
        np.gradient(ssh_a, axis=ssh_a.get_axis_num('LATITUDE'))
    )
    dssh_a_dphi_da = xr.DataArray(
        dssh_a_dphi,
        coords=ssh_a.coords,
        dims=ssh_a.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    dssh_a_dy_da = dssh_a_dphi_da * dphi_dy
    dssh_a_dy_da.name = 'dssh_a_dy'

    f = 2 * (2*np.pi/(24*60*60)) * np.sin(
        np.deg2rad(t_m_monthly_mean['LATITUDE'])
    )
    f = f.expand_dims(LONGITUDE=t_m_monthly_mean['LONGITUDE'])

    q_geostrophic_a = RHO_O * C_O * G / f * h_monthly_mean * (
        (dssh_a_dy_da * dt_m_monthly_mean_dx_da)
        - (dssh_a_dx_da * dt_m_monthly_mean_dy_da)
        + (dssh_monthly_mean_dy_da * dt_m_a_dx_da)
        - (dssh_monthly_mean_dx_da * dt_m_a_dy_da)
    )

    q_geostrophic_a = q_geostrophic_a.drop_vars('MONTH')

    q_geostrophic_a.attrs['units'] = 'W/m^2'
    q_geostrophic_a.attrs['long_name'] = (
        'Monthly Q_Geostrophic Anomaly Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_geostrophic_a.name = 'ANOMALY_GEOSTROPHIC_HEAT_FLUX'

    save_file(
        q_geostrophic_a,
        "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2018).nc"
    )


def save_geostrophic_anomaly_salinity():
    """Calculate and save Geostrophic anomaly salinity dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Geostrophic_Water_Rate-(2004-2018).nc" in logs_datasets.read():
            return

    s_m_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Salinity-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_ML_SALINITY']
    s_m_a = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Salinity_Anomalies-(2004-2018).nc"
    )['ANOMALY_ML_SALINITY']
    ssh_monthly_mean = load_and_prepare_dataset(
        "datasets/Sea_Surface_Height-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_ssh']
    ssh_a = load_and_prepare_dataset(
        "datasets/Sea_Surface_Height_Anomalies-(2004-2018).nc"
    )['ANOMALY_ssh']
    h_monthly_mean = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth-Seasonal_Mean.nc"
    )['MONTHLY_MEAN_MLD']

    s_m_monthly_mean = xr.concat([s_m_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    s_m_monthly_mean = s_m_monthly_mean.rename({'MONTH': 'TIME'})
    s_m_monthly_mean['TIME'] = s_m_a.TIME
    ssh_monthly_mean = xr.concat([ssh_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    ssh_monthly_mean = ssh_monthly_mean.rename({'MONTH': 'TIME'})
    ssh_monthly_mean['TIME'] = ssh_a.TIME
    h_monthly_mean = xr.concat([h_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
    h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
    h_monthly_mean['TIME'] = s_m_a.TIME

    # ds_m_bar
    ds_m_monthly_mean_dtheta = (
        np.gradient(s_m_monthly_mean, axis=s_m_monthly_mean.get_axis_num('LONGITUDE'))
    )
    ds_m_monthly_mean_dtheta_da = xr.DataArray(
        ds_m_monthly_mean_dtheta,
        coords=s_m_monthly_mean.coords,
        dims=s_m_monthly_mean.dims,
    )
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(s_m_monthly_mean['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=s_m_monthly_mean['LONGITUDE'])
    ds_m_monthly_mean_dx_da = ds_m_monthly_mean_dtheta_da * dtheta_dx
    ds_m_monthly_mean_dx_da.name = 'ds_m_monthly_mean_dx'

    ds_m_monthly_mean_dphi = (
        np.gradient(s_m_monthly_mean, axis=s_m_monthly_mean.get_axis_num('LATITUDE'))
    )
    ds_m_monthly_mean_dphi_da = xr.DataArray(
        ds_m_monthly_mean_dphi,
        coords=s_m_monthly_mean.coords,
        dims=s_m_monthly_mean.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    ds_m_monthly_mean_dy_da = ds_m_monthly_mean_dphi_da * dphi_dy
    ds_m_monthly_mean_dy_da.name = 'ds_m_monthly_mean_dy'

    # ds_m_a
    ds_m_a_dtheta = (
        np.gradient(s_m_a, axis=s_m_a.get_axis_num('LONGITUDE'))
    )
    ds_m_a_dtheta_da = xr.DataArray(
        ds_m_a_dtheta,
        coords=s_m_a.coords,
        dims=s_m_a.dims,
    )
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(s_m_a['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=s_m_a['LONGITUDE'])
    ds_m_a_dx_da = ds_m_a_dtheta_da * dtheta_dx
    ds_m_a_dx_da.name = 'ds_m_a_dx'

    ds_m_a_dphi = (
        np.gradient(s_m_a, axis=s_m_a.get_axis_num('LATITUDE'))
    )
    ds_m_a_dphi_da = xr.DataArray(
        ds_m_a_dphi,
        coords=s_m_a.coords,
        dims=s_m_a.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    ds_m_a_dy_da = ds_m_a_dphi_da * dphi_dy
    ds_m_a_dy_da.name = 'ds_m_a_dy'

    # dssh_bar
    dssh_monthly_mean_dtheta = (
        np.gradient(ssh_monthly_mean, axis=ssh_monthly_mean.get_axis_num('LONGITUDE'))
    )
    dssh_monthly_mean_dtheta_da = xr.DataArray(
        dssh_monthly_mean_dtheta,
        coords=ssh_monthly_mean.coords,
        dims=ssh_monthly_mean.dims,
    )
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(ssh_monthly_mean['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=ssh_monthly_mean['LONGITUDE'])
    dssh_monthly_mean_dx_da = dssh_monthly_mean_dtheta_da * dtheta_dx
    dssh_monthly_mean_dx_da.name = 'dssh_monthly_mean_dx'

    dssh_monthly_mean_dphi = (
        np.gradient(ssh_monthly_mean, axis=ssh_monthly_mean.get_axis_num('LATITUDE'))
    )
    dssh_monthly_mean_dphi_da = xr.DataArray(
        dssh_monthly_mean_dphi,
        coords=ssh_monthly_mean.coords,
        dims=ssh_monthly_mean.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    dssh_monthly_mean_dy_da = dssh_monthly_mean_dphi_da * dphi_dy
    dssh_monthly_mean_dy_da.name = 'dssh_monthly_mean_dy'

    # dssh_a
    dssh_a_dtheta = (
        np.gradient(ssh_a, axis=ssh_a.get_axis_num('LONGITUDE'))
    )
    dssh_a_dtheta_da = xr.DataArray(
        dssh_a_dtheta,
        coords=ssh_a.coords,
        dims=ssh_a.dims,
    )
    dtheta_dx = 1 / (R * np.cos(np.deg2rad(ssh_a['LATITUDE']))) / (np.pi/180)
    dtheta_dx = dtheta_dx.expand_dims(LONGITUDE=ssh_a['LONGITUDE'])
    dssh_a_dx_da = dssh_a_dtheta_da * dtheta_dx
    dssh_a_dx_da.name = 'dssh_a_dx'

    dssh_a_dphi = (
        np.gradient(ssh_a, axis=ssh_a.get_axis_num('LATITUDE'))
    )
    dssh_a_dphi_da = xr.DataArray(
        dssh_a_dphi,
        coords=ssh_a.coords,
        dims=ssh_a.dims,
    )
    dphi_dy = 1 / R / (np.pi/180)
    dssh_a_dy_da = dssh_a_dphi_da * dphi_dy
    dssh_a_dy_da.name = 'dssh_a_dy'

    f = 2 * (2*np.pi/(24*60*60)) * np.sin(
        np.deg2rad(s_m_monthly_mean['LATITUDE'])
    )
    f = f.expand_dims(LONGITUDE=s_m_monthly_mean['LONGITUDE'])

    q_geostrophic_a = RHO_O * G / f * h_monthly_mean * (
        (dssh_a_dy_da * ds_m_monthly_mean_dx_da)
        - (dssh_a_dx_da * ds_m_monthly_mean_dy_da)
        + (dssh_monthly_mean_dy_da * ds_m_a_dx_da)
        - (dssh_monthly_mean_dx_da * ds_m_a_dy_da)
    )

    q_geostrophic_a = q_geostrophic_a.drop_vars('MONTH')

    q_geostrophic_a.attrs['units'] = 'kg/m^2/s'
    q_geostrophic_a.attrs['long_name'] = (
        'Monthly Q_Geostrophic Anomaly Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_geostrophic_a.name = 'ANOMALY_GEOSTROPHIC_WATER_RATE'

    save_file(
        q_geostrophic_a,
        "datasets/Simulation-Geostrophic_Water_Rate-(2004-2018).nc"
    )


def main():
    """Main function to calculate geostrophic term."""

    save_alpha()

    save_sea_surface_height()
    save_monthly_mean_sea_surface_height()
    save_sea_surface_height_anomalies()

    save_geostrophic_anomaly_temperature()
    save_geostrophic_anomaly_salinity()


if __name__ == "__main__":
    main()
