"""
Calculate Geostrophic term.

Chengyun Zhu
2026-1-12
"""

import xarray as xr
# import numpy as np
import gsw

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file

REF_PRESSURE = 2000  # dbar (~m)
GRAVITATIONAL_CONSTANT = 9.81


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


def save_sea_surface_height():
    """Calculate and save sea surface height dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Sea_Surface_Height-(2004-2018).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset("datasets/Temperature-(2004-2018).nc")
    s = load_and_prepare_dataset("datasets/Salinity-(2004-2018).nc")

    alpha = xr.apply_ufunc(
        _get_alpha,
        s.SALINITY, t.TEMPERATURE, s.PRESSURE, s.LONGITUDE, s.LATITUDE,
        input_core_dims=[["PRESSURE"], ["PRESSURE"], ["PRESSURE"], [], []],
        output_core_dims=[["PRESSURE"]], vectorize=True
    )
    alpha_integrate = alpha.sel(PRESSURE=slice(0, REF_PRESSURE)).integrate("PRESSURE")

    ssh = (GRAVITATIONAL_CONSTANT * REF_PRESSURE + alpha_integrate) / GRAVITATIONAL_CONSTANT

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

    ssh_anomalies = get_anomaly(ssh, ssh_monthly_mean)

    save_file(
        ssh_anomalies,
        "datasets/Sea_Surface_Height_Anomalies-(2004-2018).nc"
    )


def main():
    """Main function to calculate geostrophic term."""

    save_sea_surface_height()
    save_monthly_mean_sea_surface_height()
    save_sea_surface_height_anomalies()


if __name__ == "__main__":
    main()
