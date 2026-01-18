"""
Prepare ERA5 dataset.

NOTE: This argolise must be run under conda environment with xesmf installed.

Chengyun Zhu
2026-1-3
"""

import xarray as xr
import xesmf as xe  # NOTE: This argolise must be run under conda environment with xesmf installed.

from utilities import load_and_prepare_dataset
# from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


def _era5_regrid(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Process ERA5 datasets consistent with RG-ARGO datasets.

    Prerequisites:
    - Download ERA5 data from Copernicus Climate Data Store (CDS) manually or using cdsapi.
        cdsapi code in logs.
    - Ensure downloaded data is following some formats:
        - data_format: netcdf
        - product_type: monthly_averaged_reanalysis (moda) - len(ds['number']) == None

    Parameters
    ----------
    ds: xarray.Dataset
        Input ERA5 dataset.

    Returns
    -------
    xarray.Dataset
        Processed ERA5 dataset.

    Raises
    ------
    ValueError
        If unexpected length of 'number' dimension.
    """

    ds = ds.drop_vars(["expver"])
    if ds.sizes.get('number') is None:  # moda
        ds = ds.drop_vars(["number"])
    else:
        raise ValueError(
            "Unexpected length of 'number' dimension. Is this a ERA5 dataset?"
        )

    argo_ds = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2018).nc"
    )

    regridder = xe.Regridder(ds, argo_ds, "conservative")
    ds_regrided = regridder(ds)

    ds_regrided = ds_regrided.rename({'valid_time': 'TIME'})

    ds_regrided['TIME'] = (
        (ds_regrided['TIME'].dt.year - 2004) * 12
        + (ds_regrided['TIME'].dt.month - 0.5)
    ).astype('float32')

    ds_regrided['TIME'].attrs.update({
        'units': 'months since 2004-01-01 00:00:00',
        'time_origin': '01-JAN-2004 00:00:00',
        'axis': 'T'
    })

    ds_regrided.attrs.update(ds.attrs)
    for attr in list(ds_regrided.attrs):
        if attr.startswith('GRIB'):
            del ds_regrided.attrs[attr]

    for var in ds_regrided.data_vars:
        ds_regrided[var].attrs = ds[var].attrs
        for attr in list(ds_regrided[var].attrs):
            if attr.startswith('GRIB'):
                del ds_regrided[var].attrs[attr]

    return ds_regrided


def save_turbulent_surface_stress():
    """Save the regridded ERA5 turbulent surface stress dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Turbulent_Surface_Stress-(2004-2018).nc" in logs_datasets.read():
            return

    era5_turbulent_surface_stress = xr.open_dataset(
        "datasets/ERA5_Mean_Turbulent_Surface_Stress.nc"
    )

    turbulent_surface_stress = _era5_regrid(era5_turbulent_surface_stress)

    save_file(
        turbulent_surface_stress,
        "datasets/Turbulent_Surface_Stress-(2004-2018).nc"
    )


def save_surface_heat_flux():
    """Save the regridded ERA5 surface heat flux dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Surface_Heat_Flux-(2004-2018).nc" in logs_datasets.read():
            return

    era5_surface_heat_flux = xr.open_dataset(
        "datasets/ERA5_Mean_Surface_Heat_Flux.nc"
    )

    surface_heat_flux = _era5_regrid(era5_surface_heat_flux)

    save_file(
        surface_heat_flux,
        "datasets/Surface_Heat_Flux-(2004-2018).nc"
    )


def save_surface_water_rate():
    """Save the regridded ERA5 surface water rate dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Surface_Water_Rate-(2004-2018).nc" in logs_datasets.read():
            return

    era5_surface_water_rate = xr.open_dataset(
        "datasets/ERA5_Mean_Surface_Water_Rate.nc"
    )

    surface_water_rate = _era5_regrid(era5_surface_water_rate)

    save_file(
        surface_water_rate,
        "datasets/Surface_Water_Rate-(2004-2018).nc"
    )


def main():
    """Main function to prepare datasets from RGARGO."""

    save_turbulent_surface_stress()
    save_surface_heat_flux()
    save_surface_water_rate()


if __name__ == "__main__":
    main()
