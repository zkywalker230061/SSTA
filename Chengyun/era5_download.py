"""
Download ERA5 datasets.

NOTE: This argolise must be run under conda environment with xesmf installed.

Chengyun Zhu
2024-10-22
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cdsapi
import xesmf as xe  # NOTE: This argolise must be run under conda environment with xesmf installed.

from rgargo_read import load_and_prepare_dataset
from rgargo_plot import visualise_dataset


def era5_argolise(ds: xr.Dataset) -> xr.Dataset:
    """
    Process ERA5 datasets consistent with RG-ARGO datasets.

    Prerequisites:
    - Download ERA5 data from Copernicus Climate Data Store (CDS) manually or using cdsapi.
        cdsapi example in main.
    - Ensure downloaded data is following some formats:
        - data_format: netcdf
        - product_type:
            monthly_averaged_reanalysis (moda) - len(ds['number']) == None
            monthly_averaged_ensemble_members (edmo) - len(ds['number']) == 10

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
    NotImplementedError
        If edmo processing is not implemented yet.
    ValueError
        If unexpected length of 'number' dimension.
    """

    ds = ds.drop_vars(["expver"])
    if ds.sizes.get('number') is None:  # moda
        ds = ds.drop_vars(["number"])
    elif ds.sizes.get('number') == 10:  # edmo
        raise NotImplementedError("edmo processing not implemented yet.")
    else:
        with open("_Kal'tsit.txt", encoding="utf-8") as kaltsit:
            print(kaltsit.read())
            print("Something is wrong, but now you have Kal'tsit.")
        raise ValueError(
            "Unexpected length of 'number' dimension. Is this a ERA5 dataset?"
        )

    argo_ds = load_and_prepare_dataset(
        "../datasets/Temperature-(2004-2018).nc"
    )
    regridder = xe.Regridder(ds, argo_ds, "conservative")
    ds_interpolated = regridder(ds)

    ds_interpolated = ds_interpolated.rename({'valid_time': 'TIME'})
    ds_interpolated['TIME'] = (
        (ds_interpolated['TIME'].dt.year - 2004) * 12
        + (ds_interpolated['TIME'].dt.month - 0.5)
    ).astype('float32')
    ds_interpolated['TIME'].attrs.update({
        'units': 'months since 2004-01-01 00:00:00',
        'time_origin': '01-JAN-2004 00:00:00',
        'axis': 'T'
    })

    ds_interpolated.attrs.update(ds.attrs)
    for attr in list(ds_interpolated.attrs):
        if attr.startswith('GRIB'):
            del ds_interpolated.attrs[attr]

    for var in ds_interpolated.data_vars:
        ds_interpolated[var].attrs = ds[var].attrs
        for attr in list(ds_interpolated[var].attrs):
            if attr.startswith('GRIB'):
                del ds_interpolated[var].attrs[attr]

    return ds_interpolated


def download_wind_stress():
    """Download ERA5 wind stress datasets using cdsapi."""

    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": [
            "monthly_averaged_reanalysis",
            # "monthly_averaged_ensemble_members"
        ],
        "variable": [
            "mean_eastward_turbulent_surface_stress",
            "mean_northward_turbulent_surface_stress"
        ],
        "year": [
            "2004", "2005", "2006",
            "2007", "2008", "2009",
            "2010", "2011", "2012",
            "2013", "2014", "2015",
            "2016", "2017", "2018",
            # "2019", "2020", "2021",
            # "2022", "2023", "2024",
            # "2025"
        ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        # "grid": "1.0/1.0",
        "data_format": "netcdf",
        "download_format": "unarchived",
        # "area": [79.5, -179.5, -64.5, 179.5]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()


def download_heat_flux():
    """Download ERA5 heat flux datasets using cdsapi."""

    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "mean_surface_latent_heat_flux",
            "mean_surface_net_long_wave_radiation_flux",
            "mean_surface_net_short_wave_radiation_flux",
            "mean_surface_sensible_heat_flux"
        ],
        "year": [
            "2004", "2005", "2006",
            "2007", "2008", "2009",
            "2010", "2011", "2012",
            "2013", "2014", "2015",
            "2016", "2017", "2018",
            # "2019", "2020", "2021",
            # "2022", "2023", "2024",
            # "2025"
        ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()


if __name__ == "__main__":

    # download_wind_stress()
    # download_heat_flux()

    # era5_wind_stress = xr.open_dataset("../datasets/ERA5_Mean_Turbulent_Surface_Stress.nc")
    # era5_wind_stress = era5_argolise(era5_wind_stress)
    # display(era5_wind_stress)
    # era5_wind_stress.to_netcdf("../datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress.nc")

    era5_wind_stress = load_and_prepare_dataset(
        "../datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress.nc"
    )
    display(era5_wind_stress)
    print(
        abs(era5_wind_stress['avg_iews']).mean().item(),
        abs(era5_wind_stress['avg_inss']).mean().item()
    )
    print(era5_wind_stress['avg_iews'].max().item(), era5_wind_stress['avg_inss'].max().item())

    TIME = 6.5
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    x = era5_wind_stress['LONGITUDE'].values
    y = era5_wind_stress['LATITUDE'].values
    u = era5_wind_stress['avg_iews'].sel(
        TIME=TIME
    ).values
    v = era5_wind_stress['avg_inss'].sel(
        TIME=TIME
    ).values
    magnitude = np.hypot(u, v)
    ax.streamplot(
        x, y, u, v,
        transform=ccrs.PlateCarree(),
        linewidth=1, density=1, color=magnitude
        )
    plt.show()

    # era5_heat_flux = xr.open_dataset("../datasets/ERA5_Mean_Surface_Heat_Flux.nc")
    # era5_heat_flux = era5_argolise(era5_heat_flux)
    # display(era5_heat_flux)
    # era5_heat_flux.to_netcdf("../datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc")

    era5_heat_flux = load_and_prepare_dataset(
        "../datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
    )
    display(era5_heat_flux)
    visualise_dataset(
        era5_heat_flux['avg_slhtf'].sel(TIME=0.5),
    )
    visualise_dataset(
        era5_heat_flux['avg_ishf'].sel(TIME=0.5),
    )
    visualise_dataset(
        era5_heat_flux['avg_snswrf'].sel(TIME=0.5),
        cmap='Reds'
    )
    visualise_dataset(
        era5_heat_flux['avg_snlwrf'].sel(TIME=0.5),
        cmap='Blues_r'
    )
