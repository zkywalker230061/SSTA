"""
Download ERA5 datasets.

Chengyun Zhu
2024-10-22
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import xesmf as xe

# from rgargo_read import load_and_prepare_dataset


def era5_argolise(ds: xr.Dataset) -> xr.Dataset:
    """
    Process ERA5 datasets consistent with RG-ARGO datasets.

    Prerequisites:
    - Download ERA5 data from Copernicus Climate Data Store (CDS) manually or using cdsapi.
        cdsapi example in main.
    - Ensure downloaded data is following some format consistent with RG-ARGO datasets:
        - longitude: -179.5 to 179.5
        - latitude: 79.5 to -64.5
        - grid: 1.0 x 1.0 degree
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
    ds = ds.rename({'longitude': 'LONGITUDE', 'latitude': 'LATITUDE', 'valid_time': 'TIME'})
    ds['LONGITUDE'].attrs.update({'point_spacing': 'even', 'axis': 'X'})
    ds['LATITUDE'].attrs.update({'point_spacing': 'even', 'axis': 'Y'})
    ds = ds.sortby('LONGITUDE', ascending=True)
    ds = ds.sortby('LATITUDE', ascending=True)
    ds['LATITUDE'].attrs.pop('stored_direction')
    # change time dimension from '2004-01-01T06:00:00.000000000' to 0.5, 1.5, ...
    ds['TIME'] = (
        (ds['TIME'].dt.year - 2004) * 12
        + (ds['TIME'].dt.month - 0.5)
    ).astype('float32')
    ds['TIME'].attrs.update({
        'units': 'months since 2004-01-01 00:00:00',
        'time_origin': '01-JAN-2004 00:00:00',
        'axis': 'T'
    })
    # argo_ds = load_and_prepare_dataset(
    #     "../datasets/Temperature (2004-2018).nc"
    # )
    # regridder = xe.Regridder(ds, argo_ds, "conservative")
    # ds_interpolated = regridder(ds)
    # return ds_interpolated
    return ds


if __name__ == "__main__":

    # import cdsapi

    # dataset = "reanalysis-era5-single-levels-monthly-means"
    # request = {
    #     "product_type": [
    #         "monthly_averaged_reanalysis",
    #         # "monthly_averaged_ensemble_members"
    #     ],
    #     "variable": [
    #         "mean_eastward_turbulent_surface_stress",
    #         "mean_northward_turbulent_surface_stress"
    #     ],
    #     "year": [
    #         "2004", "2005", "2006",
    #         "2007", "2008", "2009",
    #         "2010", "2011", "2012",
    #         "2013", "2014", "2015",
    #         "2016", "2017", "2018",
    #         # "2019", "2020", "2021",
    #         # "2022", "2023", "2024",
    #         # "2025"
    #     ],
    #     "month": [
    #         "01", "02", "03",
    #         "04", "05", "06",
    #         "07", "08", "09",
    #         "10", "11", "12"
    #     ],
    #     "time": ["00:00"],
    #     # "grid": "1.0/1.0",
    #     "data_format": "netcdf",
    #     "download_format": "unarchived",
    #     "area": [79.5, -179.5, -64.5, 179.5]
    # }

    # client = cdsapi.Client()
    # client.retrieve(dataset, request).download()

    # era5_edmo_ds = xr.open_dataset("../datasets/data_stream-edmo_stepType-avgad.nc")
    # era5_edmo_ds = era5_argolise(era5_edmo_ds)
    # display(era5_edmo_ds)
    era5_moda_ds = xr.open_dataset("../datasets/ERA5_Mean_Turbulent_Surface_Stress.nc")
    era5_moda_ds = era5_argolise(era5_moda_ds)
    display(era5_moda_ds)

    print(era5_moda_ds['avg_iews'].max().item(), era5_moda_ds['avg_inss'].max().item())
    print(
        abs(era5_moda_ds['avg_iews']).mean().item(),
        abs(era5_moda_ds['avg_inss']).mean().item()
    )

    TIME = 6.5
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    x = era5_moda_ds['LONGITUDE'].values
    y = era5_moda_ds['LATITUDE'].values
    u = era5_moda_ds['avg_iews'].sel(
        TIME=TIME
    ).values
    v = era5_moda_ds['avg_inss'].sel(
        TIME=TIME
    ).values
    magnitude = np.hypot(u, v)
    ax.streamplot(
        x, y, u, v,
        transform=ccrs.PlateCarree(),
        linewidth=1, density=1, color=magnitude
        )
    plt.show()
