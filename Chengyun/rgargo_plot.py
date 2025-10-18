"""
Plot RGARGO datasets.

Chengyun Zhu
2025-10-14
"""

from IPython.display import display

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)


DEFAULT_MAP_PLOT_SETTINGS = {
    'figsize': (12, 6),
    'projection': ccrs.PlateCarree(),
    'cmap': 'RdBu_r',
    'levels': 200,
    'ylim': (-90, 90),
}
DEFAULT_POINT_PLOT_SETTINGS = {
    'y': 'PRESSURE',
    'yincrease': False,
    'color': '#66CCFF',
}


def map_visualise_dataset(
    ds: xr.Dataset,
    **kwargs
) -> None:
    """
    Visualise the dataset of the globe.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to visualise.
    **kwargs
        Additional arguments for the plot method.
    """
    display(ds)
    if ds['LONGITUDE'].attrs.get('modulo') == 180:
        kwargs.setdefault('xlim', (-180, 180))
    else:
        pass
    for key, value in DEFAULT_MAP_PLOT_SETTINGS.items():
        kwargs.setdefault(key, value)

    plt.figure(figsize=kwargs.pop('figsize'))
    ax = plt.axes(projection=kwargs['projection'])
    plt.contourf(
        ds['LONGITUDE'], ds['LATITUDE'], ds,
        cmap=kwargs.pop('cmap'),
        levels=kwargs.pop('levels'),
        vmin=kwargs.pop('vmin', None),
        vmax=kwargs.pop('vmax', None)
    )
    ax.coastlines()

    ax.set_xlim(kwargs.pop('xlim', None))
    ax.set_ylim(kwargs.pop('ylim'))
    if kwargs['projection'] == ccrs.PlateCarree():
        gl = ax.gridlines(
            crs=kwargs.pop('projection'), draw_labels=True,
            linewidth=2, color='gray', alpha=0.5, linestyle='--'
        )
        gl.top_labels = False
        gl.left_labels = True
        gl.right_labels = False
        gl.xlines = False
        gl.ylines = False
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = LatitudeLocator()
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.ylabel_style = {'size': 15, 'color': 'gray'}
        gl.xlabel_style = {'size': 15, 'color': 'gray'}

    plt.colorbar(label=ds.attrs.get('units'))
    title_str = ds.name
    if 'PRESSURE' in ds.coords:
        title_str += f", Pressure = {ds['PRESSURE'].item()} dbar"
    if 'TIME' in ds.coords:
        title_str += f", Time = {ds['TIME'].item()} months since 2004-01-01"
    elif 'MONTH' in ds.coords:
        title_str += f", Month = {ds['MONTH'].item()}"
    plt.title(title_str)

    plt.show()


def point_visualise_dataset(
    ds: xr.Dataset,
    **kwargs
) -> None:
    """
    Visualise the dataset at a specific point.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to visualise.
    **kwargs
        Additional arguments for the plot method.

    Raises
    ------
    TypeError
        If the dataset is not at a single point.
    """
    display(ds)
    if ds['LONGITUDE'].size != 1 or ds['LATITUDE'].size != 1:
        raise TypeError(
            "Point visualisation requires the dataset to be at a single point."
        )
    for key, value in DEFAULT_POINT_PLOT_SETTINGS.items():
        kwargs.setdefault(key, value)
    ds.plot(**kwargs)
    plt.show()


def visualise_dataset(
        ds: xr.Dataset,
        **kwargs
) -> None:
    """
    Visualise the dataset using map and point visualisation.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to visualise.
    **kwargs
        Additional arguments for the plot method.

    Raises
    ------
    TypeError
        If the dataset is not for map visualisation or point visualisation.
    """
    if 'PRESSURE' not in ds.coords:
        map_visualise_dataset(ds, **kwargs)

    elif ds['PRESSURE'].size == 1 and ds['LONGITUDE'].size != 1 and ds['LATITUDE'].size != 1:
        map_visualise_dataset(ds, **kwargs)
    elif ds['PRESSURE'].size != 1 and ds['LONGITUDE'].size == 1 and ds['LATITUDE'].size == 1:
        point_visualise_dataset(ds, **kwargs)
    else:
        with open("_Kal'tsit.txt", encoding="utf-8") as kaltsit:
            print(kaltsit.read())
            print("Something is wrong, but now you have Kal'tsit.")
        raise TypeError(
            "Dataset must be at a single pressure level for map visualisation or "
            "at a single point for point visualisation."
        )


def main():
    """Main function for rgargo_plot.py."""

    from rgargo_read import load_and_prepare_dataset

    ds_temp = load_and_prepare_dataset(
        "../datasets/RG_ArgoClim_Temperature_2019.nc",
    )
    display(ds_temp)
    # visualise_dataset(ds_temp)  # Call Kal'tsit.

    # meant_0: Mean Temperature for 15 years at surface
    meant_0 = ds_temp['ARGO_TEMPERATURE_MEAN'].sel(PRESSURE=0, method='nearest')
    # print(meant_0.min().item(), meant_0.max().item())
    visualise_dataset(meant_0, vmin=-2, vmax=31)

    # ta_0_2004jan: Temperature Anomaly at surface in 2004-01
    ta_0_2004jan = ds_temp['ARGO_TEMPERATURE_ANOMALY'].sel(TIME=0.5).sel(
        PRESSURE=0, method='nearest'
    )
    # print(ta_0_2004jan.min().item(), ta_0_2004jan.max().item())
    visualise_dataset(ta_0_2004jan, vmin=-8, vmax=8)

    # ta_all_2024jan_e0n0: Temperature Anomaly at all depths in 2024-01 at (0°E, 0°N)
    ta_all_2004jan_e0n0 = ds_temp['ARGO_TEMPERATURE_ANOMALY'].sel(TIME=0.5).sel(
        LONGITUDE=0, LATITUDE=0, method='nearest'
    )
    # print(ta_all_2004jan_e0n0.min().item(), ta_all_2004jan_e0n0.max().item())
    visualise_dataset(ta_all_2004jan_e0n0)


if __name__ == "__main__":
    main()
