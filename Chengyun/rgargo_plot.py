"""
Plot RGARGO datasets.

Chengyun Zhu
2025-10-14
"""

from IPython.display import display

import xarray as xr
import matplotlib.pyplot as plt


DEFAULT_MAP_PLOT_SETTINGS = {
    'figsize': (10, 5),
    'ylim': (-90, 90),
    'cmap': 'RdBu_r',
}
DEFAULT_POINT_PLOT_SETTINGS = {
    'y': 'PRESSURE',
    'yincrease': False,
    'color': '#66CCFF',
}


def map_visualise_dataset(
    ds: xr.Dataset,
    *args,
    **kwargs
) -> None:
    """
    Visualise the dataset of the globe.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to visualise.
    *args, **kwargs
        Additional arguments for the plot method.

    Raises
    ------
    TypeError
        If the dataset is not at a single pressure level.
    """
    display(ds)
    if ds['PRESSURE'].size != 1:
        raise TypeError(
            "Map visualisation requires the dataset to be at a single pressure level."
        )
    if ds['LONGITUDE'].attrs.get('modulo') == 180:
        kwargs.setdefault('xlim', (-180, 180))
    else:
        pass
    for key, value in DEFAULT_MAP_PLOT_SETTINGS.items():
        kwargs.setdefault(key, value)
    ds.plot(*args, **kwargs)
    plt.show()


def point_visualise_dataset(
    ds: xr.Dataset,
    *args,
    **kwargs
) -> None:
    """
    Visualise the dataset at a specific point.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to visualise.
    *args, **kwargs
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
    ds.plot(*args, **kwargs)
    plt.show()


def visualise_dataset(
        ds: xr.Dataset,
        *args,
        **kwargs
) -> None:
    """
    Visualise the dataset using map and point visualisation.

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to visualise.
    *args, **kwargs
        Additional arguments for the plot method.

    Raises
    ------
    TypeError
        If the dataset is not for map visualisation or point visualisation.
    """
    if ds['PRESSURE'].size == 1 and ds['LONGITUDE'].size != 1 and ds['LATITUDE'].size != 1:
        map_visualise_dataset(ds, *args, **kwargs)
    elif ds['PRESSURE'].size != 1 and ds['LONGITUDE'].size == 1 and ds['LATITUDE'].size == 1:
        point_visualise_dataset(ds, *args, **kwargs)
    else:
        raise TypeError(
            "Dataset must be at a single pressure level for map visualisation or "
            "at a single point for point visualisation."
        )


def main():
    """Main function for rgargo_plot.py."""

    # pass


if __name__ == "__main__":
    main()
