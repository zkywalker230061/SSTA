"""
Error analysis for datasets.

Chengyun Zhu
2025-11-13
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from h_analysis import get_anomaly
# from rgargo_plot import visualise_dataset

MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3,
    'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9,
    'Oct': 10, 'Nov': 11, 'Dec': 12
}


def get_error(
    mean_da: xr.DataArray,
    anomaly_da: xr.DataArray
) -> xr.DataArray:
    """
    Get the error (standard deviation) of the DataArray based on the provided anomaly.
    Parameters
    ----------
    mean_da: xarray.DataArray
        Mean DataArray with 'MONTH' coordinate.
    anomaly_da: xarray.DataArray
        Anomaly DataArray with 'TIME' coordinate.

    Returns
    -------
    xarray.DataArray
        Error DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """

    if 'TIME' not in anomaly_da.dims:
        raise ValueError("The Anomaly DataArray must have a TIME dimension.")
    error_list = []
    for month_num in mean_da.coords['MONTH']:
        month_num = month_num.values
        month_anomalies = anomaly_da.sel(TIME=((anomaly_da['TIME'] + 0.5) % 12) == month_num % 12)
        error = np.sqrt(np.mean(month_anomalies**2, axis=0))
        error_list.append(error)
    error_da = xr.concat(error_list, "MONTH")
    error_da.attrs['units'] = anomaly_da.attrs.get('units')
    error_da.attrs['long_name'] = f"Error of {anomaly_da.attrs.get('long_name')}"
    error_da.name = f"ERROR_{anomaly_da.name}"
    return error_da


def get_mean_of_error(
    error_da: xr.DataArray
) -> xr.DataArray:
    """
    Get the mean of the error DataArray over lattitude and longitude.

    Parameters
    ----------
    error_da: xarray.DataArray
        Error DataArray with 'MONTH' coordinate.

    Returns
    -------
    xarray.DataArray
        Mean of error DataArray.
    """

    mean_error_da = error_da.mean(dim=['LATITUDE', 'LONGITUDE'])
    mean_error_da.attrs['long_name'] = f"Mean of {error_da.attrs.get('long_name')}"
    mean_error_da.name = f"MEAN_{error_da.name}"
    return mean_error_da


def error_analysis(da: xr.DataArray):
    """
    Perform error analysis on the provided DataArray.

    Parameters
    ----------
    da: xarray.DataArray
        The DataArray to analyze.
    """

    da_monthly_mean = get_monthly_mean(da)
    da_anomaly = get_anomaly(da, da_monthly_mean)

    da_error = get_error(da_monthly_mean, da_anomaly)
    # display(da_error)
    # visualise_dataset(
    #     da_error.sel(MONTH=1),
    #     cmap='Reds',
    # )

    da_mean_error = get_mean_of_error(da_error)
    display(da_mean_error)
    x = np.array(list(MONTHS.values()))
    plt.plot(
        x, da_mean_error,
        marker='o',
        linestyle='-',
        color='#66CCFF'
    )
    plt.xticks(x)
    plt.xlabel('Month')
    plt.ylabel(da_mean_error.attrs.get('units', ''))
    plt.title(da_mean_error.attrs.get('long_name', ''))
    plt.show()


def main():
    """Main function for error analysis."""
    mld_ds = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
    )
    # display(mld_ds)
    mld = mld_ds['MLD_PRESSURE']

    error_analysis(mld)

    heat_flux_ds = load_and_prepare_dataset(
        "../datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
    )
    # display(heat_flux_ds)
    heat_flux = (
        heat_flux_ds['avg_slhtf']
        + heat_flux_ds['avg_ishf']
        + heat_flux_ds['avg_snswrf']
        + heat_flux_ds['avg_snlwrf']
    )
    heat_flux.attrs.update(
        units='W m**-2', long_name='Net Surface Heat Flux'
    )
    heat_flux.name = 'NET_HEAT_FLUX'

    # error_analysis(heat_flux)

    ekman_ds = load_and_prepare_dataset(
        "../datasets/Ekman_Current_Anomaly.nc"
    )
    # display(ekman_ds)
    ekman = ekman_ds['Q_Ek_anom']

    # error_analysis(ekman)

    entrainment_ds = load_and_prepare_dataset(
        "../datasets/Entrainment_Heat_Flux-(2004-2018).nc"
    )
    # display(entrainment_ds)
    entrainment = entrainment_ds['ENTRAINMENT_HEAT_FLUX']

    # error_analysis(entrainment)

    heat_flux_error = get_mean_of_error(
        get_error(
            get_monthly_mean(heat_flux),
            get_anomaly(heat_flux, get_monthly_mean(heat_flux))
        )
    )

    entrainment_error = get_mean_of_error(
        get_error(
            get_monthly_mean(entrainment),
            get_anomaly(entrainment, get_monthly_mean(entrainment))
        )
    )

    ekman_error = get_mean_of_error(
        get_error(
            get_monthly_mean(ekman),
            get_anomaly(ekman, get_monthly_mean(ekman))
        )
    )

    plt.figure(figsize=(12, 6))
    x = np.array(list(MONTHS.values()))
    plt.plot(
        x, heat_flux_error,
        marker='o',
        linestyle='-',
        label='Heat Flux Standard Deviation',
        color='#FF9999'
    )
    plt.plot(
        x, entrainment_error,
        marker='o',
        linestyle='-',
        label='Entrainment Standard Deviation',
        color='#66CCFF'
    )
    plt.plot(
        x, ekman_error,
        marker='o',
        linestyle='-',
        label='Ekman Current Standard Deviation',
        color='#99FF99'
    )
    plt.xticks(x)
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation')
    plt.title('Mean Standard Deviation Over Months')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
