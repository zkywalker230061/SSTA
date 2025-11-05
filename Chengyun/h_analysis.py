"""
Find h anomalies. NOT FINISHED.

Chengyun Zhu
2024-10-24
"""

# from IPython.display import display

import xarray as xr
# import numpy as np

from rgargo_read import load_and_prepare_dataset
from rgargo_plot import visualise_dataset


MAX_DEPTH = float(500)


def get_anomaly(
    da: xr.DataArray,
    monthly_mean_da: xr.DataArray
) -> xr.DataArray:
    """
    Get the anomaly of the DataArray based on the provided monthly mean.
    Parameters
    ----------
    da: xarray.DataArray
        Input DataArray with 'TIME' coordinate.
    monthly_mean_da: xarray.DataArray
        Monthly mean DataArray with 'MONTH' coordinate.

    Returns
    -------
    xarray.DataArray
        Anomaly DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """

    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    anomalies = []
    for month_num in da.coords['TIME']:
        month_num = month_num.values
        month_mean_num = int((month_num + 0.5) % 12)
        if month_mean_num == 0:
            month_mean_num = 12
        anomalies.append(
            da.sel(TIME=month_num) - monthly_mean_da.sel(MONTH=month_mean_num)
        )
    anomaly_da = xr.concat(anomalies, "TIME")
    anomaly_da.attrs['units'] = da.attrs.get('units')
    anomaly_da.attrs['long_name'] = f"Anomaly of {da.attrs.get('long_name')}"
    anomaly_da.name = f"ANOMALY_{da.name}"
    return anomaly_da


def main():
    """Main function to find h anomalies."""

    h = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc",
    )['MLD_PRESSURE']
    # display(h)
    # visualise_dataset(
    #     h.sel(TIME=0.5),
    #     cmap='Blues',
    #     vmin=0, vmax=MAX_DEPTH
    # )
    hbar = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc",
    )['MONTHLY_MEAN_MLD_PRESSURE']
    # display(hbar)
    # visualise_dataset(
    #     hbar.sel(MONTH=1),
    #     cmap='Blues',
    #     vmin=0, vmax=MAX_DEPTH
    # )

    h_anomaly = get_anomaly(h, hbar)

    TIME = 6.5
    # print(h_anomaly.sel(TIME=TIME).max().item(), h_anomaly.sel(TIME=TIME).min().item())
    vlim = max(
        abs(h_anomaly.sel(TIME=TIME).max().item()),
        abs(h_anomaly.sel(TIME=TIME).min().item())
    )
    # display(h_anomaly)
    visualise_dataset(
        h_anomaly.sel(TIME=TIME),
        # cmap='Blues',
        vmin=-vlim, vmax=vlim
    )


if __name__ == "__main__":
    main()
