"""
Find h anomalies. NOT FINISHED.

Chengyun Zhu
2024-10-24
"""

# from IPython.display import display

import xarray as xr

from rgargo_read import load_and_prepare_dataset
from rgargo_plot import visualise_dataset


def main():
    """Main function to find h anomalies."""
    h = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc",
    )['MLD_PRESSURE']
    # display(h)
    hbar = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc",
    )['MONTHLY_MEAN_MLD_PRESSURE']
    # display(hbar)

    h_anomaly = None
    for month_num in range(1, 13):
        h_month = h.sel(
            TIME=h['TIME'][(month_num-1)::12]
        )
        hbar_month = hbar.sel(MONTH=month_num)
        h_anomaly_month = h_month - hbar_month
        if h_anomaly is None:
            h_anomaly = h_anomaly_month
        else:
            h_anomaly = xr.concat([h_anomaly, h_anomaly_month], dim="TIME")
    h_anomaly = h_anomaly.sortby("TIME")

    print(h_anomaly.sel(TIME=0.5).max().item(), h_anomaly.sel(TIME=0.5).min().item())
    # display(h_anomaly)
    visualise_dataset(
        h_anomaly.sel(TIME=0.5),
        cmap='Blues',
    )


if __name__ == "__main__":
    main()
