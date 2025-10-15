"""
Finding hbar from Temperature_Monthly_Mean.nc.

Chengyun Zhu
2024-10-15
"""

from IPython.display import display

import xarray as xr

from rgargo_read import load_and_prepare_dataset
# from rgargo_plot import visualise_dataset

HBAR_TDIFF = 1


def main():
    """Main function to find hbar."""

    t_monthly_mean = load_and_prepare_dataset(
        "../datasets/Temperature_Monthly_Mean.nc"
    )
    display(t_monthly_mean)
    hbar_t = (
        t_monthly_mean['Monthly Mean of Temperature'].sel(PRESSURE=0, MONTH=1, method='nearest')
        - HBAR_TDIFF
    )
    # display(hbar_t)
    # visualise_dataset(hbar_t)
    hbar = xr.DataArray()
    for lon in hbar_t['LONGITUDE']:
        for lat in hbar_t['LATITUDE']:
            t_to_find = hbar_t.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest').item()
            depth = hbar_t.sel(
                LONGITUDE=lon, LATITUDE=lat, value=t_to_find, method='nearest'
            ).item()
            hbar.loc[dict(LONGITUDE=lon, LATITUDE=lat, MONTH=1)] = depth
            # print(f"Longitude: {lon}, Latitude: {lat}, Depth: {depth}")


if __name__ == "__main__":
    main()
