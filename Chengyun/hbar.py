"""
Finding hbar from Temperature_Monthly_Mean.nc.

Method by Chris O.S., tweaked by Chengyun.
2024-10-15
"""

from IPython.display import display

import xarray as xr
import numpy as np

from rgargo_read import load_and_prepare_dataset
from rgargo_plot import visualise_dataset

HBAR_TDIFF = 0.8
MAX_DEPTH = float(2000)  # 1000


def find_half_depth(temp_profile, pressure):
    t_0 = temp_profile[0]  # temperature at surface == first reading
    t_mld = t_0 - HBAR_TDIFF  # temperature at mixed layer depth
    t_mld_alt = t_0 + HBAR_TDIFF

    # find where temperature falls below t_mld
    temperatures_below_mld = np.where(temp_profile <= t_mld)[0]
    temperatures_above_mld = np.where(temp_profile >= t_mld_alt)[0]
    if len(temperatures_below_mld) == 0 and len(temperatures_above_mld) != 0:
        above_mld_index = temperatures_above_mld[0]
        below_mld_index = above_mld_index - 1
    elif len(temperatures_below_mld) != 0 and len(temperatures_above_mld) == 0:
        below_mld_index = temperatures_below_mld[0]
        above_mld_index = below_mld_index - 1
    elif len(temperatures_below_mld) != 0 and len(temperatures_above_mld) != 0:
        # both found, take the above one
        above_mld_index = temperatures_above_mld[0]
        below_mld_index = above_mld_index - 1
    elif len(temperatures_below_mld) == 0 and len(temperatures_above_mld) == 0:
        if np.isnan(temp_profile[0].item()):
            return -np.inf  # land
        else:
            return MAX_DEPTH  # deep mixed layer
    else:
        raise ValueError("Unexpected condition in find_half_depth")

    # linear interpolation
    return np.interp(
        t_mld,
        [temp_profile[above_mld_index], temp_profile[below_mld_index]],
        [pressure[above_mld_index], pressure[below_mld_index]]
    )


def get_monthly_mld(
    ds: xr.Dataset,
    month: int
) -> xr.Dataset:
    ds = ds.sel(MONTH=month)
    # Apply this function along the depth dimension
    mld = xr.apply_ufunc(
        find_half_depth,
        ds['MONTHLY_MEAN_TEMPERATURE'], ds['PRESSURE'],
        input_core_dims=[['PRESSURE'], ['PRESSURE']], vectorize=True
    )

    ds.drop_vars(["PRESSURE"])  # don't need pressure anymore
    ds['MLD_PRESSURE'] = mld   # save to dataset
    ds['MLD_PRESSURE'] = ds['MLD_PRESSURE'].where(ds['MLD_PRESSURE'] <= MAX_DEPTH, MAX_DEPTH)

    return ds


def main():
    """Main function to find hbar."""

    t_monthly_mean = load_and_prepare_dataset(
        "../datasets/Temperature_Monthly_Mean.nc"
    )
    display(t_monthly_mean)

    monthly_datasets = []
    for month in range(1, 13):
        monthly_datasets.append(get_monthly_mld(t_monthly_mean, month))
    hbar_ds = xr.concat(monthly_datasets, "MONTH")
    hbar = hbar_ds['MLD_PRESSURE']

    # restore attributes
    hbar['LATITUDE'].attrs = t_monthly_mean['LATITUDE'].attrs
    hbar['LONGITUDE'].attrs = t_monthly_mean['LONGITUDE'].attrs
    hbar.attrs['units'] = 'dbar'
    hbar.attrs['long_name'] = (
        'Monthly Mean Mixed Layer Depth Pressure Jan 2004 - Dec 2018 (15.0 year)'
    )
    hbar.name = 'MONTHLY_MEAN_MLD_PRESSURE'
    # display(hbar)
    visualise_dataset(
        hbar.sel(MONTH=1, method='nearest'),
        cmap='Blues',
        vmin=0, vmax=500
    )
    # hbar.to_netcdf("../datasets/Mixed_Layer_Depth_Pressure_Monthly_Mean.nc")

    # check
    # m, lon, lat = 1, -47, 56
    # visualise_dataset(
    #     t_monthly_mean['MONTHLY_MEAN_TEMPERATURE'].sel(
    #         MONTH=m, LONGITUDE=lon, LATITUDE=lat, method='nearest'
    #     )
    # )
    # print(hbar.sel(MONTH=m, LONGITUDE=lon, LATITUDE=lat, method='nearest').item())


if __name__ == "__main__":
    main()
