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
MAX_DEPTH = float(500)  # 1000


def find_half_depth(temp_profile, pressure):
    """"
    Find the mixed layer depth (hbar) from a temperature profile.

    Parameters
    ----------
    temp_profile : np.ndarray
        1D array of temperature values at different pressures.
    pressure : np.ndarray
        1D array of pressure values corresponding to the temperature profile.

    Returns
    -------
    float
        The pressure at the mixed layer depth (hbar).
        Returns MAX_DEPTH if not found, or -inf for land.
    """

    # sort pressure and temperature to be in order of increasing pressure
    indices_increasing_pressure = np.argsort(pressure)
    pressure = pressure[indices_increasing_pressure]
    temp_profile = temp_profile[indices_increasing_pressure]

    t_0 = temp_profile[0]  # temperature at surface == first reading

    # land
    if np.isnan(t_0):
        return -np.inf

    # t_mld_min = t_0 - HBAR_TDIFF
    # t_mld_max = t_0 + HBAR_TDIFF
    # pressures_below_t_0 = np.where(temp_profile <= t_mld_min)[0]
    # pressures_above_t_0 = np.where(temp_profile >= t_mld_max)[0]

    # def return_t_max():
    #     temperature_max = np.where(temp_profile == np.nanmax(temp_profile))[0]
    #     mld_t_max = pressure[temperature_max[0]]
    #     if mld_t_max <= MAX_DEPTH:
    #         return mld_t_max
    #     return MAX_DEPTH

    # # only t above t_mld_max
    # if len(pressures_below_t_0) == 0 and len(pressures_above_t_0) != 0:
    #     above_mld_index = pressures_above_t_0[0]
    #     below_mld_index = above_mld_index - 1
    #     t_mld = t_mld_max
    # # only t below t_mld_min
    # elif len(pressures_below_t_0) != 0 and len(pressures_above_t_0) == 0:
    #     below_mld_index = pressures_below_t_0[0]
    #     above_mld_index = below_mld_index - 1
    #     t_mld = t_mld_min
    # # both found, check which is closer to surface
    # elif len(pressures_below_t_0) != 0 and len(pressures_above_t_0) != 0:
    #     if pressures_above_t_0[0] < pressures_below_t_0[0]:
    #         above_mld_index = pressures_above_t_0[0]
    #         below_mld_index = above_mld_index - 1
    #         t_mld = t_mld_max
    #     else:
    #         below_mld_index = pressures_below_t_0[0]
    #         above_mld_index = below_mld_index - 1
    #         t_mld = t_mld_min
    # # neither found, return depth of max temperature
    # else:
    #     return return_t_max()

    # mld = np.interp(
    #     t_mld,
    #     [temp_profile[above_mld_index], temp_profile[below_mld_index]],
    #     [pressure[above_mld_index], pressure[below_mld_index]]
    # )
    # if mld <= MAX_DEPTH:
    #     return mld
    # return MAX_DEPTH

    t_mld = t_0 - HBAR_TDIFF
    pressures_below_t_0 = np.where(temp_profile <= t_mld)[0]

    if len(pressures_below_t_0) == 0:
        return MAX_DEPTH

    below_mld_index = pressures_below_t_0[0]
    above_mld_index = below_mld_index - 1

    mld = np.interp(
        t_mld,
        [temp_profile[above_mld_index], temp_profile[below_mld_index]],
        [pressure[above_mld_index], pressure[below_mld_index]]
    )
    if mld <= MAX_DEPTH:
        return mld
    return MAX_DEPTH


def get_monthly_mld(
    ds: xr.Dataset,
    month: int | None = None
) -> xr.Dataset:
    """
    Get the monthly mixed layer depth (MLD) from the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing temperature and pressure data.
    month : int | None, optional
        The month to select (1-12). If None, use the entire dataset.

    Returns
    -------
    xr.Dataset
        The dataset with an added variable 'MLD_PRESSURE' representing the mixed layer depth.
    """
    if month is not None:
        ds = ds.sel(MONTH=month)
    # Apply this function along the depth dimension
    mld = xr.apply_ufunc(
        find_half_depth,
        ds['MONTHLY_MEAN_TEMPERATURE'], ds['PRESSURE'],
        input_core_dims=[['PRESSURE'], ['PRESSURE']], vectorize=True
    )

    ds.drop_vars(["PRESSURE"])  # don't need pressure anymore
    ds['MLD_PRESSURE'] = mld   # save to dataset

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
        vmin=0, vmax=MAX_DEPTH
    )
    # hbar.to_netcdf("../datasets/Mixed_Layer_Depth_Pressure_Monthly_Mean.nc")

    # check
    m, lon, lat = 1, -47, 56
    visualise_dataset(
        t_monthly_mean['MONTHLY_MEAN_TEMPERATURE'].sel(
            MONTH=m, LONGITUDE=lon, LATITUDE=lat, method='nearest'
        )
    )
    print(hbar.sel(MONTH=m, LONGITUDE=lon, LATITUDE=lat, method='nearest').item())


if __name__ == "__main__":
    main()
