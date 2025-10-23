"""
Finding hbar from Salinity_Monthly_Mean.nc.

NOTE: ANY DENSITY IN THIS FILE IS POTENTIAL DENSITY.

Method by Chris O.S., tweaked by Chengyun.
2024-10-18
"""

from IPython.display import display

import xarray as xr
import numpy as np
import gsw

from rgargo_read import load_and_prepare_dataset
from rgargo_plot import visualise_dataset

HBAR_DDIFF = 0.03
MAX_DEPTH = float(500)  # 1000


def find_half_depth(density_anomaly_profile, pressure, density_anomaly_surface_mean):
    """
    Find the mixed layer depth (hbar) from a density anomaly profile.

    Parameters
    ----------
    density_anomaly_profile : np.ndarray
        1D array of density anomaly values at different pressures.
    pressure : np.ndarray
        1D array of pressure values corresponding to the density profile.
    density_anomaly_surface_mean : float
        Mean density anomaly near the surface.

    Returns
    -------
    float
        The pressure at the mixed layer depth (hbar).
        Returns MAX_DEPTH if not found, or -inf for land.
    """

    # sort pressure and temperature to be in order of increasing pressure
    indices_increasing_pressure = np.argsort(pressure)
    pressure = pressure[indices_increasing_pressure]
    density_anomaly_profile = density_anomaly_profile[indices_increasing_pressure]

    sigma_0 = density_anomaly_surface_mean  # density at surface == first reading

    # land
    if np.isnan(sigma_0):
        return -np.inf

    sigma_mld_min = sigma_0 - HBAR_DDIFF
    sigma_mld_max = sigma_0 + HBAR_DDIFF
    pressures_below_sigma_0 = np.where(density_anomaly_profile <= sigma_mld_min)[0]
    pressures_above_sigma_0 = np.where(density_anomaly_profile >= sigma_mld_max)[0]

    # only sigma below sigma_mld_min
    if len(pressures_below_sigma_0) != 0 and len(pressures_above_sigma_0) == 0:
        below_mld_index = pressures_below_sigma_0[0]
        above_mld_index = below_mld_index - 1
        sigma_mld = sigma_mld_min
    # only sigma above sigma_mld_max
    elif len(pressures_below_sigma_0) == 0 and len(pressures_above_sigma_0) != 0:
        above_mld_index = pressures_above_sigma_0[0]
        below_mld_index = above_mld_index - 1
        sigma_mld = sigma_mld_max
    # both found, check which is closer to surface
    elif len(pressures_below_sigma_0) != 0 and len(pressures_above_sigma_0) != 0:
        if pressures_above_sigma_0[0] < pressures_below_sigma_0[0]:
            above_mld_index = pressures_above_sigma_0[0]
            below_mld_index = above_mld_index - 1
            sigma_mld = sigma_mld_max
        else:
            below_mld_index = pressures_below_sigma_0[0]
            above_mld_index = below_mld_index - 1
            sigma_mld = sigma_mld_min
    # neither found, return depth of max temperature
    else:
        return MAX_DEPTH

    mld = np.interp(
        sigma_mld,
        [density_anomaly_profile[above_mld_index], density_anomaly_profile[below_mld_index]],
        [pressure[above_mld_index], pressure[below_mld_index]]
    )
    if mld <= MAX_DEPTH:
        return mld
    return MAX_DEPTH


def find_surface_density_anomaly_mean(
    ds: xr.Dataset,
    iteration: bool = True
) -> None:
    """
    Find the mean density anomaly near the surface (top 10 dbar). TEMPORARY FUNCTION.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing density anomaly and pressure data.
    iteration : bool
        Default is True.
        If True, use iterative method. If False, use direct slicing.
    """
    if iteration:
        sigma0_surface_mean = ds.DENSITY_ANOMALY.isel(PRESSURE=0)
        i = 0
        for lon, lat in zip(ds.LONGITUDE.values, ds.LATITUDE.values):
            while (
                abs(ds.DENSITY_ANOMALY.sel(LONGITUDE=lon, LATITUDE=lat).isel(PRESSURE=i+1) - sigma0_surface_mean.sel(LONGITUDE=lon, LATITUDE=lat)) < HBAR_DDIFF
                and i + 1 < ds.sizes['PRESSURE']
            ):
                sigma0_surface_mean = (
                    ds.DENSITY_ANOMALY.isel(PRESSURE=slice(0, i+1)).mean(dim='PRESSURE')
                )
                i += 1
        ds['SURFACE_DENSITY_ANOMALY_MEAN'] = sigma0_surface_mean
    else:
        ds_surface = ds.sel(PRESSURE=slice(0, 10))
        sigma0_surface_mean = ds_surface.DENSITY_ANOMALY.mean(dim='PRESSURE')
        ds['SURFACE_DENSITY_ANOMALY_MEAN'] = sigma0_surface_mean


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
    sa = gsw.SA_from_SP(
        SP=ds.MONTHLY_MEAN_SALINITY, p=ds.PRESSURE, lon=ds.LONGITUDE, lat=ds.LATITUDE
    )
    ct = gsw.CT_from_t(
        SA=sa, t=ds.MONTHLY_MEAN_TEMPERATURE, p=ds.PRESSURE
    )
    sigma0 = gsw.sigma0(SA=sa, CT=ct)
    ds['DENSITY_ANOMALY'] = sigma0
    ds['DENSITY_ANOMALY'].attrs = {"units": "kg/m^3"}

    find_surface_density_anomaly_mean(ds)

    # Apply this function along the depth dimension
    mld = xr.apply_ufunc(
        find_half_depth,
        ds['DENSITY_ANOMALY'], ds['PRESSURE'], ds['SURFACE_DENSITY_ANOMALY_MEAN'],
        input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True
    )

    ds.drop_vars(["PRESSURE"])  # don't need pressure anymore
    ds['MLD_PRESSURE'] = mld   # save to dataset

    return ds


def main():
    """Main function to find hbar."""

    s_monthly_mean = load_and_prepare_dataset(
        "../datasets/Salinity_Monthly_Mean.nc"
    )
    t_monthly_mean = load_and_prepare_dataset(
        "../datasets/Temperature_Monthly_Mean.nc"
    )
    monthly_mean = xr.merge([t_monthly_mean, s_monthly_mean])
    display(monthly_mean)

    monthly_datasets = []
    for month in range(1, 13):
        monthly_datasets.append(get_monthly_mld(monthly_mean, month))
    hbar_ds = xr.concat(monthly_datasets, "MONTH")
    hbar = hbar_ds['MLD_PRESSURE']

    # restore attributes
    hbar['LATITUDE'].attrs = monthly_mean['LATITUDE'].attrs
    hbar['LONGITUDE'].attrs = monthly_mean['LONGITUDE'].attrs
    hbar.attrs['units'] = 'dbar'
    hbar.attrs['long_name'] = (
        'Monthly Mean Mixed Layer Depth Pressure Jan 2004 - Dec 2018 (15.0 year)'
    )
    hbar.name = 'MONTHLY_MEAN_MLD_PRESSURE'
    # display(hbar)
    visualise_dataset(
        hbar.sel(MONTH=9, method='nearest'),
        cmap='Blues',
        vmin=0, vmax=MAX_DEPTH
    )
    # hbar.to_netcdf("../datasets/Mixed_Layer_Depth_Pressure_Monthly_Mean.nc")

    # check
    m, lon, lat = 1, -47, 56
    visualise_dataset(
        monthly_mean['MONTHLY_MEAN_SALINITY'].sel(
            MONTH=m, LONGITUDE=lon, LATITUDE=lat, method='nearest'
        )
    )
    print(hbar.sel(MONTH=m, LONGITUDE=lon, LATITUDE=lat, method='nearest').item())


if __name__ == "__main__":
    main()
