"""
Calculate Mixed Layer Depth (h) and (hbar) datasets.

Chengyun Zhu
2026-1-6
"""

import xarray as xr
import numpy as np
import gsw

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean  # , get_anomaly
from utilities import save_file

HBAR_DDIFF = 0.03
MAX_DEPTH = float(2000)


def _find_half_depth(density_anomaly_profile, pressure, density_anomaly_surface_mean):
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
        index2 = pressures_below_sigma_0[0]
        index1 = index2 - 1
        sigma_mld = sigma_mld_min
    # only sigma above sigma_mld_max
    elif len(pressures_below_sigma_0) == 0 and len(pressures_above_sigma_0) != 0:
        index2 = pressures_above_sigma_0[0]
        index1 = index2 - 1
        sigma_mld = sigma_mld_max
    # both found, check which is closer to surface
    elif len(pressures_below_sigma_0) != 0 and len(pressures_above_sigma_0) != 0:
        if pressures_above_sigma_0[0] < pressures_below_sigma_0[0]:
            index2 = pressures_above_sigma_0[0]
            index1 = index2 - 1
            sigma_mld = sigma_mld_max
        else:
            index2 = pressures_below_sigma_0[0]
            index1 = index2 - 1
            sigma_mld = sigma_mld_min
    else:
        return MAX_DEPTH

    mld = np.interp(
        sigma_mld,
        [density_anomaly_profile[index1], density_anomaly_profile[index2]],
        [pressure[index1], pressure[index2]]
    )
    if mld <= MAX_DEPTH:
        return mld
    return MAX_DEPTH


def _find_surface_density_anomaly_mean(ds: xr.Dataset) -> None:
    """
    Find the mean density anomaly near the surface (top 10 dbar).
    TODO: TEMPORARY FUNCTION.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing density anomaly and pressure data.
    """

    ds_surface = ds.sel(PRESSURE=[2.5])
    sigma0_surface_mean = ds_surface.DENSITY_ANOMALY.mean(dim='PRESSURE')
    ds['SURFACE_DENSITY_ANOMALY_MEAN'] = sigma0_surface_mean


def _find_depth_iteration(density_anomaly_profile, pressure):
    """
    Find the mixed layer depth (hbar) from a density anomaly profile using iterative method.

    Parameters
    ----------
    density_anomaly_profile : np.ndarray
        1D array of density anomaly values at different pressures.
    pressure : np.ndarray
        1D array of pressure values corresponding to the density profile.

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

    sigma0_surface_mean = density_anomaly_profile[:1].mean()

    # land
    if np.isnan(sigma0_surface_mean):
        return -np.inf

    i = 1
    while (
        abs(density_anomaly_profile[i+1] - sigma0_surface_mean) < HBAR_DDIFF
        and i <= len(pressure)-1
    ):
        sigma0_surface_mean = (
            density_anomaly_profile[:i+1].mean()
        )
        i += 1
    mld = pressure[i]
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
        The dataset with an added variable 'MLD' representing the mixed layer depth.
    """

    if month is not None:
        ds = ds.isel(TIME=month)
    sa = gsw.SA_from_SP(
        SP=ds.SALINITY, p=ds.PRESSURE, lon=ds.LONGITUDE, lat=ds.LATITUDE
    )
    ct = gsw.CT_from_t(
        SA=sa, t=ds.TEMPERATURE, p=ds.PRESSURE
    )
    sigma0 = gsw.sigma0(SA=sa, CT=ct)
    ds['DENSITY_ANOMALY'] = sigma0
    ds['DENSITY_ANOMALY'].attrs = {"units": "kg/m^3"}

    _find_surface_density_anomaly_mean(ds)
    mld = xr.apply_ufunc(
        _find_half_depth,
        ds['DENSITY_ANOMALY'], ds['PRESSURE'], ds['SURFACE_DENSITY_ANOMALY_MEAN'],
        input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True
    )

    # mld = xr.apply_ufunc(
    #     _find_depth_iteration,
    #     ds['DENSITY_ANOMALY'], ds['PRESSURE'],
    #     input_core_dims=[['PRESSURE'], ['PRESSURE']],
    #     vectorize=True
    # )

    ds.drop_vars(["PRESSURE"])  # don't need pressure anymore
    ds['MLD'] = mld   # save to dataset

    return ds


def save_mld():
    """Save monthly mixed layer depth dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Depth-(2004-2018).nc" in logs_datasets.read():
            return

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2018).nc"
    )
    s = load_and_prepare_dataset(
        "datasets/Salinity-(2004-2018).nc"
    )
    ds = xr.merge([t, s])

    monthly_datasets = []
    for month in range(0, 180):
        monthly_datasets.append(get_monthly_mld(ds, month))
    h_ds = xr.concat(monthly_datasets, dim="TIME",
                     coords="different", compat='equals')
    h = h_ds['MLD']

    # restore attributes
    h['LATITUDE'].attrs = ds['LATITUDE'].attrs
    h['LONGITUDE'].attrs = ds['LONGITUDE'].attrs
    h.attrs['units'] = 'dbar'
    h.attrs['long_name'] = (
        'Mixed Layer Depth (h)'
    )
    h.name = 'MLD'

    save_file(
        h,
        "datasets/Mixed_Layer_Depth-(2004-2018).nc"
    )


def save_monthly_mean_mld():
    """Save the monthly mean of the mixed layer depth dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Mixed_Layer_Depth-Seasonal_Cycle_Mean.nc" in logs_datasets.read():
            return

    h = load_and_prepare_dataset(
        "datasets/Mixed_Layer_Depth-(2004-2018).nc",
    )['MLD']
    h_monthly_mean = get_monthly_mean(h)

    save_file(
        h_monthly_mean,
        "datasets/Mixed_Layer_Depth-Seasonal_Cycle_Mean.nc"
    )


def main():
    """Main function to prepare datasets from RGARGO."""

    save_mld()
    save_monthly_mean_mld()


if __name__ == "__main__":
    main()
