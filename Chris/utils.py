import xarray as xr
import pandas as pd
#import eofs
import xeofs as xe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np

matplotlib.use('TkAgg')


def _time_standard(ds: xr.Dataset, mid_month: bool = False) -> xr.Dataset:
    """
    Add a standard netCDF time coordinate to the dataset.
    Not finished becuase data not linked to this new time coordinate yet.
    Original:
        'TIME' - axis T
    New:
        'time' - axis t, calendar 360_day

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset with time variable to decode.
    mid_month: bool
        Default is False.
        If True, set time to the middle of the month.
        If False, set time to the start of the month.

    Returns
    -------
    xarray.Dataset
        Dataset with new 'time' coordinate.

    Raises
    ------
    ValueError
        If mid_month is not True or False.
    """
    _, reference_date = ds['TIME'].attrs['units'].split('since')
    if mid_month is False:
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['TIME'], freq='MS')
    elif mid_month is True:
        ds['time'] = (pd.date_range(start=reference_date, periods=ds.sizes['TIME'], freq='MS')
                      + pd.DateOffset(days=14))
    else:
        raise ValueError("mid_month must be True or False.")
    ds['time'].attrs['calendar'] = '360_day'
    ds['time'].attrs['axis'] = 't'
    return ds


def _longitude_180(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert longitude from [0, 360] to [-180, 180].

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset with 'LONGITUDE' coordinate.

    Returns
    -------
    xarray.Dataset
        Dataset with 'LONGITUDE' in [-180, 180] and sorted.
    """
    lon_atrib = ds.coords['LONGITUDE'].attrs
    ds['LONGITUDE'] = ((ds['LONGITUDE'] + 180) % 360) - 180
    ds = ds.sortby(ds['LONGITUDE'])
    ds['LONGITUDE'].attrs.update(lon_atrib)
    ds['LONGITUDE'].attrs['modulo'] = 180
    return ds


def load_and_prepare_dataset(
        filepath: str,
        time_standard: bool = False,
        time_standard_mid_month: bool = False,
        longitude_180: bool = True
) -> xr.Dataset | None:
    """
    Load, standardize time, and convert longitude for RG-ARGO dataset.

    Parameters
    ----------
    filepath: str
        Path to the RG-ARGO netCDF file.
    time_standard: bool
        Default is False.
        If True, standardize the time coordinate.
    time_standard_mid_month: bool
        Default is False.
        If True, standardize the time coordinate to the middle of the month.
    longitude_180: bool
        Default is True.
        If True, convert longitude to [-180, 180].

    Returns
    -------
    xarray.Dataset or None
        The processed dataset, or None if loading fails.
    """
    try:
        with xr.open_dataset(filepath, decode_times=False) as ds:
            if time_standard:
                ds = _time_standard(ds)
            if time_standard_mid_month:
                ds = _time_standard(ds, mid_month=True)
            if longitude_180:
                ds = _longitude_180(ds)
            return ds
    except (OSError, ValueError, KeyError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3,
    'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9,
    'Oct': 10, 'Nov': 11, 'Dec': 12
}


def get_monthly_mean(
    da: xr.DataArray,
) -> xr.DataArray:
    """
    Get the monthly mean of the DataArray.

    Parameters
    ----------
    da: xarray.DataArray
        The DataArray to process.

    Returns
    -------
    xarray.DataArray
        The monthly mean DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """
    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    monthly_means = []
    for _, month_num in MONTHS.items():
        monthly_means.append(
            da.sel(TIME=da['TIME'][month_num-1::12]).mean(dim='TIME')
        )
    monthly_mean_da = xr.concat(monthly_means, dim='MONTH')
    monthly_mean_da = monthly_mean_da.assign_coords(MONTH=list(MONTHS.values()))
    monthly_mean_da['MONTH'].attrs['units'] = 'month'
    monthly_mean_da['MONTH'].attrs['axis'] = 'M'
    monthly_mean_da.attrs['units'] = da.attrs.get('units')
    #monthly_mean_da.attrs['long_name'] = f"Seasonal Cycle Mean of {da.attrs.get('long_name')}"
    #monthly_mean_da.name = f"MONTHLY_MEAN_{da.name}"
    return monthly_mean_da


def get_anomaly(raw_ds, variable_name, monthly_mean):
    #raw_ds = raw_ds[variable_name]
    anomalies = []
    for month in raw_ds.coords['TIME']:
        month = month.values
        compare_to_month_mean = int((month + 0.5) % 12)
        if compare_to_month_mean == 0:
            compare_to_month_mean = 12
        month_mean = monthly_mean.sel(MONTH=compare_to_month_mean)
        anomaly = raw_ds.sel(TIME=month)[variable_name] - month_mean
        anomalies.append(anomaly)
    anomaly_ds = xr.concat(anomalies, "TIME")
    anomaly_ds = anomaly_ds.drop_vars("MONTH")
    raw_ds[variable_name+'_ANOMALY'] = anomaly_ds
    return raw_ds


def get_eof(dataset, mask, modes=3, time_name="TIME", lat_name="LATITUDE", long_name="LONGITUDE", max_iterations=50,
            tolerance=1e-4):
    time_size = dataset.sizes[time_name]
    lat_size = dataset.sizes[lat_name]
    long_size = dataset.sizes[long_name]

    # apply mask
    ocean = mask.to_numpy().astype(bool)
    points = ocean.sum()  # number of ocean grid cells

    # weight by area
    lat = dataset[lat_name].to_numpy()
    lat_weighted = np.sqrt(np.cos(np.deg2rad(lat)))
    lat_weighted = np.clip(lat_weighted, 1e-6, None)
    weight_map = np.repeat(lat_weighted[:, None], long_size, axis=1)
    weight_map = weight_map[ocean]  # apply mask
    X0_full = dataset.to_numpy()
    X0 = X0_full[:, ocean]

    valid_cols = ~np.all(np.isnan(X0), axis=0)
    X0 = X0[:, valid_cols]
    weight_map = weight_map[valid_cols]

    ocean_valid = np.zeros_like(ocean, dtype=bool)
    ocean_valid[ocean] = valid_cols  # True for ocean points we kept

    # guess the value of NaN positions using the mean of each column
    mask_nan = np.isnan(X0)
    column_mean = np.nanmean(X0, axis=0)
    column_mean = np.where(np.isfinite(column_mean), column_mean, 0.0)

    X_with_guesses = np.where(mask_nan, column_mean[None, :], X0)
    prev = X_with_guesses.copy()
    weight_map = np.clip(weight_map, 1e-3, None)

    # iterative EOF
    for iteration in range(max_iterations):
        print(iteration)
        X_mean = np.nanmean(X_with_guesses, axis=0)
        X_centered = X_with_guesses - X_mean
        X_weighted = X_centered * weight_map
        X_weighted = np.nan_to_num(X_weighted, nan=0.0, posinf=0.0, neginf=0.0)
        U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)
        k = modes

        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]

        X_weighted_reconstructed = (U_k * s_k) @ Vt_k
        X_reconstructed = X_weighted_reconstructed / weight_map + X_mean

        X_new = X_with_guesses.copy()  # update only NaN ocean points
        X_new[mask_nan] = X_reconstructed[mask_nan]

        error = np.nanmean((X_new[mask_nan] - prev[mask_nan]) ** 2)
        print(error)
        if error < tolerance:  # if converged, stop iterating
            break

        prev = X_with_guesses
        X_with_guesses = X_new

    reconstructed_ds = np.full((time_size, lat_size, long_size), np.nan)
    reconstructed_ds[:, ocean_valid] = X_with_guesses

    smoothed_ds = xr.DataArray(reconstructed_ds, dims=dataset.dims, coords=dataset.coords)

    explained_variance = (s ** 2) / (s ** 2).sum()

    return smoothed_ds, explained_variance


def get_simple_eof(dataset, mask=None, modes=3, clean_nan=False):
    if mask is not None:
        ocean = mask.to_numpy().astype(bool)
        dataset = dataset.where(ocean)
    if clean_nan:
        dataset = dataset.dropna(dim="LATITUDE", how="all").dropna(dim="LONGITUDE", how="all")
        dataset = dataset.fillna(dataset.mean(dim="TIME"))

    model = xe.single.EOF(n_modes=modes)
    model.fit(dataset, dim="TIME")
    components = model.components()  # spatial EOFs
    scores = model.scores()  # PC time series
    explained_variance = model.explained_variance_ratio()
    reconstructed = model.inverse_transform(scores)  # smoothed reconstruction using first 3 modes
    return reconstructed, explained_variance


def make_movie(dataset):
    times = dataset.TIME.values

    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    pcolormesh = ax.pcolormesh(dataset.LONGITUDE.values, dataset.LATITUDE.values,
                               dataset.isel(TIME=0), cmap='RdBu_r')
    title = ax.set_title(f'Time = {times[0]}')

    cbar = plt.colorbar(pcolormesh, ax=ax, label='Modelled anomaly from surface heat flux')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    def update(frame):
        month = int((times[frame] + 0.5) % 12)
        if month == 0:
            month = 12
        year = 2004 + int((times[frame]) / 12)
        pcolormesh.set_array(dataset.isel(TIME=frame).values.ravel())
        #pcolormesh.set_clim(vmin=float(model_anomaly_ds.isel(TIME=frame).min()), vmax=float(model_anomaly_ds.isel(TIME=frame).max()))
        pcolormesh.set_clim(vmin=-10, vmax=10)
        cbar.update_normal(pcolormesh)
        title.set_text(f'Year: {year}; Month: {month}')
        return [pcolormesh, title]

    animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
    plt.show()
