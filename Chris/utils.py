import xarray as xr
import pandas as pd
#import eofs
#import xeofs as xe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib
import numpy as np
from scipy.linalg import svd
#import wpca
#from wpca import EMPCA
#import wpca
#from ppca import PPCA

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
            da.sel(TIME=da['TIME'][month_num - 1::12]).mean(dim='TIME')
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
    raw_ds[variable_name + '_ANOMALY'] = anomaly_ds
    return raw_ds


def get_eof_with_nan_consideration(dataset, mask, modes, monthly_mean_ds=None, time_name="TIME", lat_name="LATITUDE",
                                   long_name="LONGITUDE", max_iterations=50, tolerance=1e-6, start_mode=0):
    # if some values in the dataset are NaN (if they are absurd e.g. infinite, set to NaN beforehand), then estimate
    # the true value of the NaN with the column mean (==mean at each point over all time) then perform EOF
    # based off various EMPCA packages, none of which really worked too well, hence the need for a homegrown solution
    time_size = dataset.sizes[time_name]
    lat_size = dataset.sizes[lat_name]
    long_size = dataset.sizes[long_name]

    ocean = mask.to_numpy().astype(bool)  # ocean mask
    X0_full = dataset.to_numpy()
    X0 = X0_full[:, ocean]  # apply ocean mask

    valid_cols = ~np.all(np.isnan(X0), axis=0)  # if the whole column is NaN, remove the column
    X0 = X0[:, valid_cols]
    ocean_valid = np.zeros_like(ocean, dtype=bool)
    ocean_valid[ocean] = valid_cols  # mask with only valid columns

    # weight by latitude to account for varying grid size
    lat = dataset[lat_name].to_numpy()
    lat_weighted = np.sqrt(np.cos(np.deg2rad(lat)))
    weight_map = np.repeat(lat_weighted[:, None], long_size, axis=1)
    weight_map = weight_map[ocean][valid_cols]  # apply ocean and valid column masks
    weight_map = np.clip(weight_map, 1e-3, None)  # prevent small values to avoid big numbers from divide

    mask_nan = np.isnan(X0)
    if monthly_mean_ds is not None:
        # get month in year
        time_vals = dataset[time_name].values
        month_in_year = np.mod(np.floor(time_vals + 0.5).astype(int), 12)
        month_in_year[month_in_year == 0] = 12
        month_da = xr.DataArray(month_in_year, coords={time_name: dataset[time_name]}, dims=(time_name,))
        monthly_mean_to_fill = monthly_mean_ds.sel({"MONTH": month_da})

        column_mean = dataset.mean(time_name, skipna=True)
        monthly_mean_to_fill = monthly_mean_to_fill.fillna(column_mean)

        # convert to numpy to fill in the nans with guesses
        monthly_mean_to_fill_np = monthly_mean_to_fill.to_numpy()
        monthly_mean_to_fill_np_mask = monthly_mean_to_fill_np[:, ocean]  # apply ocean mask
        monthly_mean_to_fill_np_mask = monthly_mean_to_fill_np_mask[:, valid_cols]

        X_with_guesses = X0.copy()
        X_with_guesses[mask_nan] = monthly_mean_to_fill_np_mask[mask_nan]  # fill missing values
    else:
        # without monthly means, just use the mean over the entire dataset (per-column mean)
        column_mean = np.nanmean(X0, axis=0)
        column_mean = np.where(np.isfinite(column_mean), column_mean, 0.0)
        X_with_guesses = np.where(mask_nan, column_mean[None, :], X0)

    # EM iterations to reconstruct incomplete values following https://ahippert.github.io/pdfs/igarss_2020.pdf
    for iteration in range(max_iterations):
        print(iteration)
        X_mean = np.nanmean(X_with_guesses, axis=0)  # get mean over time axis
        X_centered = X_with_guesses - X_mean[None, :]
        X_weighted = X_centered * weight_map[None, :]  # latitude weight

        # use SVD to estimate what the missing NaNs should be.
        U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)  # singular-value decomposition
        #X_weighted_reconstructed = (U * s) @ Vt
        #X_reconstructed = X_weighted_reconstructed / weight_map[None, :] + X_mean[None, :]  # remove weight, readd mean

        k_opt = modes  # TODO: choose the actual optimum; this is just a placeholder for now; see paper
        U_k = U[:, :k_opt]
        s_k = s[:k_opt]
        Vt_k = Vt[:k_opt, :]
        X_weighted_reconstructed_truncated = (U_k * s_k) @ Vt_k
        X_reconstructed_truncated = X_weighted_reconstructed_truncated / weight_map[None, :] + X_mean[None, :]

        X_new = X_with_guesses.copy()
        X_new[mask_nan] = X_reconstructed_truncated[mask_nan]
        if np.any(mask_nan):
            error = np.nanmean((X_new[mask_nan] - X_with_guesses[mask_nan]) ** 2)
        else:
            error = 0.0
        print(error)
        X_with_guesses = X_new
        if error < tolerance:  # if converge, stop iterating
            break

    X_mean = np.nanmean(X_with_guesses, axis=0)
    X_mean = np.where(np.isfinite(X_mean), X_mean, 0.0)
    X_centered = X_with_guesses - X_mean[None, :]
    X_weighted = X_centered * weight_map[None, :]

    U, s, Vt = np.linalg.svd(X_weighted, full_matrices=False)  # SVD again, to take only desired EOF modes

    U_modes = U[:, start_mode:modes]  # remove unwanted mods
    s_modes = s[start_mode:modes]
    Vt_modes = Vt[start_mode:modes, :]
    X_weighted_reconstructed = (U_modes * s_modes) @ Vt_modes
    X_reconstructed = X_weighted_reconstructed / weight_map[None, :] + X_mean[None, :]
    PCs = U[:, start_mode:modes] * s[start_mode:modes]
    EOFs = np.full((modes - start_mode, lat_size, long_size), np.nan)

    # reshape EOFs to have the right latitude/longitude coordinates
    all_positions = np.arange(lat_size * long_size).reshape(lat_size, long_size)
    valid_positions = all_positions[ocean].reshape(-1)[valid_cols]

    for k in range(modes - start_mode):
        eof_k = Vt_modes[k, :]
        eof_reshape = np.full((lat_size * long_size), np.nan)
        eof_reshape[valid_positions] = eof_k
        EOFs[k] = eof_reshape.reshape(lat_size, long_size)

    reconstructed_ds = np.full((time_size, lat_size, long_size), np.nan)
    reconstructed_ds[:, ocean_valid] = X_reconstructed
    smoothed_ds = xr.DataArray(reconstructed_ds, dims=dataset.dims, coords=dataset.coords)
    explained_variance = (s ** 2) / (s ** 2).sum()
    EOFs_da = xr.DataArray(EOFs, dims=("MODE", lat_name, long_name), coords={"MODE": np.arange(start_mode, modes), lat_name: dataset.coords[lat_name], long_name: dataset.coords[long_name]})

    return smoothed_ds, explained_variance, PCs, EOFs_da


def get_eof_from_ppca_py(dataset, mask, modes, monthly_mean_ds=None, time_name="TIME", lat_name="LATITUDE",
                         long_name="LONGITUDE", max_iterations=50, tolerance=1e-6):
    # use a python package rather than a home-grown solution; https://github.com/brdav/ppca-cpp
    # M. Tipping & C. Bishop. Probabilistic Principal Component Analysis. JRSS B, 1999.
    # seems not to work so well. prefer homegrown solution. long processing time and results seem weird.

    # same initial processing as homegrown way (except 'flatten' to deal with ocean masking)
    time_size = dataset.sizes[time_name]
    lat_size = dataset.sizes[lat_name]
    long_size = dataset.sizes[long_name]

    ocean = mask.to_numpy().astype(bool)  # ocean mask
    X_full = dataset.to_numpy()
    X_flat = X_full.reshape(time_size, -1)
    ocean_flat = ocean.flatten()
    ocean_indices = np.where(ocean_flat)[0]
    X = X_flat[:, ocean_flat]  # apply ocean mask

    if monthly_mean_ds is not None:
        time_vals = dataset[time_name].values
        month_in_year = np.mod(np.floor(time_vals + 0.5).astype(int), 12)
        month_in_year[month_in_year == 0] = 12
        month_da = xr.DataArray(month_in_year, coords={time_name: dataset[time_name]}, dims=(time_name,))
        monthly_mean_to_fill = monthly_mean_ds.sel({"MONTH": month_da}).to_numpy()
        monthly_mean_to_fill = monthly_mean_to_fill.reshape(time_size, -1)
        monthly_mean_to_fill = monthly_mean_to_fill[:, ocean_flat]  # apply ocean mask
        X = np.where(np.isnan(X), monthly_mean_to_fill, X)

    valid_cols = ~np.all(np.isnan(X), axis=0)  # remove column if all nan
    X_valid = X[:, valid_cols]
    ocean_valid_flat = ocean_indices[valid_cols]  # indices in full flattened grid

    # weighting by latitude
    lat = dataset[lat_name].to_numpy()
    lat_weighted = np.sqrt(np.cos(np.deg2rad(lat)))  # 1D over lat
    weight_map = np.repeat(lat_weighted[:, None], long_size, axis=1)
    weight_flat = weight_map.flatten()[ocean_flat][valid_cols]
    X_valid = X_valid * weight_flat[None, :]

    model = PPCA(n_components=modes)  # use PPCA_py package to compute
    model.fit(X_valid)

    # reconstruct principal components
    principal_components = (X_valid - model.mean_) @ model.components_.T
    X_reconstructed = principal_components @ model.components_ + model.mean_
    X_reconstructed = X_reconstructed / weight_flat[None, :]

    reconstructed_ds = np.full((time_size, lat_size * long_size), np.nan)
    reconstructed_ds[:, ocean_valid_flat] = X_reconstructed
    reconstructed_ds = reconstructed_ds.reshape((time_size, lat_size, long_size))
    reconstructed_ds = xr.DataArray(reconstructed_ds, dims=dataset.dims, coords=dataset.coords)

    EOFs = model.components_
    explained_variance = model.explained_variance_

    return reconstructed_ds, EOFs, model, explained_variance


def get_eof(dataset, modes, mask=None, clean_nan=False):
    if mask is not None:
        ocean = mask.to_numpy().astype(bool)
        dataset = dataset.where(ocean)
    if clean_nan:
        dataset = dataset.dropna(dim="LATITUDE", how="any").dropna(dim="LONGITUDE", how="any")
        #dataset = dataset.fillna(dataset.mean(dim="TIME"))

    model = xe.single.EOF(n_modes=modes)
    model.fit(dataset, dim="TIME")
    components = model.components()  # spatial EOFs
    scores = model.scores()  # PC time series
    explained_variance = model.explained_variance_ratio()

    # bootstrapping for significant modes, from https://xeofs.readthedocs.io/en/latest/content/user_guide/auto_examples/4validation/plot_bootstrap.html#sphx-glr-content-user-guide-auto-examples-4validation-plot-bootstrap-py
    # n_boot = 50
    # bs = xe.validation.EOFBootstrapper(n_bootstraps=n_boot)
    # bs.fit(model)
    # bs_expvar = bs.explained_variance()
    # ci_expvar = bs_expvar.quantile([0.025, 0.975], "n")  # 95% confidence intervals
    # q025 = ci_expvar.sel(quantile=0.025)
    # q975 = ci_expvar.sel(quantile=0.975)
    # is_significant = q025 - q975.shift({"mode": -1}) > 0
    # n_significant_modes = (
    #     is_significant.where(is_significant is True).cumsum(skipna=False).max().fillna(0)
    # )
    # print("{:} modes are significant at alpha=0.05".format(n_significant_modes.values))

    #reconstructed = model.inverse_transform(scores)
    return components, explained_variance, scores


def make_movie(dataset, vmin, vmax, colorbar_label=None, ENSO_ds=None, savepath=None, cmap='RdBu_r'):
    times = dataset.TIME.values

    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    pcolormesh = ax.pcolormesh(dataset.LONGITUDE.values, dataset.LATITUDE.values,
                               dataset.isel(TIME=0), cmap=cmap)
    title = ax.set_title(f'Time = {times[0]}')

    cbar = plt.colorbar(pcolormesh, ax=ax, label=colorbar_label)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    def update(frame):
        month = int((times[frame] + 0.5) % 12)
        if month == 0:
            month = 12
        year = 2004 + int((times[frame]) / 12)
        pcolormesh.set_array(dataset.isel(TIME=frame).values.ravel())
        #pcolormesh.set_clim(vmin=float(model_anomaly_ds.isel(TIME=frame).min()), vmax=float(model_anomaly_ds.isel(TIME=frame).max()))
        pcolormesh.set_clim(vmin=vmin, vmax=vmax)
        cbar.update_normal(pcolormesh)
        if (ENSO_ds is not None):
            enso_index = ENSO_ds.isel(time=frame).value.values.item()
            title.set_text(f'Year: {year}; Month: {month}; ENSO index: {round(enso_index, 4)}')
        else:
            title.set_text(f'Year: {year}; Month: {month}')
        return [pcolormesh, title]

    animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
    if savepath is not None:
        animation.save(savepath, fps=2, dpi=200)
    plt.show()


def remove_empty_attributes(dataset):
    for variable in dataset.variables:
        attributes = dataset[variable].attrs
        for key, value in list(attributes.items()):
            if value is None:
                attributes[key] = ""
    return dataset


def compute_gradient_lat(
        field: xr.DataArray,
        R=6.4e6) -> xr.DataArray:
    """
    Compute the gradient of the field with respect to latitude and longitude
    > taken directly from Jason/Julia's repository
    """

    # Convert degrees to radians for calculation
    lat_rad = np.deg2rad(field['LATITUDE'])
    # lon_rad = np.deg2rad(field['Longitude'])

    # Calculate the spacing in meters
    dlat = (np.pi / 180) * R

    # Focusing on lat for now

    # Masks for neighbouring points
    # This is needed to identify valid neighbouring points for gradient calculation
    valid = field.notnull()  # Creates a BOOL array where valid data (ocean) is set to True
    has_prev = valid.shift(LATITUDE=1, fill_value=False)  # Cell above is ocean
    has_next = valid.shift(LATITUDE=-1, fill_value=False)  # Cell below is ocean
    has_prev2 = valid.shift(LATITUDE=2, fill_value=False)  # Two cells above is ocean
    has_next2 = valid.shift(LATITUDE=-2, fill_value=False)  # Two cells below is ocean

    # We can then define our interior points
    interior = valid & has_prev & has_next  # Maps out the interior points
    # Define the edge points of our interior
    start = valid & ~has_prev
    end = valid & ~has_next

    # Now we define start/end by how many valid neighbours exist ahead/behind them
    start_run_3pt = start & has_next & has_next2  # Starting point for 2nd order forward difference
    end_run_3pt = end & has_prev & has_prev2  # End point for 2nd order backward difference
    start_run_2pt = start & has_next & ~has_next2  # Starting point for 1st order forward difference
    end_run_2pt = end & has_prev & ~has_prev2  # End point for 1st order backward difference
    single = start & ~has_next  # Point with no valid neighbours (will leave as NaN)

    # Precompute shifted fields for vectorised operations
    f_prev = field.shift(LATITUDE=1)
    f_next = field.shift(LATITUDE=-1)
    f_prev2 = field.shift(LATITUDE=2)
    f_next2 = field.shift(LATITUDE=-2)

    # Initialise gradient array with NaNs
    grad = xr.full_like(field, np.nan)

    # Central difference for interior points (2nd order)
    grad = grad.where(~interior, ((f_next - f_prev) / (2 * dlat)))

    # Start of runs (forward differences)
    grad = grad.where(~start_run_3pt, ((-3 * field + 4 * f_next - f_next2) / (2 * dlat)))
    grad = grad.where(~start_run_2pt, ((f_next - field) / dlat))

    # End of runs (backward differences)
    grad = grad.where(~end_run_3pt, ((3 * field - 4 * f_prev + f_prev2) / (2 * dlat)))
    grad = grad.where(~end_run_2pt, ((field - f_prev) / dlat))

    # Single points left as NaN
    return grad


def compute_gradient_lon(
        field: xr.DataArray,
        R=6.4e6) -> xr.DataArray:
    """
    Compute the gradient of the field with respect to latitude and longitude
    > taken directly from Jason/Julia's repository
    """

    # Convert degrees to radians for calculation
    lat_rad = np.deg2rad(field['LATITUDE'])
    # lon_rad = np.deg2rad(field['Longitude'])

    # Calculate the spacing in meters
    dlon = (np.pi / 180) * R * np.cos(lat_rad)
    dlon = xr.DataArray(dlon, coords=field['LATITUDE'].coords, dims=['LATITUDE'])

    # Masks for neighbouring points
    # This is needed to identify valid neighbouring points for gradient calculation
    valid = field.notnull()  # Creates a BOOL array where valid data (ocean) is set to True
    has_west = valid.roll(LONGITUDE=1, roll_coords=False)  # Cell above is ocean
    has_east = valid.roll(LONGITUDE=-1, roll_coords=False)  # Cell below is ocean
    has_west2 = valid.roll(LONGITUDE=2, roll_coords=False)  # Two cells above is ocean
    has_east2 = valid.roll(LONGITUDE=-2, roll_coords=False)  # Two cells below is ocean

    # We can then define our interior points
    interior = valid & has_west & has_east  # Maps out the interior points
    # Define the edge points of our interior
    start = valid & ~has_west
    end = valid & ~has_east

    # Now we define start/end by how many valid neighbours exist ahead/behind them
    start_run_3pt = start & has_east & has_east2  # Starting point for 2nd order forward difference
    end_run_3pt = end & has_west & has_west2  # End point for 2nd order backward difference
    start_run_2pt = start & has_east & ~has_east2  # Starting point for 1st order forward difference
    end_run_2pt = end & has_west & ~has_west2  # End point for 1st order backward difference
    single = start & ~has_east  # Point with no valid neighbours (will leave as NaN)

    # Precompute shifted fields for vectorised operations
    f_prev = field.roll(LONGITUDE=1, roll_coords=False)
    f_next = field.roll(LONGITUDE=-1, roll_coords=False)
    f_prev2 = field.roll(LONGITUDE=2, roll_coords=False)
    f_next2 = field.roll(LONGITUDE=-2, roll_coords=False)

    # Initialise gradient array with NaNs
    grad = xr.full_like(field, np.nan)

    # Central difference for interior points (2nd order)
    grad = grad.where(~interior, ((f_next - f_prev) / (2 * dlon)))

    # Start of runs (forward differences)
    grad = grad.where(~start_run_3pt, ((-3 * field + 4 * f_next - f_next2) / (2 * dlon)))
    grad = grad.where(~start_run_2pt, ((f_next - field) / dlon))

    # End of runs (backward differences)
    grad = grad.where(~end_run_3pt, ((3 * field - 4 * f_prev + f_prev2) / (2 * dlon)))
    grad = grad.where(~end_run_2pt, ((field - f_prev) / dlon))

    # Single points left as NaN
    return grad


def get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN, INCLUDE_ENTRAINMENT, INCLUDE_GEOSTROPHIC, USE_DOWNLOADED_SSH=False, gamma0=10, INCLUDE_GEOSTROPHIC_DISPLACEMENT=False, INCLUDE_EKMAN_MEAN_ADVECTION=False, OTHER_MLD=False, MAX_GRAD_TSUB=False, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False):
    save_name = ""
    if INCLUDE_SURFACE:
        save_name = save_name + "1"
    else:
        save_name = save_name + "0"
    if INCLUDE_EKMAN:
        save_name = save_name + "1"
    else:
        save_name = save_name + "0"
    if INCLUDE_ENTRAINMENT:
        save_name = save_name + "1"
    else:
        save_name = save_name + "0"
    if INCLUDE_GEOSTROPHIC:
        save_name = save_name + "1"
        if INCLUDE_GEOSTROPHIC_DISPLACEMENT:
            save_name = save_name + "_geostrophiccurrent"
        if USE_DOWNLOADED_SSH:
            save_name = save_name + "_downloadedSSH"
    else:
        save_name = save_name + "0"
    if INCLUDE_EKMAN_MEAN_ADVECTION:
        save_name = save_name + "_ekmanmeanadv"
    if OTHER_MLD:
        save_name = save_name + "_otherMLD"
    if MAX_GRAD_TSUB:
        save_name = save_name + "_maxgrad"
    if ENTRAINMENT_VEL_ANOM_FORC:
        save_name = save_name + "_entrainvelanomforcing"
    if LOG_ENTRAINMENT_VELOCITY:
        save_name = save_name + "_logentrainmentvelocity"
    if gamma0 != 10.0:
        save_name += "_gamma" + str(gamma0)
    return save_name


def coriolis_parameter(lat):
    omega = 2 * np.pi / (24 * 3600)
    phi_rad = np.deg2rad(lat)
    f = 2 * omega * np.sin(phi_rad)
    f = xr.DataArray(f, coords={'LATITUDE': lat}, dims=['LATITUDE'])
    f.attrs['units'] = 's^-1'
    return f

def get_month_from_time(time):
    month = (time + 0.5) % 12
    if month == 0:
        month = 12.0
    return month
