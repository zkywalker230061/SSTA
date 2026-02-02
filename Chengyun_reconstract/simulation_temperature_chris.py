"""
Simulation of mixed-layer temperature anomalies using various models.

Chengyun Zhu and Chris O'Sullivan

2026-2-2
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

from utilities import load_and_prepare_dataset
# from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = False
INCLUDE_GEOSTROPHIC = True

# first method for geostrophic: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-3039/egusphere-2025-3039.pdf
CLEAN_CHRIS_PREV_CUR = True        # only really useful when entrainment is turned on


HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "datasets/Simulation-Surface_Heat_Flux-(2004-2018).nc"
# ['avg_ie'] and ['avg_tprate']
EKMAN_ANOMALY_DATA_PATH = "datasets/Simulation-Ekman_Heat_Flux-(2004-2018).nc"
# [ANOMALY_EKMAN_HEAT_FLUX]
TEMP_DATA_PATH = "datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "datasets/Mixed_Layer_Depth-(2004-2018).nc"
# ['MLD']
H_BAR_DATA_PATH = "datasets/Mixed_Layer_Depth-Seasonal_Mean.nc"
# ['MONTHLY_MEAN_MLD']
ENTRAINMENT_VEL_DATA_PATH = "datasets/Mixed_Layer_Entrainment_Velocity-(2004-2018).nc"
# ['w_e']
T_SUB_ANOMALY_DATA_PATH = "datasets/Sub_Layer_Temperature_Anomalies-(2004-2018).nc"
# ['ANOMALY_SUB_TEMPERATURE']
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2018).nc"
# ['ANOMALY_GEOSTROPHIC_HEAT_FLUX']

# ------------------------------------------------------------------
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


def make_movie(dataset, vmin, vmax, colorbar_label=None, ENSO_ds=None):
    times = dataset.TIME.values

    fig, ax = plt.subplots()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines()
    pcolormesh = ax.pcolormesh(dataset.LONGITUDE.values, dataset.LATITUDE.values,
                               dataset.isel(TIME=0), cmap='RdBu_r')
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
    plt.show()


def remove_empty_attributes(dataset):
    for variable in dataset.variables:
        attributes = dataset[variable].attrs
        for key, value in list(attributes.items()):
            if value is None:
                attributes[key] = ""
    return dataset


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
# ------------------------------------------------------------------


rho_0 = 1025.0
c_0 = 4100
gamma_0 = 30
g = 9.81

temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)

t_m_a = load_and_prepare_dataset('datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc')
t_m_a = t_m_a.drop_vars('MONTH')

heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH)
heat_flux_ds['NET_HEAT_FLUX_ANOMALY'] = heat_flux_ds['ANOMALY_avg_slhtf'] + heat_flux_ds['ANOMALY_avg_ishf'] + \
                                         heat_flux_ds['ANOMALY_avg_snswrf'] + heat_flux_ds['ANOMALY_avg_snlwrf']
surface_flux_da = heat_flux_ds['NET_HEAT_FLUX_ANOMALY']

ekman_anomaly_ds = load_and_prepare_dataset(EKMAN_ANOMALY_DATA_PATH)
ekman_anomaly_da = ekman_anomaly_ds['ANOMALY_EKMAN_HEAT_FLUX']
ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)
ekman_anomaly_da = ekman_anomaly_da.where(
        (ekman_anomaly_da['LATITUDE'] > 5) | (ekman_anomaly_da['LATITUDE'] < -5), 0
    )

geostrophic_anomaly_ds = load_and_prepare_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH)
geostrophic_anomaly_da = geostrophic_anomaly_ds["ANOMALY_GEOSTROPHIC_HEAT_FLUX"]

hbar_ds = load_and_prepare_dataset(H_BAR_DATA_PATH)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

t_sub_anomaly_ds = load_and_prepare_dataset(T_SUB_ANOMALY_DATA_PATH)
t_sub_anomaly_da = t_sub_anomaly_ds["ANOMALY_SUB_TEMPERATURE"]

entrainment_vel_ds = load_and_prepare_dataset(ENTRAINMENT_VEL_DATA_PATH)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc"
)['MONTHLY_MEAN_w_e']
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']



def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60


delta_t = month_to_second(1)

# initialise lists for temperature anomalies for each model
chris_prev_cur_model_anomalies = []
chris_mean_k_model_anomalies = []
chris_prev_k_model_anomalies = []
chris_capped_exponent_model_anomalies = []
explicit_model_anomalies = []
implicit_model_anomalies = []
semi_implicit_model_anomalies = []

# initialise lists for entrainment fluxes for each model; for categorising each component
entrainment_fluxes_prev_cur = []
entrainment_fluxes_mean_k = []
entrainment_fluxes_prev_k = []
entrainment_fluxes_capped_exponent = []
entrainment_fluxes_explicit = []
entrainment_fluxes_implicit = []
entrainment_fluxes_semi_implicit = []

added_baseline = False
for month in heat_flux_ds.TIME.values:
    # find the previous and current month from 1 to 12 to access the monthly-averaged data (hbar, entrainment vel.)
    prev_month = month - 1
    month_in_year = int((month + 0.5) % 12)
    if month_in_year == 0:
        month_in_year = 12
    prev_month_in_year = month_in_year - 1
    if prev_month_in_year == 0:
        prev_month_in_year = 12

    if not added_baseline:  # just adds the baseline of a whole bunch of zero
        base = t_m_a.sel(TIME=month)['ANOMALY_ML_TEMPERATURE'] - \
               t_m_a.sel(TIME=month)['ANOMALY_ML_TEMPERATURE']
        base = base.expand_dims(TIME=[month])
        chris_prev_cur_model_anomalies.append(base)
        chris_mean_k_model_anomalies.append(base)
        chris_prev_k_model_anomalies.append(base)
        chris_capped_exponent_model_anomalies.append(base)
        explicit_model_anomalies.append(base)
        implicit_model_anomalies.append(base)
        semi_implicit_model_anomalies.append(base)
        added_baseline = True

    else:
        # store previous readings Tm(n-1)
        prev_chris_prev_cur_tm_anom = chris_prev_cur_model_anomalies[-1].isel(TIME=-1)
        prev_chris_mean_k_tm_anom = chris_mean_k_model_anomalies[-1].isel(TIME=-1)
        prev_chris_prev_k_tm_anom = chris_prev_k_model_anomalies[-1].isel(TIME=-1)
        prev_chris_capped_exponent_k_tm_anom = chris_capped_exponent_model_anomalies[-1].isel(TIME=-1)
        prev_explicit_k_tm_anom = explicit_model_anomalies[-1].isel(TIME=-1)
        prev_implicit_k_tm_anom = implicit_model_anomalies[-1].isel(TIME=-1)
        prev_semi_implicit_k_tm_anom = semi_implicit_model_anomalies[-1].isel(TIME=-1)

        # get previous data

        # OLD METHOD FOR GEOSTROPHIC
        # if INCLUDE_GEOSTROPHIC:
        #     prev_tsub_anom_at_cur_loc = t_sub_da.sel(TIME=prev_month)
        #     alpha = g / f * sea_surface_grad_ds['sla_anomaly_grad_long']
        #     beta = g / f * sea_surface_grad_ds['sla_anomaly_grad_lat']
        #     back_x = prev_tsub_anom_at_cur_loc['LONGITUDE'] + alpha * month_to_second(1)
        #     back_y = prev_tsub_anom_at_cur_loc['LATITUDE'] - beta * month_to_second(1)
        #     prev_tsub_anom = prev_tsub_anom_at_cur_loc.interp(LONGITUDE=back_x, LATITUDE=back_y)
        # else:
        #     prev_tsub_anom = t_sub_da.sel(TIME=prev_month)

        prev_tsub_anom = t_sub_anomaly_da.sel(TIME=prev_month)
        prev_heat_flux_anom = surface_flux_da.sel(TIME=prev_month)
        prev_ekman_anom = ekman_anomaly_da.sel(TIME=prev_month)
        prev_entrainment_vel = entrainment_vel_da.sel(MONTH=prev_month_in_year)
        prev_geo_anom = geostrophic_anomaly_da.sel(TIME=prev_month)
        prev_hbar = hbar_da.sel(MONTH=prev_month_in_year)

        # get current data
        cur_tsub_anom = t_sub_anomaly_da.sel(TIME=month)
        cur_heat_flux_anom = surface_flux_da.sel(TIME=month)
        cur_ekman_anom = ekman_anomaly_da.sel(TIME=month)
        cur_entrainment_vel = entrainment_vel_da.sel(MONTH=month_in_year)
        cur_geo_anom = geostrophic_anomaly_da.sel(TIME=month)
        cur_hbar = hbar_da.sel(MONTH=month_in_year)

        # generate the right dataset depending on whether surface flux and/or Ekman and/or geostrophic terms are desired
        if INCLUDE_SURFACE and INCLUDE_EKMAN:
            cur_surf_ek = cur_heat_flux_anom + cur_ekman_anom
            prev_surf_ek = prev_heat_flux_anom + prev_ekman_anom

        elif INCLUDE_SURFACE:
            cur_surf_ek = cur_heat_flux_anom
            prev_surf_ek = prev_heat_flux_anom

        elif INCLUDE_EKMAN:
            cur_surf_ek = cur_ekman_anom
            prev_surf_ek = prev_ekman_anom

        else:       # just a way to get a zero dataset
            cur_surf_ek = cur_ekman_anom - cur_ekman_anom
            prev_surf_ek = prev_ekman_anom - prev_ekman_anom

        if INCLUDE_GEOSTROPHIC:
            cur_surf_ek = cur_surf_ek + cur_geo_anom
            prev_surf_ek = prev_surf_ek + prev_geo_anom

        if INCLUDE_ENTRAINMENT:
            cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar) + cur_entrainment_vel / cur_hbar * cur_tsub_anom
            cur_a = cur_entrainment_vel / cur_hbar + gamma_0 / (rho_0 * c_0 * cur_hbar)
            cur_k = (gamma_0 / (rho_0 * c_0) + cur_entrainment_vel) / cur_hbar

            prev_b = prev_surf_ek / (rho_0 * c_0 * prev_hbar) + prev_entrainment_vel / prev_hbar * prev_tsub_anom
            prev_a = prev_entrainment_vel / prev_hbar + gamma_0 / (rho_0 * c_0 * prev_hbar)
            prev_k = (gamma_0 / (rho_0 * c_0) + prev_entrainment_vel) / prev_hbar
        else:
            cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar)
            cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)
            cur_k = cur_a

            prev_b = prev_surf_ek / (rho_0 * c_0 * prev_hbar)
            prev_a = gamma_0 / (rho_0 * c_0 * prev_hbar)
            prev_k = prev_a

        exponent_prev_cur = prev_k * month_to_second(prev_month) - cur_k * month_to_second(month)
        exponent_mean_k = -0.5 * (prev_k + cur_k) * delta_t
        exponent_prev_k = prev_k * month_to_second(prev_month) - prev_k * month_to_second(month)
        exponent_capped = exponent_prev_cur.where(exponent_prev_cur <= 0, 0)

        # update anomalies
        if INCLUDE_ENTRAINMENT:
            cur_chris_prev_cur_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_prev_cur_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_prev_cur)
            cur_chris_mean_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_mean_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_mean_k)
            cur_chris_prev_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_prev_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_prev_k)
            cur_chris_capped_exponent_k_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_surf_ek / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_chris_capped_exponent_k_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_surf_ek / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent_capped)
        else:
            cur_chris_prev_cur_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_prev_cur_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_prev_cur)
            cur_chris_mean_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_mean_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_mean_k)
            cur_chris_prev_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_prev_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_prev_k)
            cur_chris_capped_exponent_k_tm_anom = cur_surf_ek / gamma_0 + (prev_chris_capped_exponent_k_tm_anom - prev_surf_ek / gamma_0) * np.exp(exponent_capped)

        cur_explicit_k_tm_anom = prev_explicit_k_tm_anom + delta_t * (prev_b - prev_a * prev_explicit_k_tm_anom)
        cur_implicit_k_tm_anom = (prev_implicit_k_tm_anom + delta_t * cur_b) / (1 + delta_t * cur_a)
        cur_semi_implicit_k_tm_anom = (prev_semi_implicit_k_tm_anom + delta_t * prev_b) / (1 + delta_t * cur_a)

        # reformat and save each model
        cur_chris_prev_cur_tm_anom = cur_chris_prev_cur_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_prev_cur_tm_anom = cur_chris_prev_cur_tm_anom.expand_dims(TIME=[month])
        chris_prev_cur_model_anomalies.append(cur_chris_prev_cur_tm_anom)

        cur_chris_mean_k_tm_anom = cur_chris_mean_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_mean_k_tm_anom = cur_chris_mean_k_tm_anom.expand_dims(TIME=[month])
        chris_mean_k_model_anomalies.append(cur_chris_mean_k_tm_anom)

        cur_chris_prev_k_tm_anom = cur_chris_prev_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_prev_k_tm_anom = cur_chris_prev_k_tm_anom.expand_dims(TIME=[month])
        chris_prev_k_model_anomalies.append(cur_chris_prev_k_tm_anom)

        cur_chris_capped_exponent_k_tm_anom = cur_chris_capped_exponent_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_chris_capped_exponent_k_tm_anom = cur_chris_capped_exponent_k_tm_anom.expand_dims(TIME=[month])
        chris_capped_exponent_model_anomalies.append(cur_chris_capped_exponent_k_tm_anom)

        cur_explicit_k_tm_anom = cur_explicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_explicit_k_tm_anom = cur_explicit_k_tm_anom.expand_dims(TIME=[month])
        explicit_model_anomalies.append(cur_explicit_k_tm_anom)

        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.expand_dims(TIME=[month])
        implicit_model_anomalies.append(cur_implicit_k_tm_anom)

        cur_semi_implicit_k_tm_anom = cur_semi_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_semi_implicit_k_tm_anom = cur_semi_implicit_k_tm_anom.expand_dims(TIME=[month])
        semi_implicit_model_anomalies.append(cur_semi_implicit_k_tm_anom)

        # get entrainment flux components; for categorising each component
        if INCLUDE_ENTRAINMENT:
            entrainment_flux_prev_cur = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_prev_cur_tm_anom)
            entrainment_fluxes_prev_cur.append(entrainment_flux_prev_cur)

            entrainment_flux_mean_k = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_mean_k_tm_anom)
            entrainment_fluxes_mean_k.append(entrainment_flux_mean_k)

            entrainment_flux_prev_k = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_prev_k_tm_anom)
            entrainment_fluxes_prev_k.append(entrainment_flux_prev_k)

            entrainment_flux_capped_exponent = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_chris_capped_exponent_k_tm_anom)
            entrainment_fluxes_capped_exponent.append(entrainment_flux_capped_exponent)

            entrainment_flux_explicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_explicit_k_tm_anom)
            entrainment_fluxes_explicit.append(entrainment_flux_explicit)

            entrainment_flux_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_implicit_k_tm_anom)
            entrainment_fluxes_implicit.append(entrainment_flux_implicit)

            entrainment_flux_semi_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_semi_implicit_k_tm_anom)
            entrainment_fluxes_semi_implicit.append(entrainment_flux_semi_implicit)


# concatenate anomalies into a ds
chris_prev_cur_model_anomaly_ds = xr.concat(chris_prev_cur_model_anomalies, 'TIME')
chris_mean_k_model_anomaly_ds = xr.concat(chris_mean_k_model_anomalies, 'TIME')
chris_prev_k_model_anomaly_ds = xr.concat(chris_prev_k_model_anomalies, 'TIME')
chris_capped_exponent_model_anomaly_ds = xr.concat(chris_capped_exponent_model_anomalies, 'TIME')
explicit_model_anomaly_ds = xr.concat(explicit_model_anomalies, 'TIME')
implicit_model_anomaly_ds = xr.concat(implicit_model_anomalies, 'TIME')
semi_implicit_model_anomaly_ds = xr.concat(semi_implicit_model_anomalies, 'TIME')

# rename all models
chris_prev_cur_model_anomaly_ds = chris_prev_cur_model_anomaly_ds.rename("CHRIS_PREV_CUR")
chris_mean_k_model_anomaly_ds = chris_mean_k_model_anomaly_ds.rename("CHRIS_MEAN_K")
chris_prev_k_model_anomaly_ds = chris_prev_k_model_anomaly_ds.rename("CHRIS_PREV_K")
chris_capped_exponent_model_anomaly_ds = chris_capped_exponent_model_anomaly_ds.rename("CHRIS_CAPPED_EXPONENT")
explicit_model_anomaly_ds = explicit_model_anomaly_ds.rename("EXPLICIT")
implicit_model_anomaly_ds = implicit_model_anomaly_ds.rename("IMPLICIT")
semi_implicit_model_anomaly_ds = semi_implicit_model_anomaly_ds.rename("SEMI_IMPLICIT")

# combine to a single ds
all_anomalies_ds = xr.merge([chris_prev_cur_model_anomaly_ds, chris_mean_k_model_anomaly_ds, chris_prev_k_model_anomaly_ds, chris_capped_exponent_model_anomaly_ds, explicit_model_anomaly_ds, implicit_model_anomaly_ds, semi_implicit_model_anomaly_ds])

# remove whatever seasonal cycle remains
model_names = ["CHRIS_PREV_CUR", "CHRIS_MEAN_K", "CHRIS_PREV_K", "CHRIS_CAPPED_EXPONENT", "EXPLICIT", "IMPLICIT", "SEMI_IMPLICIT"]
for variable_name in model_names:
    monthly_mean = get_monthly_mean(all_anomalies_ds[variable_name])
    all_anomalies_ds[variable_name] = get_anomaly(all_anomalies_ds, variable_name, monthly_mean)[variable_name + "_ANOMALY"]
    all_anomalies_ds = all_anomalies_ds.drop_vars(variable_name + "_ANOMALY")

# clean up prev_cur model
if CLEAN_CHRIS_PREV_CUR:
    all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = all_anomalies_ds["CHRIS_PREV_CUR"].where((all_anomalies_ds["CHRIS_PREV_CUR"] > -10) & (all_anomalies_ds["CHRIS_PREV_CUR"] < 10))
    n_modes = 20
    monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
    map_mask = temperature_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5)
    eof_ds, variance, PCs, EOFs = get_eof_with_nan_consideration(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"], map_mask, modes=n_modes, monthly_mean_ds=None, tolerance=1e-2)
    all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = eof_ds.rename("CHRIS_PREV_CUR_CLEAN")
    chris_prev_cur_clean_monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
    all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = get_anomaly(all_anomalies_ds, "CHRIS_PREV_CUR_CLEAN", chris_prev_cur_clean_monthly_mean)["CHRIS_PREV_CUR_CLEAN_ANOMALY"]
    all_anomalies_ds = all_anomalies_ds.drop_vars("CHRIS_PREV_CUR_CLEAN_ANOMALY")

# save
all_anomalies_ds = remove_empty_attributes(all_anomalies_ds) # when doing the seasonality removal, some units are None
#all_anomalies_ds.to_netcdf("../datasets/all_anomalies.nc")

# format entrainment flux datasets
if INCLUDE_ENTRAINMENT:
    entrainment_flux_prev_cur_ds = xr.concat(entrainment_fluxes_prev_cur, 'TIME')
    entrainment_flux_prev_cur_ds = entrainment_flux_prev_cur_ds.drop_vars(["MONTH"])
    entrainment_flux_prev_cur_ds = entrainment_flux_prev_cur_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_prev_cur_ds = entrainment_flux_prev_cur_ds.rename("ENTRAINMENT_FLUX_PREV_CUR_ANOMALY")

    entrainment_flux_mean_k_ds = xr.concat(entrainment_fluxes_mean_k, 'TIME')
    entrainment_flux_mean_k_ds = entrainment_flux_mean_k_ds.drop_vars(["MONTH"])
    entrainment_flux_mean_k_ds = entrainment_flux_mean_k_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_mean_k_ds = entrainment_flux_mean_k_ds.rename("ENTRAINMENT_FLUX_MEAN_K_ANOMALY")

    entrainment_flux_prev_k_ds = xr.concat(entrainment_fluxes_prev_k, 'TIME')
    entrainment_flux_prev_k_ds = entrainment_flux_prev_k_ds.drop_vars(["MONTH"])
    entrainment_flux_prev_k_ds = entrainment_flux_prev_k_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_prev_k_ds = entrainment_flux_prev_k_ds.rename("ENTRAINMENT_FLUX_PREV_K_ANOMALY")

    entrainment_flux_capped_exponent_ds = xr.concat(entrainment_fluxes_capped_exponent, 'TIME')
    entrainment_flux_capped_exponent_ds = entrainment_flux_capped_exponent_ds.drop_vars(["MONTH"])
    entrainment_flux_capped_exponent_ds = entrainment_flux_capped_exponent_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_capped_exponent_ds = entrainment_flux_capped_exponent_ds.rename("ENTRAINMENT_FLUX_CAPPED_EXPONENT_ANOMALY")

    entrainment_flux_explicit_ds = xr.concat(entrainment_fluxes_explicit, 'TIME')
    entrainment_flux_explicit_ds = entrainment_flux_explicit_ds.drop_vars(["MONTH"])
    entrainment_flux_explicit_ds = entrainment_flux_explicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_explicit_ds = entrainment_flux_explicit_ds.rename("ENTRAINMENT_FLUX_EXPLICIT_ANOMALY")

    entrainment_flux_implicit_ds = xr.concat(entrainment_fluxes_implicit, 'TIME')
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.drop_vars(["MONTH"])
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.rename("ENTRAINMENT_FLUX_IMPLICIT_ANOMALY")

    entrainment_flux_semi_implicit_ds = xr.concat(entrainment_fluxes_semi_implicit, 'TIME')
    entrainment_flux_semi_implicit_ds = entrainment_flux_semi_implicit_ds.drop_vars(["MONTH"])
    entrainment_flux_semi_implicit_ds = entrainment_flux_semi_implicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_semi_implicit_ds = entrainment_flux_semi_implicit_ds.rename("ENTRAINMENT_FLUX_SEMI_IMPLICIT_ANOMALY")


# merge the relevant fluxes into a single dataset
flux_components_to_merge = []
variable_names = []
if INCLUDE_SURFACE:
    surface_flux_da = surface_flux_da.rename("SURFACE_FLUX_ANOMALY")
    flux_components_to_merge.append(surface_flux_da)
    variable_names.append("SURFACE_FLUX_ANOMALY")
if INCLUDE_EKMAN:
    ekman_anomaly_da = ekman_anomaly_da.rename("EKMAN_FLUX_ANOMALY")
    flux_components_to_merge.append(ekman_anomaly_da)
    variable_names.append("EKMAN_FLUX_ANOMALY")
if INCLUDE_GEOSTROPHIC:
    geostrophic_anomaly_da = geostrophic_anomaly_da.rename("GEOSTROPHIC_FLUX_ANOMALY")
    flux_components_to_merge.append(geostrophic_anomaly_da)
    variable_names.append("GEOSTROPHIC_FLUX_ANOMALY")
if INCLUDE_ENTRAINMENT:
    flux_components_to_merge.append(entrainment_flux_prev_cur_ds)
    flux_components_to_merge.append(entrainment_flux_mean_k_ds)
    flux_components_to_merge.append(entrainment_flux_prev_k_ds)
    flux_components_to_merge.append(entrainment_flux_capped_exponent_ds)
    flux_components_to_merge.append(entrainment_flux_explicit_ds)
    flux_components_to_merge.append(entrainment_flux_implicit_ds)
    flux_components_to_merge.append(entrainment_flux_semi_implicit_ds)
    variable_names.append("ENTRAINMENT_FLUX_PREV_CUR_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_MEAN_K_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_PREV_K_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_CAPPED_EXPONENT_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_EXPLICIT_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_IMPLICIT_ANOMALY")
    variable_names.append("ENTRAINMENT_FLUX_SEMI_IMPLICIT_ANOMALY")

flux_components_ds = xr.merge(flux_components_to_merge)

# remove whatever seasonal cycle may remain from the components
for variable_name in variable_names:
    monthly_mean = get_monthly_mean(flux_components_ds[variable_name])
    flux_components_ds[variable_name] = get_anomaly(flux_components_ds, variable_name, monthly_mean)[variable_name + "_ANOMALY"]
    flux_components_ds = flux_components_ds.drop_vars(variable_name + "_ANOMALY")

flux_components_ds = remove_empty_attributes(flux_components_ds)
print(flux_components_ds)

# flux_components_ds.to_netcdf("flux_components.nc")
# all_anomalies_ds.to_netcdf("temperature_model_anomalies.nc")

# make_movie(all_anomalies_ds["EXPLICIT"], -5, 5)
make_movie(all_anomalies_ds["IMPLICIT"], -2, 2)
# make_movie(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"], -5, 5)
# make_movie(all_anomalies_ds["CHRIS_MEAN_K"], -5, 5)

observed = xr.open_dataset(
    "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc", decode_times=False
)['ANOMALY_ML_TEMPERATURE']
observed = xr.open_dataset(
    "datasets/Temperature_Anomalies-(2004-2018).nc", decode_times=False
)['ANOMALY_TEMPERATURE'].sel(PRESSURE=2.5)
make_movie(observed, -2, 2)

print(all_anomalies_ds["IMPLICIT"].max().item(), all_anomalies_ds["IMPLICIT"].min().item())
print(all_anomalies_ds["IMPLICIT"].mean().item())
print(abs(all_anomalies_ds["IMPLICIT"]).mean().item())

print(observed.max().item(), observed.min().item())
print(observed.mean().item())
print(abs(observed).mean().item())
print('-----')

rmse_difference = np.sqrt(((observed - all_anomalies_ds["IMPLICIT"]) ** 2).mean(dim=['TIME']))
rmse_observed = np.sqrt((observed ** 2).mean(dim=['TIME']))

print(rmse_difference.mean().item())
print(rmse_observed.mean().item())
rmse_difference.plot(x='LONGITUDE', y='LATITUDE', cmap='viridis', vmin=0, vmax=3)
plt.show()

rmse = rmse_difference / rmse_observed
print(rmse.mean().item())
rmse.plot(x='LONGITUDE', y='LATITUDE', cmap='viridis', vmin=0, vmax=3)
plt.show()

corr = xr.corr(observed, all_anomalies_ds["IMPLICIT"], dim='TIME')
corr.plot(x='LONGITUDE', y='LATITUDE', cmap='viridis', vmin=-1, vmax=1)
plt.show()
