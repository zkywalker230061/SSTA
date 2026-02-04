import numpy as np
import xarray as xr
from read_nc import fix_rg_time



c_o = 4100                         #specific heat capacity of seawater = 4100 Jkg^-1K^-1
omega = 2*np.pi/(24*3600)         #Earth's angular velocity
#f = 2*omega*np.sin(phi)            #Coriolis Parameter


def ekman_current_anomaly(tau_x_anom, tau_y_anom, dTm_dx_monthly, dTm_dy_monthly, f_2d, fmin=1e-5):
    """
    Compute Q'_Ek = c_o * (τx'/xf * dTmbar/dy - τy'/f * dTmbar/dx)
    Output dims: (TIME, LATITUDE, LONGITUDE)
    """
    # Create month index per TIME
    m = month_idx(tau_x_anom["TIME"])

    # Select gradient of the corresponding month for each TIME (aligns MONTH -> TIME)
    dTm_dx_t = dTm_dx_monthly.sel(MONTH=m)
    dTm_dy_t = dTm_dy_monthly.sel(MONTH=m)

    # Broadcast Coriolis parameter
    f = f_2d.broadcast_like(tau_x_anom)
    mask = np.abs(f) > fmin  # avoid division near equator

    # Compute Q'_Ek
    Q_ek = c_o * ((tau_x_anom * dTm_dy_t / f) - (tau_y_anom * dTm_dx_t / f))
    Q_ek_x = c_o * (tau_x_anom * dTm_dy_t / f)
    Q_ek_y = c_o * (tau_y_anom * dTm_dx_t / f)
    Q_ek = Q_ek.where(mask)
    Q_ek_x = Q_ek_x.where(mask)
    Q_ek_y = Q_ek_y.where(mask)

    Q_ek.name = "Q_Ek_anom"
    Q_ek.attrs.update({
        "description": "Ekman term anomaly using τ' and monthly mean Tm gradients",
        "formula": "c_o * (τx'/f * dTmbar/dy - τy'/f * dTmbar/dx)",
    })
    return Q_ek, Q_ek_x, Q_ek_y

def ekman_current_anomaly_salinity(tau_x_anom, tau_y_anom, dTm_dx_monthly, dTm_dy_monthly, f_2d, fmin=1e-5):
    """
    Compute Q'_Ek = (τx'/xf * dTmbar/dy - τy'/f * dTmbar/dx)
    Output dims: (TIME, LATITUDE, LONGITUDE)
    """
    # Create month index per TIME
    m = month_idx(tau_x_anom["TIME"])

    # Select gradient of the corresponding month for each TIME (aligns MONTH -> TIME)
    dTm_dx_t = dTm_dx_monthly.sel(MONTH=m)
    dTm_dy_t = dTm_dy_monthly.sel(MONTH=m)

    # Broadcast Coriolis parameter
    f = f_2d.broadcast_like(tau_x_anom)
    mask = np.abs(f) > fmin  # avoid division near equator

    # Compute Q'_Ek
    Q_ek = ((tau_x_anom * dTm_dy_t / f) - (tau_y_anom * dTm_dx_t / f))
    Q_ek_x = (tau_x_anom * dTm_dy_t / f)
    Q_ek_y = (tau_y_anom * dTm_dx_t / f)
    Q_ek = Q_ek.where(mask)
    Q_ek_x = Q_ek_x.where(mask)
    Q_ek_y = Q_ek_y.where(mask)

    Q_ek.name = "Q_Ek_anom"
    Q_ek.attrs.update({
        "description": "Ekman term anomaly using τ' and monthly mean Sm gradients",
        "formula": "(τx'/f * dTmbar/dy - τy'/f * dTmbar/dx)",
    })
    return Q_ek, Q_ek_x, Q_ek_y

def month_idx (time_da: xr.DataArray) -> xr.DataArray:
    n = time_da.sizes['TIME']
    # Repeat 1..12 along TIME; align coords to TIME
    month_idx = (xr.DataArray(np.arange(n) % 12 + 1, dims=['TIME'])
                 .assign_coords(TIME=time_da))
    month_idx.name = 'MONTH'
    return month_idx

def get_monthly_mean(da: xr.DataArray,) -> xr.DataArray:
    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    
    m = month_idx(da['TIME'])
    monthly_mean_da = da.groupby(m).mean('TIME', keep_attrs=True)
    # monthly_means = []
    # for _, month_num in MONTHS.items():
    #     monthly_means.append(
    #         da.sel(TIME=da['TIME'][month_num-1::12]).mean(dim='TIME')
    #     )
    # monthly_mean_da = xr.concat(monthly_means, dim='MONTH')
    # monthly_mean_da = monthly_mean_da.assign_coords(MONTH=list(MONTHS.values()))
    # monthly_mean_da['MONTH'].attrs['units'] = 'month'
    # monthly_mean_da['MONTH'].attrs['axis'] = 'M'
    # monthly_mean_da.attrs['units'] = da.attrs.get('units')
    # monthly_mean_da.attrs['long_name'] = f"Seasonal Cycle Mean of {da.attrs.get('long_name')}"
    # monthly_mean_da.name = f"MONTHLY_MEAN_{da.name}"
    return monthly_mean_da


def get_anomaly(full_field, monthly_mean):
    """
    Calculates the anomaly of a full time-series DataArray
    by subtracting its corresponding monthly mean.
    """
    if 'TIME' not in full_field.dims:
        raise ValueError("The full_field DataArray must have a TIME dimension.")
    
    # Get the month index (1-12) for each item in the full_field
    m = month_idx(full_field['TIME'])
    anom = full_field.groupby(m) - monthly_mean
    return anom

def coriolis_parameter(lat):
    phi_rad = np.deg2rad(lat)
    f = 2 * omega * np.sin(phi_rad)
    f = xr.DataArray(f, coords={'LATITUDE': lat}, dims=['LATITUDE'])
    f.attrs['units'] = 's^-1'
    return f

def repeat_monthly_field(ds, var_name, n_repeats=15):
    """
    Take a dataset with a monthly 3D field (MONTH, LATITUDE, LONGITUDE)
    and repeat it n_repeats times along the MONTH axis to create a new
    time-like dimension of length 12 * n_repeats.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset. Must contain:
        - coord "MONTH" of length 12
        - coord "LATITUDE"
        - coord "LONGITUDE"
        - data variable `var_name` with dims ("MONTH","LATITUDE","LONGITUDE")
    var_name : str
        Name of the data variable to tile, e.g. "MONTHLY_MEAN_MLD_PRESSURE".
    n_repeats : int, default 15
        How many times to repeat the 12-month cycle.

    Returns
    -------
    xarray.Dataset
        Dataset with:
        - new dim "TIME" of length 12 * n_repeats
        - coords "TIME", "LATITUDE", "LONGITUDE"
        - data variable renamed to the same var_name but
          now on ("TIME","LATITUDE","LONGITUDE")
    """

    month_vals = ds["MONTH"].values  # e.g. [1,2,...,12]

    time_coord = np.tile(month_vals, n_repeats).astype(float)

    for i in range(len(time_coord)):
        time_coord[i] = time_coord[i] + (i // 12) * 12

    time_coord = time_coord - 0.5  # length = 12 * n_repeats

    data_var = ds.values  # shape (12, lat, lon)
    data_tiled = np.tile(data_var, (n_repeats, 1, 1))

    out = xr.Dataset(
        {
            var_name: (
                ("TIME", "LATITUDE", "LONGITUDE"),
                data_tiled,
            )
        },
        coords={
            "TIME": time_coord,
            "LATITUDE": ds["LATITUDE"].values,
            "LONGITUDE": ds["LONGITUDE"].values,
        },
    )
    return out