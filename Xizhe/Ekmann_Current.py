import numpy as np
import xarray as xr
from calculate_Tm_Sm import vertical_integral
from grad_field import compute_gradient_lat, compute_gradient_lon
from read_nc import fix_rg_time
import matplotlib.pyplot as plt


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
    Q_ek = Q_ek.where(mask)

    Q_ek.name = "Q_Ek_anom"
    Q_ek.attrs.update({
        "description": "Ekman term anomaly using τ' and monthly mean Tm gradients",
        "formula": "c_o * (τx'/f * dTmbar/dy - τy'/f * dTmbar/dx)",
    })
    return Q_ek

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

# def get_anomaly(full_field, monthly_mean):
#     anom = full_field - monthly_mean
#     return anom

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


if __name__ == "__main__":
    file_path = '/Users/julia/Desktop/SSTA/datasets/windstress.nc'
    grad_lat_file_path = '/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature_Gradient_Lat.nc'
    grad_lon_file_path = '/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature_Gradient_Lon.nc'

    ds = xr.open_dataset(               # (TIME: 180, LATITUDE: 145, LONGITUDE: 360)
        file_path,                      # Data variables:
        engine="netcdf4",                       # avg_iews   (TIME, LATITUDE, LONGITUDE) float32 38MB ...
        decode_times=False,                     # avg_inss   (TIME, LATITUDE, LONGITUDE) float32 38MB ...       
        mask_and_scale=True)            # * TIME       (TIME) float32 720B 0.5 1.5 2.5 3.5 ... 176.5 177.5 178.5 179.5
    
    ds_grad_lat = xr.open_dataset(
        grad_lat_file_path,             # (MONTH: 12, LATITUDE: 145, LONGITUDE: 360)
        engine="netcdf4",               # * MONTH      (MONTH) int64 96B 1 2 3 4 5 6 7 8 9 10 11 12
        decode_times=False,             # Data variables: __xarray_dataarray_variable__
        mask_and_scale=True)
    
    ds_grad_lon = xr.open_dataset(
        grad_lon_file_path,             # (MONTH: 12, LATITUDE: 145, LONGITUDE: 360)
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True)
    
    ds_tau_x = ds['avg_iews']           # (TIME: 180, LATITUDE: 145, LONGITUDE: 360)
    ds_tau_y = ds['avg_inss']           # TIME  0.5 1.5 2.5 3.5 ... 176.5 177.5 178.5 179.5
    dTm_dy_monthly = ds_grad_lat["__xarray_dataarray_variable__"]  # (MONTH, LAT, LON)
    dTm_dx_monthly = ds_grad_lon["__xarray_dataarray_variable__"]  # (MONTH, LAT, LON)
    # print("ds\n", ds)
    # print('ds_grad_lat:\n', ds_grad_lat)
    # print('ds_grad_lon:\n', ds_grad_lon)
    # print('ds_tau_x:\n', ds_tau_x)

    monthly_mean_tau_x = get_monthly_mean(ds_tau_x)     #(MONTH: 12, LATITUDE: 145, LONGITUDE: 360)
    monthly_mean_tau_y = get_monthly_mean(ds_tau_y)

    tau_x_anom = get_anomaly(ds_tau_x, monthly_mean_tau_x)  #(TIME: 180, LATITUDE: 145, LONGITUDE: 360)
    tau_y_anom = get_anomaly(ds_tau_y, monthly_mean_tau_y)

    # print ('monthly_mean_tau_x: \n', monthly_mean_tau_x)
    # print('tau_x_anom: \n', tau_x_anom)

    lat = ds["LATITUDE"]
    f_2d = coriolis_parameter(lat).broadcast_like(ds_tau_x)

    print('tau_x_anom.dims',tau_x_anom.dims)
    print('tau_y_anom.dims', tau_y_anom.dims)
    print('ds_grad_lat["__xarray_dataarray_variable__"].dims: ',ds_grad_lat["__xarray_dataarray_variable__"].dims)
    print('ds_grad_lon["__xarray_dataarray_variable__"].dims: ', ds_grad_lon["__xarray_dataarray_variable__"].dims)
    print('f_2d.dims',f_2d.dims)
    print ('Corioslis Parameter: \n',f_2d)
    
    Q_ek_anom = ekman_current_anomaly(tau_x_anom, tau_y_anom, dTm_dx_monthly, dTm_dy_monthly, f_2d)
    Q_ek_anom.TIME.attrs["units"] = "months since 2004-01-01" 
    Q_ek_anom = fix_rg_time(Q_ek_anom, mode="datetime")
    print(
        'ekmann current anomaly:', Q_ek_anom
        # ds_grad_lat,

        #  'original dataset:\n', ds,
        #'\n ds_tau_x: \n', ds_tau_x,
        #'\n ds_tau_y: \n', ds_tau_y,
        # '\n Monthly mean tau_x: \n', monthly_mean_tau_x,
        # '\n monthly mean tau_y: \n', monthly_mean_tau_y.shape,
        # '\n Full Field Monthly mean tau_x: \n', full_field_monthly_mean_tau_x,
        # '\n Full Field monthly mean tau_y: \n', full_field_monthly_mean_tau_y,
        # '\ntau x anomaly: \n', tau_x_anom['avg_iews'].values,
        # '\n tau y anomaly: \n', tau_y_anom['avg_inss'].values,
        #ds["avg_iews"]
    )


    date = "2004-02-01"
    Q_plot = Q_ek_anom.sel(TIME=f"{date}")

    plt.figure(figsize=(10, 5))
    Q_plot.plot(
        cmap="RdBu_r",
        vmin=-np.nanpercentile(Q_plot, 99),
        vmax=np.nanpercentile(Q_plot, 99),
        cbar_kwargs={"label": "Q'_Ek (arbitrary units)"}
    )
    plt.title(f"Ekman Current Anomaly ({date})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()