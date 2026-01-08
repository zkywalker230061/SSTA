#%%
import numpy as np
import xarray as xr
from read_nc import fix_rg_time

c_o = 4100                         #specific heat capacity of seawater = 4100 Jkg^-1K^-1
omega = 2*np.pi/(24*3600)         #Earth's angular velocity
#f = 2*omega*np.sin(phi)            #Coriolis Parameter


def ekmann_current (tau_x, tau_y, dTm_dx, dTm_dy,f):
    Q_ek = (c_o/f) *(tau_x*dTm_dy - tau_y*dTm_dx)
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
    
    # Group the full_field by its month index and then subtract
    # the corresponding monthly_mean. This preserves the original 3D shape.
    anom = full_field.groupby(m) - monthly_mean
    return anom

def coriolis_parameter(lat_ds):
    phi_rad = np.deg2rad(lat_ds)
    f = 2 * omega * np.sin(phi_rad)
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
    file_path = r"C:\Users\jason\MSciProject\era5_interpolated.nc"
    grad_lat_file_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature_Gradient_Lat.nc"
    grad_lon_file_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature_Gradient_Lon.nc"

    ds = xr.open_dataset(
        file_path, 
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True)
    
    ds_grad_lat = xr.open_dataset(
        grad_lat_file_path, 
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True)
    
    ds_grad_lon = xr.open_dataset(
        grad_lon_file_path, 
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True)
    
    ds_tau_x = ds['avg_iews']
    ds_tau_y = ds['avg_inss']
 
    monthly_mean_tau_x = get_monthly_mean(ds_tau_x)
    monthly_mean_tau_y = get_monthly_mean(ds_tau_y)

    #print ('monthly_mean_tau_x: \n', monthly_mean_tau_x)

    # full_field_monthly_mean_tau_x = repeat_monthly_field(monthly_mean_tau_x, 'avg_iews')
    # full_field_monthly_mean_tau_y = repeat_monthly_field(monthly_mean_tau_y, 'avg_inss')

    # tau_x_anom = get_anomaly(ds_tau_x, full_field_monthly_mean_tau_x)
    # tau_y_anom = get_anomaly(ds_tau_y, full_field_monthly_mean_tau_y)

    ds_grad_lat = repeat_monthly_field(ds_grad_lat["temp_gradient_lat"], "temp_gradient_lat")
    ds_grad_lon = repeat_monthly_field(ds_grad_lon["temp_gradient_lon"], "temp_gradient_lon")

    tau_x_anom = get_anomaly(ds_tau_x, monthly_mean_tau_x)
    tau_y_anom = get_anomaly(ds_tau_y, monthly_mean_tau_y)

    lat = ds["LATITUDE"]
    # f = coriolis_parameter(lat)

    f = coriolis_parameter(lat).broadcast_like(tau_x_anom)
    f = fix_rg_time(f, mode='datetime')
    
    # print('tau_x_anom.dims',tau_x_anom.dims)
    # print('tau_y_anom.dims', tau_y_anom.dims)
    # print(ds_grad_lat.dims)
    # print(ds_grad_lon.dims)
    # print(f.dims)
    # print(tau_x_anom)
    # print(ds_grad_lat)
    # print(f)

    Q_ek_anom = ekmann_current(
        tau_x_anom,       # This is the tau_x DataArray
        tau_y_anom,       # This is the tau_y DataArray
        ds_grad_lon["temp_gradient_lon"],      # This is dTm_dx (longitude gradient)
        ds_grad_lat["temp_gradient_lat"],      # This is dTm_dy (latitude gradient)
        f                         # This is the coriolis parameter (already a DataArray)
    )

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
    print(f)


# %%
