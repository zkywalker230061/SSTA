"""
Writing a function to calculate the gradients of our fields
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Xizhe.utils_read_nc import fix_rg_time, fix_longitude_coord
from Xizhe.utils_Tm_Sm import depth_dbar_to_meter, _full_field, mld_dbar_to_meter, vertical_integral, z_to_xarray


def compute_gradient_lat(
        field: xr.DataArray,
        R=6.4e6) -> xr.DataArray:
    """
    Compute the gradient of the field with respect to latitude and longitude
    """

    # Convert degrees to radians for calculation
    lat_rad = np.deg2rad(field['LATITUDE'])
    #lon_rad = np.deg2rad(field['Longitude'])

    # Calculate the spacing in meters
    dlat = (np.pi/180) * R
    
    

    # Focusing on lat for now

    # Masks for neighbouring points
    # This is needed to identify valid neighbouring points for gradient calculation
    valid = field.notnull() # Creates a BOOL array where valid data (ocean) is set to True
    has_prev = valid.shift(LATITUDE=1, fill_value=False) # Cell above is ocean 
    has_next = valid.shift(LATITUDE=-1, fill_value=False) # Cell below is ocean
    has_prev2 = valid.shift(LATITUDE=2, fill_value=False) # Two cells above is ocean
    has_next2 = valid.shift(LATITUDE=-2, fill_value=False) # Two cells below is ocean

    # We can then define our interior points 
    interior = valid & has_prev & has_next # Maps out the interior points 
    # Define the edge points of our interior
    start = valid & ~has_prev
    end = valid & ~has_next

    # Now we define start/end by how many valid neighbours exist ahead/behind them
    start_run_3pt = start & has_next & has_next2 # Starting point for 2nd order forward difference
    end_run_3pt = end & has_prev & has_prev2 # End point for 2nd order backward difference
    start_run_2pt = start & has_next & ~has_next2 # Starting point for 1st order forward difference
    end_run_2pt = end & has_prev & ~has_prev2 # End point for 1st order backward difference
    single = start & ~has_next # Point with no valid neighbours (will leave as NaN)

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
    """

    # Convert degrees to radians for calculation
    lat_rad = np.deg2rad(field['LATITUDE'])
    #lon_rad = np.deg2rad(field['Longitude'])

    # Calculate the spacing in meters
    dlon = (np.pi/180) * R * np.cos(lat_rad)
    dlon = xr.DataArray(dlon, coords=field['LATITUDE'].coords, dims=['LATITUDE'])

  

    # Masks for neighbouring points
    # This is needed to identify valid neighbouring points for gradient calculation
    valid = field.notnull() # Creates a BOOL array where valid data (ocean) is set to True
    has_west = valid.roll(LONGITUDE=1, roll_coords=False) # Cell above is ocean 
    has_east = valid.roll(LONGITUDE=-1, roll_coords=False) # Cell below is ocean
    has_west2 = valid.roll(LONGITUDE=2, roll_coords=False) # Two cells above is ocean
    has_east2 = valid.roll(LONGITUDE=-2, roll_coords=False) # Two cells below is ocean

    # We can then define our interior points 
    interior = valid & has_west & has_east # Maps out the interior points 
    # Define the edge points of our interior
    start = valid & ~has_west
    end = valid & ~has_east

    # Now we define start/end by how many valid neighbours exist ahead/behind them
    start_run_3pt = start & has_east & has_east2 # Starting point for 2nd order forward difference
    end_run_3pt = end & has_west & has_west2 # End point for 2nd order backward difference
    start_run_2pt = start & has_east & ~has_east2 # Starting point for 1st order forward difference
    end_run_2pt = end & has_west & ~has_west2 # End point for 1st order backward difference
    single = start & ~has_east # Point with no valid neighbours (will leave as NaN)

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

def load_pressure_data(path: str, varname: str, *, compute_time_mode: str = "datetime",) -> xr.DataArray:
    """Load MLD in PRESSURE units, fix time, convert to meters (positive down)."""

    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=True)
    #ds = fix_rg_time(ds, mode=compute_time_mode)

    pressure = ds[varname] # Coordinates = (TIME: 180, LATITUDE: 145, LONGITUDE: 360)
    lat_1D = ds["LATITUDE"]
    lat_3D = xr.broadcast(lat_1D, pressure)[0].transpose(*pressure.dims)
    depth_m   = mld_dbar_to_meter(pressure, lat_3D)
    depth_m   = fix_longitude_coord(depth_m)

    # print('depth_bar:\n',depth_bar, depth_bar.shape)
    # print(lat_3D)
    # print('depth_m:\n',depth_m)
    # print('depth_m after fix_longitude:\n', depth_m)
    return depth_m

if __name__ == "__main__":
    #-----------------------------------------------------------------------------------------------------------
    temp_file_path = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
    salinity_file_path = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc"
    h_bar_file_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"

    ds_temp = xr.open_dataset(
        temp_file_path,
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True,
    )

    ds_sal = xr.open_dataset(
        salinity_file_path,
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True,
    )

    h_bar = xr.open_dataset(
        h_bar_file_path,
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True
    )

    #----------------------------------------------------------------------------------------------------
    T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]
    T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]
    T_full = _full_field(T_mean, T_anom)
    T_full = fix_longitude_coord(T_full)

    ds_Tbar_monthly = get_monthly_mean(T_full)                                  # MONTH: 12, PRESSURE: 58, LATITUDE: 145, LONGITUDE: 360)
    ds_h_bar_monthly = load_pressure_data(h_bar_file_path,              # MONTH: 12, LATITUDE: 145, LONGITUDE: 360
                                        'MONTHLY_MEAN_MLD_PRESSURE', 
                                        compute_time_mode="none")

    #------------------------------------------------------------------------------------------------------
    p = ds_temp['PRESSURE']
    lat = ds_temp['LATITUDE']
    depth = depth_dbar_to_meter(p,lat)

    ZDIM = "PRESSURE"
    YDIM = "LATITUDE"
    XDIM = "LONGITUDE"
    TDIM = "TIME"
    T_VAR = "ARGO_TEMPERATURE_MEAN"
    S_VAR = "ARGO_SALINITY_MEAN"
    T_VAR_ANOMALY = "ARGO_TEMPERATURE_ANOMALY"

    dz = z_to_xarray(depth, ds_Tbar_monthly) 

    #------------------------------------

    Temp_mld_bar = vertical_integral(ds_Tbar_monthly, dz, ds_h_bar_monthly)  

    gradient_lat = compute_gradient_lat(Temp_mld_bar)
    #print('gradient_lat:\n',gradient_lat)
    gradient_lon = compute_gradient_lon(Temp_mld_bar)
    #print('gradient_lon:\n',gradient_lon)
    grad_mag = np.sqrt(gradient_lat**2 + gradient_lon**2)
    #print('grad_mag:\n',grad_mag)


    #----Plot Gradient Map (Lat)----------------------------------------------------
    date = 10
    datetime = fix_rg_time(date)


    t0 = gradient_lat.sel(MONTH=date)

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto", vmin= -1e-5, vmax=1e-5
    )
    plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    plt.title(f"Mixed Layer Temperature Gradient (Lat)- {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    #----Plot Gradient Map (Lon)----------------------------------------------------
    t0 = gradient_lon.sel(MONTH=date)

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto", vmin = -1e-5, vmax=1e-5
    )
    plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    plt.title(f"Mixed Layer Temperature Gradient (Lon)- {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------------
    t0 = grad_mag.sel(MONTH=date)

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto", vmin = 0, vmax=1e-5
    )
    plt.colorbar(pc, label=f"Temperature Gradient (/m)")
    plt.title(f"Mixed Layer Temperature Gradient Magnitude- {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()