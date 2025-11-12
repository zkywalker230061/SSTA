import re
import xarray as xr
import pandas as pd
import numpy as np
import gsw


def fix_rg_time(ds, mode="datetime"):
    """
    mode = 'NONE'.  -> TIME keep its original form for exporting files. 
    mode='datetime' -> TIME as numpy datetime64[ns] (recommended)
    mode='period'   -> TIME as pandas PeriodIndex (monthly)
    mode='int'      -> TIME as integer months since reference (0..179)
    """
    if "TIME" not in ds.coords:
        return ds
    if mode == "NONE":
        return ds
    
    units = ds.TIME.attrs.get("units", "")
    m = re.match(r"months\s+since\s+(\d{4}-\d{2}-\d{2})", units, re.I)
    if not m:
        return ds  # nothing to change

    ref = pd.Timestamp(m.group(1))
    months = ds.TIME.values.astype(int)

    if mode == "datetime":
        new = pd.to_datetime([ref + pd.DateOffset(months=int(n)) for n in months])
        ds = ds.assign_coords(TIME=new)
        ds.TIME.attrs.update({"standard_name": "time", "calendar": "gregorian", "decoded_from": units})
    elif mode == "period":
        new = pd.period_range(start=ref, periods=len(months), freq="M")
        ds = ds.assign_coords(TIME=new)
        ds.TIME.attrs.update({"freq": "M", "decoded_from": units})
    elif mode == "int":
        ds = ds.assign_coords(TIME=("TIME", months))
        ds.TIME.attrs.update({"units": units, "note": "integer months since reference"})
    return ds

def fix_longitude_coord(ds):
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

def get_anomaly(
    da: xr.DataArray,
    monthly_mean_da: xr.DataArray
) -> xr.DataArray:
    """
    Get the anomaly of the DataArray based on the provided monthly mean.
    Parameters
    ----------
    da: xarray.DataArray
        Input DataArray with 'TIME' coordinate.
    monthly_mean_da: xarray.DataArray
        Monthly mean DataArray with 'MONTH' coordinate.

    Returns
    -------
    xarray.DataArray
        Anomaly DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """

    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    anomalies = []
    for month_num in da.coords['TIME']:
        month_num = month_num.values
        month_mean_num = int((month_num + 0.5) % 12)
        if month_mean_num == 0:
            month_mean_num = 12
        anomalies.append(
            da.sel(TIME=month_num) - monthly_mean_da.sel(MONTH=month_mean_num)
        )
    anomaly_da = xr.concat(anomalies, "TIME")
    anomaly_da.attrs['units'] = da.attrs.get('units')
    anomaly_da.attrs['long_name'] = f"Anomaly of {da.attrs.get('long_name')}"
    anomaly_da.name = f"ANOMALY_{da.name}"
    return anomaly_da

def load_pressure_data(path: str, varname: str, *, time_mode: str = "datetime",) -> xr.DataArray:
    """Load MLD in PRESSURE units, fix time, convert to meters (positive down)."""

    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=True)
    #ds = fix_rg_time(ds, mode=time_mode)

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

def load_and_prepare_dataset(
        filepath: str,
        time_mode: bool = False,
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
            if time_mode:
                ds = fix_rg_time(ds)
            if longitude_180:
                ds = fix_longitude_coord(ds)
            return ds
    except (OSError, ValueError, KeyError) as e:
        print(f"Error loading {filepath}: {e}")
        return None

def depth_dbar_to_meter(p: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """
    TEOS-10 pressure->depth for every latitude.
    Inputs:
      p   : PRESSURE (dbar), 1D over 'PRESSURE'
      lat : LATITUDE (deg), 1D over 'LATITUDE'
    Returns:
      depth (m, positive down) with dims ('LATITUDE','PRESSURE')
    """
    p_ = np.asarray(p.values, dtype=float)         # (P,)
    la_ = np.asarray(lat.values, dtype=float)      # (Y,)
    # z_from_p returns negative below sea level; flip sign.
    z = gsw.z_from_p(p_[None, :], la_[:, None])    # (Y, P), negative down
    depth = xr.DataArray(
        -z,
        dims=("LATITUDE", "PRESSURE"),
        coords={"LATITUDE": lat, "PRESSURE": p},
        name="depth",
        attrs={"units": "m", "positive": "down", "note": "TEOS-10 from gsw.z_from_p"},
    )
    return depth

def mld_dbar_to_meter(p_mld: xr.DataArray, lat_3D: xr.DataArray) -> xr.DataArray:
    """
    Convert MLD pressure (dbar) with dims (TIME, LATITUDE, LONGITUDE)
    to depth (m, positive down) with the same dims, using GSW.
    """
    # Broadcast latitude to match p_mld shape and order
    # lat3d = xr.broadcast(lat_1d, p_mld)[0].transpose(*p_mld.dims)

    # gsw.z_from_p returns negative below sea level; flip sign for positive-down
    h_m = -xr.apply_ufunc(
        gsw.z_from_p,
        p_mld, lat_3D,
        input_core_dims=[[], []],      # already share (TIME, LATITUDE, LONGITUDE)
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
    )
    h_m.name = "MLD_depth"
    h_m.attrs.update({"units": "m", "positive": "down", "note": "TEOS-10 gsw.z_from_p"})
    return h_m


if __name__ == "__main__":
    file_path = '/Users/julia/Desktop/SSTA/datasets/windstress.nc'
    ds = xr.open_dataset(
        file_path, 
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True)
    #print(ds.TIME)
    #print(ds)
    ds= fix_rg_time(ds)
    print (ds['TIME'])
    #print(ds['avg_iews'].values)
