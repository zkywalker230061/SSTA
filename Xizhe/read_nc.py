import re
import xarray as xr
import pandas as pd


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


if __name__ == "__main__":
    file_path = '/Users/xxz/Desktop/SSTA/datasets/windstress.nc'
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
