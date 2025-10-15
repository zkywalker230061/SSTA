import xarray as xr
import pandas as pd
import re

def fix_rg_time(ds, mode="datetime"):
    """
    mode='datetime' -> TIME as numpy datetime64[ns] (recommended)
    mode='period'   -> TIME as pandas PeriodIndex (monthly)
    mode='int'      -> TIME as integer months since reference (0..179)
    """
    if "TIME" not in ds.coords:
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


#Read the datasets
ds_temp = xr.open_dataset(
    "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc",
    engine="netcdf4",
    decode_times=False,   # disable decoding to avoid error
    mask_and_scale=True,
)

ds_sal = xr.open_dataset(
    "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

# fix TIME axis manually
ds_temp = fix_rg_time(ds_temp)
ds_sal = fix_rg_time(ds_sal)


#----------------------For Checking---------------------------------#
#print(ds_sal)
#print(ds_temp)

print(ds_temp.PRESSURE)

#print('salinity \n',ds_sal.ARGO_SALINITY_MEAN)
#print('temperature \n', ds_temp.ARGO_TEMPERATURE_MEAN)

#print(ds_temp.TIME.dtype)
#print(ds_temp.TIME[:200])
