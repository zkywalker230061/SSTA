################################################
# Assume h = 100 m
# T_m(x,y,t) = (1/h) ∫_0^h T(x,y,z,t) dz
# S_m(x,y,t) = (1/h) ∫_0^h S(x,y,z,t) dz
# Using gsw to convert pressure to depth
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import re
from typing import Optional
#from Jason.rg_sst_map.calculte_Tm import vertical_integral 

# H_M=100


# #----1. Read .nc Files--------------------------------------------------------
# def fix_rg_time(ds, mode="datetime"):
#     """
#     mode='datetime' -> TIME as numpy datetime64[ns] (recommended)
#     mode='period'   -> TIME as pandas PeriodIndex (monthly)
#     mode='int'      -> TIME as integer months since reference (0..179)
#     """
#     if "TIME" not in ds.coords:
#         return ds

#     units = ds.TIME.attrs.get("units", "")
#     m = re.match(r"months\s+since\s+(\d{4}-\d{2}-\d{2})", units, re.I)
#     if not m:
#         return ds  # nothing to change

#     ref = pd.Timestamp(m.group(1))
#     months = ds.TIME.values.astype(int)

#     if mode == "datetime":
#         new = pd.to_datetime([ref + pd.DateOffset(months=int(n)) for n in months])
#         ds = ds.assign_coords(TIME=new)
#         ds.TIME.attrs.update({"standard_name": "time", "calendar": "gregorian", "decoded_from": units})
#     elif mode == "period":
#         new = pd.period_range(start=ref, periods=len(months), freq="M")
#         ds = ds.assign_coords(TIME=new)
#         ds.TIME.attrs.update({"freq": "M", "decoded_from": units})
#     elif mode == "int":
#         ds = ds.assign_coords(TIME=("TIME", months))
#         ds.TIME.attrs.update({"units": units, "note": "integer months since reference"})
#     return ds

# ds_temp = xr.open_dataset(
#     "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc",
#     engine="netcdf4",
#     decode_times=False,
#     mask_and_scale=True,
# )

# ds_sal = xr.open_dataset(
#     "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
#     engine="netcdf4",
#     decode_times=False,
#     mask_and_scale=True,
# )

# ds_temp = fix_rg_time(ds_temp)
# ds_sal = fix_rg_time(ds_sal)

#-----2. Convert Pressure (dbar) to Depth (m)---------------------------------------------------------------------------
def depth_from_pressure(p: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
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

#-----3. Data = Mean + Anomaly-----------------------------------------------------------------------------------------------------
def _full_field(mean_data: xr.DataArray, anom_data: xr.DataArray, time=None) -> xr.DataArray:
    """
    Reconstruct full field = mean + anomaly.
    If time is provided (e.g., '2010-06-01'), returns 3D (PRESSURE, LATITUDE, LONGITUDE);
    else returns 4D (TIME, PRESSURE, LATITUDE, LONGITUDE).
    """
    data = anom_data + mean_data  # broadcasts mean (P, Y, X) across TIME
    if time is not None and "TIME" in data.dims:
        data = data.sel(TIME=pd.Timestamp(time))
    return data

#-----4. z to xarray -------------------------------------------------------------------------------------
h_meters = 100.0
ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
TDIM = "TIME"
T_VAR = "ARGO_TEMPERATURE_MEAN"
S_VAR = "ARGO_SALINITY_MEAN"
T_VAR_ANOMALY = "ARGO_TEMPERATURE_ANOMALY"

def z_to_xarray(z,
               T: xr.DataArray,
               zdim: str = ZDIM,
               ydim: str = YDIM,
               xdim: str = XDIM) -> xr.DataArray:
    """
    Returns an xarray DataArray of depth z (in meters) broadcast to the shape of T.
    """
    if not isinstance(z, xr.DataArray):
        z = np.asarray(z)
    # Z array from gsw has shape (Latitude, Depth)
        z_new = xr.DataArray(z, dims=[ydim, zdim],
                             coords={ydim: T.coords[ydim],
                                     zdim: T.coords[zdim]})
    else:
        z_new = z
    
    # Ensure z_new has the required dimensions
    if zdim not in z_new.dims or ydim not in z_new.dims:
        raise ValueError(f"{zdim} or {ydim} not in {z_new.dims}")
    
    # Expand z_new to include missing dimensions in T
    if xdim in T.dims and xdim not in z_new.dims:
        z_new = z_new.expand_dims({xdim: T.coords[xdim]})
    # if tdim in T.dims and tdim not in z_new.dims:
    #     z_new = z_new.expand_dims({tdim: T.coords[tdim]})

    z_broadcast = z_new.broadcast_like(T)
    return z_broadcast


#-----5. Vertical Integral------------------------------------------------------------
def vertical_integral(
    T: xr.DataArray,
    z,
    top: float = 0.0,
    bottom: float = h_meters,
    normalise: str = "available",
    zdim: str = ZDIM,
) -> xr.DataArray:
    """
    Calculate the vertical integral of a 1D field using the trapezoidal rule.
    """
    # Ensure both inputs T and Z hae the vertical dimensions last in their order of dimensions
    if zdim not in T.dims:
        raise ValueError(f"{zdim} not in T.dims")
    
     
    T_sorted = T.transpose(..., zdim)
    z_sorted = xr.broadcast(z, T_sorted)[0].transpose(..., zdim)
    
    # Creating segments of adjacent levels
    T_next = T_sorted.shift({zdim: -1})
    z_next = z_sorted.shift({zdim: -1})

    # Get the boundaries of each segment
    segment_top = np.minimum(z_sorted, z_next)
    segment_bottom = np.maximum(z_sorted, z_next)

    # Clip data to the integration limits
    
    z_low = xr.zeros_like(segment_top) + top
    z_high = xr.zeros_like(segment_top) + bottom
    overlap = (np.minimum(segment_bottom, z_high) - np.maximum(segment_top, z_low)).clip(0)

    # Compute the trapezoidal area of each segment
    segment_area = 0.5 * (T_sorted + T_next) * overlap

    # Integrate over the vertical dimension and normalise if needed 
    num = segment_area.sum(dim=zdim, skipna=True)
    den = overlap.sum(dim=zdim, skipna=True)
    if normalise == "available":
        out = (num / den).where(den != 0)
    elif normalise == "full":
        out = (num / (bottom-top)).where(den >= (bottom-top))
    else:
        raise ValueError("normalise must be 'available' or 'full'")
    
    # GPT aided
    out = out.rename(f"T_upper{int(abs(bottom))}")
    out.attrs.update({
    "long_name": f"Upper {int(abs(bottom))} m mean temperature (trapezoidal)",
    "units": T.attrs.get("units", "degC")
    })

    return out






# if __name__ == "__main__":
#     ds_temp = xr.open_dataset(
#     "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc",
#     engine="netcdf4",
#     decode_times=False,
#     mask_and_scale=True,
#     )

#     ds_sal = xr.open_dataset(
#         "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
#         engine="netcdf4",
#         decode_times=False,
#         mask_and_scale=True,
#     )

#     ds_temp = fix_rg_time(ds_temp)
#     ds_sal = fix_rg_time(ds_sal)
#     p = ds_temp['PRESSURE']
#     lat = ds_temp['LATITUDE']

#     depth = depth_from_pressure(p,lat)

#     print(depth)
#     print(len(depth))
#     print(depth.shape)
#     print(len(depth[0]))