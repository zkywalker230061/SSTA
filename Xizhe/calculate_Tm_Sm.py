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
from calculate_Tm_Jason import z_to_xarray, vertical_integral
#from Jason.rg_sst_map.calculte_Tm import vertical_integral 

H_M=100

#----1. Read .nc Files--------------------------------------------------------
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

ds_temp = xr.open_dataset(
    "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc",
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

ds_sal = xr.open_dataset(
    "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

ds_temp = fix_rg_time(ds_temp)
ds_sal = fix_rg_time(ds_sal)

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


# ---- 4. Core vertical overlapped trapezoid integral ---------------------------



# def mean_temperature_Tm(ds_temp: xr.Dataset, h_m: float = H_M, time=None) -> xr.DataArray:
#     """
#     Mixed-layer mean temperature (0–h_m m) using true meters from TEOS-10.
#     Returns (LATITUDE, LONGITUDE) if 'time' given; else (TIME, LATITUDE, LONGITUDE).
#     """
#     T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]          # (P, Y, X)
#     T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]       # (T, P, Y, X)
#     T_full = _full_field(T_mean, T_anom, time=time)         # (T?, P, Y, X)

#     z = depth_from_pressure(ds_temp["PRESSURE"], ds_temp["LATITUDE"])  # (LATITUDE, PRESSURE)
#     Tm = vertical_integral(
#         V=T_full,
#         z_mid=z,
#         top=0.0,
#         bottom=float(h_m),
#         zdim="PRESSURE",
#         normalise="available",
#         min_cover=0.25 * float(h_m),   # require at least 25% coverage; tweak if you prefer
#         out_name=f"T_mean_0to{int(h_m)}m",
#     )
#     Tm.attrs.setdefault("units", T_mean.attrs.get("units", "degC"))
#     Tm.attrs["note"] = "Mean of (ARGO_TEMPERATURE_MEAN + ANOMALY) over 0–h using TEOS-10 depths"
#     return Tm


# def mean_salinity_Ts(ds_sal: xr.Dataset, h_m: float = H_M, time=None) -> xr.DataArray:
#     """
#     Mixed-layer mean salinity (0–h_m m) using true meters from TEOS-10.
#     Returns (LATITUDE, LONGITUDE) if 'time' given; else (TIME, LATITUDE, LONGITUDE).
#     """
#     S_mean = ds_sal["ARGO_SALINITY_MEAN"]              # (P, Y, X)
#     S_anom = ds_sal["ARGO_SALINITY_ANOMALY"]           # (T, P, Y, X)
#     S_full = _full_field(S_mean, S_anom, time=time)         # (T?, P, Y, X)

#     z = depth_from_pressure(ds_temp["PRESSURE"], ds_temp["LATITUDE"])  # (LATITUDE, PRESSURE)

#     Sm = vertical_integral(
#         V=S_full,
#         z_mid=z,
#         top=0.0,
#         bottom=float(h_m),
#         zdim="PRESSURE",
#         normalise="available",
#         min_cover=0.25 * float(h_m),
#         out_name=f"S_mean_0to{int(h_m)}m",
#     )
#     Sm.attrs.setdefault("units", S_mean.attrs.get("units", "psu"))
#     Sm.attrs["note"] = "Mean of (ARGO_SALINITY_MEAN + ANOMALY) over 0–h using TEOS-10 depths"
#     return Sm





# #----Probably Wrong--------------------------------------------------------------------------------------
# def _thickness_weights_m(ds, h_m=H_M):
#     """
#     Build layer thickness (m) within 0..h_m for each latitude, based on depth at mid-levels.
#     Returns w with dims (PRESSURE, LATITUDE) that sum to ≤ h_m at each latitude.
#     """
#     p = ds["PRESSURE"].values.astype(float)       # (P,)
#     lat = ds["LATITUDE"].values.astype(float)     # (Y,)

#     # Depth at mid-levels (m, positive down), shape (Y, P)
#     d_mid = get_depth(p, lat)

#     # Upper interfaces: 0 for top cell; then midpoints between adjacent mid-levels
#     upper = np.empty_like(d_mid)
#     upper[:, 0] = 0.0
#     upper[:, 1:] = 0.5 * (d_mid[:, :-1] + d_mid[:, 1:])

#     # Lower interfaces: shift upper by one; last one extended (will be clipped by h_m)
#     lower = np.empty_like(d_mid)
#     lower[:, :-1] = upper[:, 1:]
#     lower[:, -1] = np.maximum(h_m, d_mid[:, -1])  # safe sentinel beyond h_m

#     # Thickness contribution inside [0, h_m]
#     thick = np.maximum(0.0, np.minimum(h_m, lower) - np.maximum(0.0, upper))  # (Y, P)

#     # Return as DataArray with dims (PRESSURE, LATITUDE) to align with your variables
#     w = xr.DataArray(
#         thick.T,
#         dims=("PRESSURE", "LATITUDE"),
#         coords={"PRESSURE": ds["PRESSURE"], "LATITUDE": ds["LATITUDE"]},
#         name="thickness_m",
#         attrs={"units": "m", "description": f"Layer thickness within 0–{h_m} m"},
#     )
#     return w



if __name__ == "__main__":
    p = ds_temp['PRESSURE']
    lat = ds_temp['LATITUDE']

    depth = depth_from_pressure(p,lat)

    print(depth)
    print(len(depth))
    print(depth.shape)
    print(len(depth[0]))