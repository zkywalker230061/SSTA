################################################
# Assume h = 100 m
# T_m(x,y,t) = (1/h) ∫_0^h T(x,y,z,t) dz
# S_m(x,y,t) = (1/h) ∫_0^h S(x,y,z,t) dz
# Using gsw to convert pressure to depth

import numpy as np
import pandas as pd
import xarray as xr
import gsw

H_M = 100.0  # slab depth in meters

def get_depth(p, lat):
    """
    Convert pressure (dbar) to depth (m, positive down) for each LATITUDE using TEOS-10.
    p: 1D array-like of PRESSURE mid-levels (dbar), shape (P,)
    lat: 1D array-like of LATITUDE (deg_north), shape (Y,)
    Returns: depth array with shape (Y, P), in meters (positive down).
    """
    z = gsw.z_from_p(np.asarray(p)[None, :], np.asarray(lat)[:, None])  # (Y, P), negative down
    return -z  # make positive down

def _thickness_weights_m(ds, h_m=H_M):
    """
    Build layer thickness (m) within 0..h_m for each latitude, based on depth at mid-levels.
    Returns w with dims (PRESSURE, LATITUDE) that sum to ≤ h_m at each latitude.
    """
    p = ds["PRESSURE"].values.astype(float)       # (P,)
    lat = ds["LATITUDE"].values.astype(float)     # (Y,)

    # Depth at mid-levels (m, positive down), shape (Y, P)
    d_mid = get_depth(p, lat)

    # Upper interfaces: 0 for top cell; then midpoints between adjacent mid-levels
    upper = np.empty_like(d_mid)
    upper[:, 0] = 0.0
    upper[:, 1:] = 0.5 * (d_mid[:, :-1] + d_mid[:, 1:])

    # Lower interfaces: shift upper by one; last one extended (will be clipped by h_m)
    lower = np.empty_like(d_mid)
    lower[:, :-1] = upper[:, 1:]
    lower[:, -1] = np.maximum(h_m, d_mid[:, -1])  # safe sentinel beyond h_m

    # Thickness contribution inside [0, h_m]
    thick = np.maximum(0.0, np.minimum(h_m, lower) - np.maximum(0.0, upper))  # (Y, P)

    # Return as DataArray with dims (PRESSURE, LATITUDE) to align with your variables
    w = xr.DataArray(
        thick.T,
        dims=("PRESSURE", "LATITUDE"),
        coords={"PRESSURE": ds["PRESSURE"], "LATITUDE": ds["LATITUDE"]},
        name="thickness_m",
        attrs={"units": "m", "description": f"Layer thickness within 0–{h_m} m"},
    )
    return w

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

def mean_temperature_Tm(ds_temp: xr.Dataset, h_m: float = H_M, time=None) -> xr.DataArray:
    """
    Mixed-layer mean temperature (0–h_m m) using true meters from TEOS-10.
    Returns (LATITUDE, LONGITUDE) if 'time' given; else (TIME, LATITUDE, LONGITUDE).
    """
    T_mean = ds_temp["ARGO_TEMPERATURE_MEAN"]          # (P, Y, X)
    T_anom = ds_temp["ARGO_TEMPERATURE_ANOMALY"]       # (T, P, Y, X)
    T = _full_field(T_mean, T_anom, time=time)         # (T?, P, Y, X)

    # Thickness weights in meters (P, Y) → broadcast to T dims
    w_py = _thickness_weights_m(ds_temp, h_m=h_m)      # (P, Y)
    w = w_py
    if "TIME" in T.dims:
        w = w.expand_dims(TIME=T["TIME"])
    w = w.expand_dims(LONGITUDE=T["LONGITUDE"])

    Tm = (T * w).sum(dim="PRESSURE") / h_m
    Tm.name = "T_m"
    Tm.attrs.update({
        "long_name": f"Mixed-layer mean temperature (0–{int(h_m)} m)",
        "units": "degC",
        "method": "Thickness-weighted integral in meters using TEOS-10 (gsw.z_from_p)",
    })
    return Tm

def mean_salinity_Ts(ds_sal: xr.Dataset, h_m: float = H_M, time=None) -> xr.DataArray:
    """
    Mixed-layer mean salinity (0–h_m m) using true meters from TEOS-10.
    Returns (LATITUDE, LONGITUDE) if 'time' given; else (TIME, LATITUDE, LONGITUDE).
    """
    S_mean = ds_sal["ARGO_SALINITY_MEAN"]              # (P, Y, X)
    S_anom = ds_sal["ARGO_SALINITY_ANOMALY"]           # (T, P, Y, X)
    S = _full_field(S_mean, S_anom, time=time)         # (T?, P, Y, X)

    w_py = _thickness_weights_m(ds_sal, h_m=h_m)       # (P, Y)
    w = w_py
    if "TIME" in S.dims:
        w = w.expand_dims(TIME=S["TIME"])
    w = w.expand_dims(LONGITUDE=S["LONGITUDE"])

    Sm = (S * w).sum(dim="PRESSURE") / h_m
    Sm.name = "S_m"
    Sm.attrs.update({
        "long_name": f"Mixed-layer mean salinity (0–{int(h_m)} m)",
        "units": S.attrs.get("units", "psu"),
        "method": "Thickness-weighted integral in meters using TEOS-10 (gsw.z_from_p)",
    })
    return Sm
