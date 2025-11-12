################################################
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import re
from typing import Optional



#-----2. Convert Pressure (dbar) to Depth (m)---------------------------------------------------------------------------
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
    depth_h: xr.DataArray,
    normalise: str = "available",
    zdim: str = ZDIM,
) -> xr.DataArray:
    """
    Calculate the vertical integral of a 1D field using the trapezoidal rule.
    """

    # Define top layer point for each grid point
    sea_surface = xr.zeros_like(depth_h)

    # Ensure both inputs T and Z hae the vertical dimensions last in their order of dimensions
    if zdim not in T.dims:
        raise ValueError(f"{zdim} not in T.dims")
    
     
    T_sorted = T.transpose(..., zdim)
    z_sorted = z.broadcast_like(T_sorted).transpose(..., zdim)

    # Creating segments of adjacent levels
    T_next = T_sorted.shift({zdim: -1})
    z_next = z_sorted.shift({zdim: -1})

    # Get the boundaries of each segment
    segment_shallower = np.minimum(z_sorted, z_next)
    segment_deeper = np.maximum(z_sorted, z_next)

    # Define a small number to prevent floating-point errors at the boundaries
    EPSILON = 1e-9 

    # Clip data to the integration limits
    # Add epsilon to z_high to ensure the boundary segment is included
    z_high = (xr.zeros_like(segment_deeper) + depth_h) + EPSILON
    z_low = xr.zeros_like(segment_shallower) + sea_surface
    overlap = (np.minimum(segment_deeper, z_high) - np.maximum(segment_shallower, z_low)).clip(0)
    # Compute the trapezoidal area of each segment
    segment_area = 0.5 * (T_sorted + T_next) * overlap

    # Integrate over the vertical dimension and normalise if needed 
    num = segment_area.sum(dim=zdim, skipna=True)
    den = overlap.sum(dim=zdim, skipna=True)
    if normalise == "available":
        out = (num / den).where(den != 0)
    elif normalise == "full":
        out = (num / (depth_h - sea_surface)).where(den >= (depth_h - sea_surface))
    else:
        raise ValueError("normalise must be 'available' or 'full'")

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