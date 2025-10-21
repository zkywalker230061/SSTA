import numpy as np
import xarray as xr
import gsw

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
               xdim: str = XDIM,
               tdim: str = TDIM) -> xr.DataArray:
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
    if tdim in T.dims and tdim not in z_new.dims:
        z_new = z_new.expand_dims({tdim: T.coords[tdim]})

    z_broadcast = z_new.broadcast_like(T)
    return z_broadcast



# Using 1D Trapezoidal rule to calculate the vertical integral
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

    # Clip data to the integration limits
    z_high = xr.zeros_like(segment_deeper) + depth_h
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
    
    # GPT aided
    # out = out.rename(f"T_upper{int(np.max(bottom))}")
    # out.attrs.update({
    # "long_name": f"Upper {int(np.max(bottom))} m mean temperature (trapezoidal)",
    # "units": T.attrs.get("units", "degC")
    # })

    return out