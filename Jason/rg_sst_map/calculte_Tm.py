import numpy as np
import xarray as xr
import gsw

h_meters = 100.0
ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
T_VAR = "ARGO_TEMPERATURE_MEAN"
S_VAR = "ARGO_SALINITY_MEAN"
T_VAR_ANOMALY = "ARGO_TEMPERATURE_ANOMALY"

# Using 1D Trapezoidal rule to calculate the vertical integral
def vertical_integral(
    T: xr.DataArray,
    z: xr.DataArray,
    top: float = 2.5,
    bottom: float = -h_meters,
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
    z_sorted = z.transpose(..., zdim)

    # Creating segments of adjacent levels
    T_next = T_sorted.shift({zdim: -1})
    z_next = z_sorted.shift({zdim: -1})

    # Get the boundaries of each segment
    segment_top = xr.ufuncs.maximum(z_sorted, z_next)
    segment_bottom = xr.ufuncs.minimum(z_sorted, z_next)

    # Clip data to the integration limits
    z_high = xr.zeros_like(segment_top) + top
    z_low = xr.zeros_like(segment_bottom) + bottom
    overlap = (xr.ufuncs.minimum(segment_top, z_high) - xr.ufuncs.maximum(segment_bottom, z_low)).clip(0)

    # Compute the trapezoidal area of each segment
    segment_area = 0.5 * (T_sorted + T_next) * overlap

    # Integrate over the vertical dimension and normalise if needed 
    num = segment_area.sum(dim=zdim, skipna=True)
    den = overlap.sum(dim=zdim, skipna=True)
    if normalise == "available":
        out = (num / den).where(den != 0)
    elif normalise == "full":
        out = (num / (top - bottom)).where(den >= (top - bottom))
    else:
        raise ValueError("normalise must be 'available' or 'full'")
    
    # GPT aided
    out = out.rename(f"T_upper{int(abs(bottom))}")
    out.attrs.update({
    "long_name": f"Upper {int(abs(bottom))} m mean temperature (trapezoidal)",
    "units": T.attrs.get("units", "degC")
    })
    
    return out
