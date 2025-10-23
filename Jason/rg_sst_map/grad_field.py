"""
Writing a function to calculate the gradients of our fields
"""

import numpy as np
import xarray as xr


def compute_gradients(
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






