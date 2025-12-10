"""
Writing a function to calculate the gradients of our fields
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset


def compute_gradient_lat(
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



def compute_gradient_lon(
        field: xr.DataArray,
        R=6.4e6) -> xr.DataArray:
    """
    Compute the gradient of the field with respect to latitude and longitude
    """

    # Convert degrees to radians for calculation
    lat_rad = np.deg2rad(field['LATITUDE'])
    #lon_rad = np.deg2rad(field['Longitude'])

    # Calculate the spacing in meters
    dlon = (np.pi/180) * R * np.cos(lat_rad)
    dlon = xr.DataArray(dlon, coords=field['LATITUDE'].coords, dims=['LATITUDE'])

  

    # Masks for neighbouring points
    # This is needed to identify valid neighbouring points for gradient calculation
    valid = field.notnull() # Creates a BOOL array where valid data (ocean) is set to True
    has_west = valid.roll(LONGITUDE=1, roll_coords=False) # Cell above is ocean 
    has_east = valid.roll(LONGITUDE=-1, roll_coords=False) # Cell below is ocean
    has_west2 = valid.roll(LONGITUDE=2, roll_coords=False) # Two cells above is ocean
    has_east2 = valid.roll(LONGITUDE=-2, roll_coords=False) # Two cells below is ocean

    # We can then define our interior points 
    interior = valid & has_west & has_east # Maps out the interior points 
    # Define the edge points of our interior
    start = valid & ~has_west
    end = valid & ~has_east

    # Now we define start/end by how many valid neighbours exist ahead/behind them
    start_run_3pt = start & has_east & has_east2 # Starting point for 2nd order forward difference
    end_run_3pt = end & has_west & has_west2 # End point for 2nd order backward difference
    start_run_2pt = start & has_east & ~has_east2 # Starting point for 1st order forward difference
    end_run_2pt = end & has_west & ~has_west2 # End point for 1st order backward difference
    single = start & ~has_east # Point with no valid neighbours (will leave as NaN)

    # Precompute shifted fields for vectorised operations
    f_prev = field.roll(LONGITUDE=1, roll_coords=False)
    f_next = field.roll(LONGITUDE=-1, roll_coords=False)
    f_prev2 = field.roll(LONGITUDE=2, roll_coords=False)
    f_next2 = field.roll(LONGITUDE=-2, roll_coords=False)

    # Initialise gradient array with NaNs
    grad = xr.full_like(field, np.nan)

    # Central difference for interior points (2nd order)
    grad = grad.where(~interior, ((f_next - f_prev) / (2 * dlon)))

    # Start of runs (forward differences)
    grad = grad.where(~start_run_3pt, ((-3 * field + 4 * f_next - f_next2) / (2 * dlon)))
    grad = grad.where(~start_run_2pt, ((f_next - field) / dlon))

    # End of runs (backward differences)
    grad = grad.where(~end_run_3pt, ((3 * field - 4 * f_prev + f_prev2) / (2 * dlon)))
    grad = grad.where(~end_run_2pt, ((field - f_prev) / dlon))

    # Single points left as NaN
    return grad

all_anomalies_path = r"C:\Users\jason\MSciProject\all_anomalies.nc"
observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc"
hopohopo_path = r"C:\Users\jason\MSciProject\chris_prev_cur_scheme_denoised.nc"


# Load Data
observed_temp_ds = xr.open_dataset(observed_path, decode_times=False) 
all_anomalies = load_and_prepare_dataset(all_anomalies_path)
hopohopo = load_and_prepare_dataset(hopohopo_path)

# Extract Variables
temperature = observed_temp_ds['__xarray_dataarray_variable__']
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

schemes = {
    "Explicit": all_anomalies["EXPLICIT"],
    "Implicit": all_anomalies["IMPLICIT"],
    "Semi-Implicit": all_anomalies["SEMI_IMPLICIT"],
    "Chris Mean K": all_anomalies["CHRIS_MEAN_K"],
    "Chris Capped": all_anomalies["CHRIS_CAPPED_EXPONENT"],
    "Hopohopo": hopohopo["__xarray_dataarray_variable__"]

}

for name, data in schemes.items():
    print(f"Calculating gradients for {name}...")
    temp_monthly_mean = get_monthly_mean(data)
    temp_anomaly = get_anomaly(data, temp_monthly_mean) # This is (T_m) bar
    gradient_lat = compute_gradient_lat(temp_anomaly)
    gradient_lon = compute_gradient_lon(temp_anomaly)
    date = 0.5
    t0 = gradient_lat.sel(TIME=date)

    # Copy the colormap and set NaN color
    cmap = plt.get_cmap("nipy_spectral").copy()
    cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
        cmap=cmap, shading="auto", vmin= -1e-5, vmax=1e-5
    )
    plt.colorbar(pc, label="Temperature Gradient (°C/m)")
    plt.title(f"Mixed Layer Temperature Gradient (Lat)- {date}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()












# print('vertical_integral:\n', temp_mld_bar)
# #%% ----5. Gradient Test --------
# gradient_lat = compute_gradient_lat(temp_mld_bar)
# print('gradient_lat:\n',gradient_lat)

# gradient_lon = compute_gradient_lon(temp_mld_bar)
# print('gradient_lon:\n',gradient_lon)

# grad_mag = np.sqrt(gradient_lat**2 + gradient_lon**2)
# print('grad_mag:\n',grad_mag)


# #%%
# if __name__ == "__main__":
#     #----Plot Temperature Map----------------------------------------------------
#     date = 7
#     t0 = temp_mld_bar.sel(MONTH=date)

#     # Copy the colormap and set NaN color
#     cmap = plt.get_cmap("RdYlBu_r").copy()
#     cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

#     plt.figure(figsize=(10,5))
#     pc = plt.pcolormesh(
#         t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#         cmap=cmap, shading="auto", vmin=-5, vmax=35
#     )
#     plt.colorbar(pc, label="Mean Temperature (°C)")
#     plt.title(f"Mixed Layer Temperature - {date}")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.tight_layout()
#     plt.show()
#     #----Plot Gradient Map (Lat)----------------------------------------------------
#     t0 = gradient_lat.sel(MONTH=date)

#     # Copy the colormap and set NaN color
#     cmap = plt.get_cmap("RdYlBu_r").copy()
#     cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

#     plt.figure(figsize=(10,5))
#     pc = plt.pcolormesh(
#         t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#         cmap=cmap, shading="auto", vmin= -1e-5, vmax=1e-5
#     )
#     plt.colorbar(pc, label="Temperature Gradient (°C/m)")
#     plt.title(f"Mixed Layer Temperature Gradient (Lat)- {date}")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.tight_layout()
#     plt.show()

#     #----Plot Gradient Map (Lon)----------------------------------------------------
#     t0 = gradient_lon.sel(MONTH=7)

#     # Copy the colormap and set NaN color
#     cmap = plt.get_cmap("RdYlBu_r").copy()
#     cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

#     plt.figure(figsize=(10,5))
#     pc = plt.pcolormesh(
#         t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#         cmap=cmap, shading="auto", vmin = -1e-5, vmax=1e-5
#     )
#     plt.colorbar(pc, label="Temperature Gradient (°C/m)")
#     plt.title(f"Mixed Layer Temperature Gradient (Lon)- {date}")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.tight_layout()
#     plt.show()
