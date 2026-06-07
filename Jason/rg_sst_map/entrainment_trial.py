from IPython.display import display
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import gsw
from dask.diagnostics import ProgressBar
from chris_utils import make_movie

from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset, depth_dbar_to_meter, fix_longitude_coord
from utils_Tm_Sm import z_to_xarray
from utils_ekman import repeat_monthly_field_array


TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
TM_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Datasets.nc"
# T_SUB_MAX_GRAD_PATH = r"C:\Users\jason\MSciProject\Tsub_Max_Gradient_Method.nc"
MLD_DATA_PATH = r"C:\Users\jason\MSciProject\h.nc"
T_SUB_PATH = r"C:\Users\jason\MSciProject\t_sub.nc"
ENTRAINMENT_VELOCITY_PATH = r"C:\Users\jason\MSciProject\Entrainment_Vel_h.nc"




TEMP_DATA_PATH_EXTENDED = r"C:\Users\jason\MSciProject\Temperature-(2004-2025).nc"
MLD_DATA_EXTENDED_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth-(2004-2025).nc"
ENTRAINMENT_CLIM_VELOCITY_PATH_EXTENDED = r"C:\Users\jason\MSciProject\Mixed_Layer_Entrainment_Velocity-Clim_Mean.nc"
TM_DATA_PATH_EXTENDED = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature-(2004-2025).nc"
T_SUB_PATH_EXTENDED = r"C:\Users\jason\MSciProject\Sub_Layer_Temperature-(2004-2025).nc"
T_SUB_MAX_GRAD_PATH = r"C:\Users\jason\MSciProject\Sub_Layer_Temperature_Max_Gradient_Method-(2004-2025).nc"

CALCULATE_ENT_VEL = False
CALCULATE_TSUB = False


t_sub_max_grad_da = xr.open_dataset(T_SUB_MAX_GRAD_PATH, decode_times=False)["SUB_TEMPERATURE"]


Tm = xr.open_dataset(TM_DATA_PATH_EXTENDED, decode_times=False)["ML_TEMPERATURE"]
print(Tm)
# tm = t_ds["ARGO_TEMPERATURE_MEAN"].expand_dims(TIME=len(t_ds.TIME), axis=0)
# ta = t_ds["ARGO_TEMPERATURE_ANOMALY"]
# t_full = tm + ta


t_full = xr.open_dataset(TEMP_DATA_PATH_EXTENDED, decode_times=False)["TEMPERATURE"]
print(t_full)

p_temp = t_full['PRESSURE']
lat_temp = t_full['LATITUDE']

depth_temp = depth_dbar_to_meter(p_temp,lat_temp)

dz_temp = z_to_xarray(depth_temp, t_full) 
print(dz_temp)
# t_full = fix_longitude_coord(t_full)
# dz_temp = fix_longitude_coord(dz_temp)
# print(dz_temp)



# Read Data Paths
mld_da = xr.open_dataset(MLD_DATA_EXTENDED_PATH, decode_times=False)["MLD"]

# Tm_ds = xr.open_dataset(TM_DATA_PATH, decode_times=False)
t_sub_da = xr.open_dataset(T_SUB_PATH_EXTENDED, decode_times=False)["SUB_TEMPERATURE"]
print(t_sub_da)
ent_vel_da = xr.open_dataset(ENTRAINMENT_CLIM_VELOCITY_PATH_EXTENDED, decode_times=False)["MONTHLY_MEAN_w_e"]

# Extract DataArrays
# Tm = Tm_ds["MIXED_LAYER_TEMP"]
# t_sub_da = t_sub["SUB_TEMPERATURE"]


# Convert h from dbar to metres
mld_meters = -gsw.z_from_p(mld_da, mld_da.LATITUDE)
mld_meters.attrs['units'] = 'm'
mld_meters.attrs['long_name'] = ('Mixed Layer Depth (h)')




RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # average seconds in a month

def save_entrainment_velocity():
    """Save the entrainment velocity (w_e) dataset."""

    mld_ds = load_and_prepare_dataset(MLD_DATA_PATH)
    # display(mld_ds)
    mld = mld_ds['MLD']

    w_e = (
        np.gradient(mld, axis=mld.get_axis_num('TIME'))
        / SECONDS_MONTH  # convert to dbar/s
    )
    # set negative entrainment velocity to zero
    w_e = np.where(w_e < 0, 0, w_e)
    w_e_da = xr.DataArray(
        w_e,
        coords=mld.coords,
        dims=mld.dims,
        name='ENTRAINMENT_VELOCITY'
    )

    # Covert dbar/s to 
    reference_depth = 100

    z0 = gsw.z_from_p(reference_depth, w_e_da.LATITUDE)
    z1 = gsw.z_from_p(reference_depth + 1.0, w_e_da.LATITUDE)

    metres_per_dbar = np.abs(z1-z0)

    ent_vel = w_e_da * metres_per_dbar
    ent_vel.attrs['units'] = 'm/s'
    ent_vel.attrs['long_name'] = 'Entrainment Velocity in meters per second'

    # display(ent_vel)
    # visualise_dataset(
    #     w_e_da.sel(TIME=0.5, method='nearest'),
    #     cmap='Blues',
    #     # vmin=-0.5, vmax=0.5
    # )

    ent_vel_ds = ent_vel.to_dataset(name='ENTRAINMENT_VELOCITY')
    
    ent_vel_ds.to_netcdf(r"C:\Users\jason\MSciProject\Entrainment_Vel_h.nc")


local_mld_limit = mld_meters.max(dim='TIME')


def find_max_gradient_tsub(temp_profile, depth_profile, mld_value, local_limit):
    # 1. Strip NaNs to look at only real water
    valid_mask = ~np.isnan(temp_profile)
    if not np.any(valid_mask):
        return np.nan
        
    T = temp_profile[valid_mask]
    Z = depth_profile[valid_mask]
    
    # 2. Calculate gradient
    if len(T) < 2: # Need at least 2 points to find a gradient
        return np.nan
    grad = np.gradient(T, Z)
    
    # 3. Define the search window
    search_mask = (Z > mld_value) & (Z < (local_limit + 100))
    
    if not np.any(search_mask):
        # If the MLD is already at the very bottom, entrainment is over
        # Return the last valid point as Tsub
        return T[-1]
    
    # 4. Find the max grad
    grad_abs = np.abs(grad[search_mask])
    indices_in_mask = np.where(search_mask)[0]
    
    max_idx_within_mask = np.argmax(grad_abs)
    return T[indices_in_mask[max_idx_within_mask]]

def calculate_tsub_dataset(t_full, mld_meters, depth_xr):
    """
    Applies the max-gradient search across the entire 4D dataset.
    mld_meters should be (TIME, LAT, LON) already converted to meters.
    """
    # 1. Pre-calculate the local mld max
    local_max_mld = mld_meters.max(dim='TIME')

    # 2. Ensure alignment between datasets
    mld_meters = mld_meters.reindex_like(t_full, method='nearest')
    local_max_mld = local_max_mld.reindex_like(t_full, method='nearest')

    # 3. Vectorized search
    t_sub = xr.apply_ufunc(
        find_max_gradient_tsub,
        t_full,
        depth_xr,
        mld_meters,
        local_max_mld,  
        input_core_dims=[['PRESSURE'], ['PRESSURE'], [], []], 
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    t_sub.name = 'SUB_TEMPERATURE'
    return t_sub

if CALCULATE_ENT_VEL:
    print("--- Calculating Entrainment Velocity... ---")
    save_entrainment_velocity()

if CALCULATE_TSUB:
    t_profile = t_full.isel(TIME=0, LATITUDE=50, LONGITUDE=50)
    z_profile = dz_temp.isel(TIME=0, LATITUDE=50, LONGITUDE=50)
    mld_val = mld_meters.isel(TIME=0, LATITUDE=50, LONGITUDE=50)
    limit_val = local_mld_limit.isel(LATITUDE=50, LONGITUDE=50)


    t_sub_point = find_max_gradient_tsub(t_profile, z_profile, mld_val, limit_val)
    t_sub_reference_point = t_sub_da.isel(TIME=0, LATITUDE=50, LONGITUDE=50)


    print(f"--- Single Point Test (Lat Index 50, Lon Index 50) ---")
    print(f"Mixed Layer Temp (Tm): {Tm.isel(TIME=0, LATITUDE=50, LONGITUDE=50).values:.4f} °C")
    print(f"New Subsurface Temp (Tsub): {float(t_sub_point):.4f} °C")
    print(f"Old Subsurface Temp (Tsub): {float(t_sub_reference_point):.4f} °C")
    print(f"New Thermal Jump (Tsub - Tm): {float(t_sub_point - Tm.isel(TIME=0, LATITUDE=50, LONGITUDE=50)):.4f} °C")
    print(f"Old Thermal Jump (Tsub - Tm): {float(t_sub_reference_point - Tm.isel(TIME=0, LATITUDE=50, LONGITUDE=50)):.4f} °C")


    # Checking if Tsub < Tm for new method (max gradient)
    if t_sub_point < Tm.isel(TIME=0, LATITUDE=50, LONGITUDE=50):
        print("Success: Tsub is colder than the mixed layer.")
    else:
        print("Warning: Tsub is warmer than Tm. Check your search mask logic.")

    # Checking if Tsub < Tm for old method
    if t_sub_reference_point < Tm.isel(TIME=0, LATITUDE=50, LONGITUDE=50):
        print("Success: Tsub is colder than the mixed layer.")
    else:
        print("Warning: Tsub is warmer than Tm. Check your search mask logic.")



    # Actual run
    t_sub_new_da = calculate_tsub_dataset(t_full, mld_meters, dz_temp)

    print("Calculating T_sub (Maximum Gradient Method)...")


    with ProgressBar():
        t_sub_new_da.to_netcdf(r"C:\Users\jason\MSciProject\Sub_Layer_Temperature_Max_Gradient_Method-(2004-2025).nc") 


# date = 11.5
# t0 = t_sub_max_grad_da.sel(TIME=f"{date}")

# # Copy the colormap and set NaN color
# cmap = plt.get_cmap("RdYlBu_r").copy()
# cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

# plt.figure(figsize=(10,5))
# pc = plt.pcolormesh(
#     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#     cmap=cmap, shading="auto"
# )
# plt.colorbar(pc, label="T_Sub (°C)")
# plt.title(f"T_Sub - {date}")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.tight_layout()
# plt.show()


Tm_bar = get_monthly_mean(Tm)
Tm_prime = get_anomaly(Tm, Tm_bar)


t_sub_old_bar = get_monthly_mean(t_sub_da)
t_sub_old_prime = get_anomaly(t_sub_da, t_sub_old_bar)

t_sub_new_bar = get_monthly_mean(t_sub_max_grad_da)
t_sub_new_prime = get_anomaly(t_sub_max_grad_da, t_sub_new_bar)



ent_vel_bar = repeat_monthly_field_array(ent_vel_da, n_repeats=22)



delta_t_old = t_sub_old_prime - Tm_prime
delta_t_new = t_sub_new_prime - Tm_prime


ent_flux_old = RHO_O * C_O * ent_vel_bar * delta_t_old
ent_flux_new = RHO_O * C_O * ent_vel_bar * delta_t_new

# date = 17.5

# t0 = Q_ent_prime_old.sel(TIME=f"{date}")

# # Copy the colormap and set NaN color
# cmap = plt.get_cmap("RdYlBu_r").copy()
# cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

# plt.figure(figsize=(10,5))
# pc = plt.pcolormesh(
#     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#     cmap=cmap, shading="auto", vmin = -50, vmax = 50
# )
# plt.colorbar(pc, label="Flux (W/m^2)")
# plt.title(f"Entrainment Anomaly - {date}")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.tight_layout()
# plt.show()

# date = 17.5 
# t0 = Q_ent_prime_new.sel(TIME=f"{date}")

# # Copy the colormap and set NaN color
# cmap = plt.get_cmap("RdYlBu_r").copy()
# cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

# plt.figure(figsize=(10,5))
# pc = plt.pcolormesh(
#     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#     cmap=cmap, shading="auto", vmin=-50, vmax=50
# )
# plt.colorbar(pc, label="Flux (W/m^2)")
# plt.title(f"Entrainment Anomaly - {date}")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.tight_layout()
# plt.show()


output_path1 = r"C:\Users\jason\MSciProject\Entrainment_Flux_Anomaly-(2004-2025).mp4"
output_path2 = r"C:\Users\jason\MSciProject\Entrainment_Flux_Anomaly_Max_Grad_T_sub-(2004-2025).mp4"
make_movie(ent_flux_old, -50, 50, savepath=output_path1)
make_movie(ent_flux_new, -50, 50, savepath=output_path2)
