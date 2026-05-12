import xarray as xr
import numpy as np
import gsw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset, depth_dbar_to_meter, fix_longitude_coord
from utils_Tm_Sm import z_to_xarray
from utils_ekman import repeat_monthly_field_array
from chris_utils import make_movie

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

#---1. Defining Data Paths ----------------------------
TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
TM_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Datasets.nc"
T_SUB_MAX_GRAD_PATH = r"C:\Users\jason\MSciProject\Tsub_Max_Gradient_Method_h.nc"
T_SUB_PATH = r"C:\Users\jason\MSciProject\t_sub.nc"
MLD_DATA_PATH = r"C:\Users\jason\MSciProject\h.nc"
ENTRAINMENT_VELOCITY_PATH = r"C:\Users\jason\MSciProject\Entrainment_Vel_h.nc"

#---2. Opening Files and Defining Variables for Analysis --------------------------

# Temperature
t_ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
tm = t_ds["ARGO_TEMPERATURE_MEAN"].expand_dims(TIME=len(t_ds.TIME), axis=0)
ta = t_ds["ARGO_TEMPERATURE_ANOMALY"]
t_full = tm + ta
t_full = fix_longitude_coord(t_full)


# Mixed Layer Temp (Tm)
Tm_ds = xr.open_dataset(TM_DATA_PATH, decode_times=False)
Tm = Tm_ds["MIXED_LAYER_TEMP"]


# Tsub (Max Grad)
t_sub_max_grad = xr.open_dataset(T_SUB_MAX_GRAD_PATH, decode_times=False)
t_sub_max_grad_da = t_sub_max_grad["SUB_TEMPERATURE"]

# Tsub (Old Method)
t_sub = xr.open_dataset(T_SUB_PATH, decode_times=False)
t_sub_da = t_sub["SUB_TEMPERATURE"]

# MLD_metres (Mixed Layer Depth in Metres)
mld_ds = xr.open_dataset(MLD_DATA_PATH, decode_times=False)
mld_da = mld_ds["MLD"]
mld_meters = -gsw.z_from_p(mld_da, mld_da.LATITUDE)
mld_meters.attrs['units'] = 'm'
mld_meters.attrs['long_name'] = ('Mixed Layer Depth (h)')

# dz_temp (4D Depth in Metres)
p_temp = t_ds['PRESSURE']
lat_temp = t_ds['LATITUDE']
depth_temp = depth_dbar_to_meter(p_temp,lat_temp)

dz_temp = z_to_xarray(depth_temp, t_full) 

# Entrainment Velocity (used to compute Entrainment Flux)
ent_vel_ds = xr.open_dataset(ENTRAINMENT_VELOCITY_PATH, decode_times=False)
ent_vel_da = ent_vel_ds["ENTRAINMENT_VELOCITY"]


#---3. Defining Locations for Analysis---------------------

year_index = 1 # 1 is for 2004 and 15 is for 2018

TIME_IDX_FEB = 1*year_index
TIME_IDX_AUG = 7*year_index

locations = {
    "Southern Ocean (SH)": {
        'coords': {'LATITUDE': -52.5, 'LONGITUDE': -95.5},
        'seasons': {'Winter': TIME_IDX_AUG, 'Summer': TIME_IDX_FEB},
        'hemisphere': 'South'
    },
    "North Atlantic 1 (NH)": {
        'coords': {'LATITUDE': 41.5, 'LONGITUDE': -50.5},
        'seasons': {'Winter': TIME_IDX_FEB, 'Summer': TIME_IDX_AUG},
        'hemisphere': 'North'
    },
    "North Atlantic 2 (NH)": {
        'coords': {'LATITUDE': 50.5, 'LONGITUDE': -25.5},
        'seasons': {'Winter': TIME_IDX_FEB, 'Summer': TIME_IDX_AUG},
        'hemisphere': 'North'
    },
    "Indian Ocean (SH)": {
        'coords': {'LATITUDE': -20.5, 'LONGITUDE': 74.5},
        'seasons': {'Winter': TIME_IDX_AUG, 'Summer': TIME_IDX_FEB},
        'hemisphere': 'SOUTH'
    },
    "North Pacific (NH)": {
        'coords': {'LATITUDE': 29.5, 'LONGITUDE': -149.5},
        'seasons': {'Winter': TIME_IDX_FEB, 'Summer': TIME_IDX_AUG},
        'hemisphere': 'North'
    },
    "Cape Agulhas (SH)": {
        'coords': {'LATITUDE': -39.5, 'LONGITUDE': 25.5},
        'seasons': {'Winter': TIME_IDX_AUG, 'Summer': TIME_IDX_FEB},
        'hemisphere': 'South'
    }
}

#---4. Defining Function to produce Tsub (Max Grad) vs Old Tsub ---------------
# Graph inspired by Fig 3. in Holte and Tolley (2009)

def plot_location_comparison(loc_name, config, t_full, tm, dz, mld, t_sub_old, t_sub_new):
    """
    Docstring for plot_location_comparison
    
    :param loc_name: Description
    :param config: Description
    :param t_full: Description
    :param dz: Description
    :param mld: Description
    :param t_sub_old: Description
    :param t_sub_new: Description
    """

    # Define coords and seasons for each location
    coords = config['coords']
    seasons = config['seasons']

    # Define fig to plot
    fig, axes = plt.subplots(1, 2, figsize=(14,8), sharey=True)

    # Define separate panel labels
    panel_labels = ['(a)', '(b)']

    # For loop 
    for i, (season_name, time_idx) in enumerate(seasons.items()):
        ax = axes[i]

        # Need to select data
        try:
            T_selected = t_full.isel(TIME=time_idx).sel(coords, method='nearest').values
            Tm_selected = tm.isel(TIME=time_idx).sel(coords, method='nearest').values
            Z_selected = dz.isel(TIME=time_idx).sel(coords, method='nearest').values
            mld_selected = mld.isel(TIME=time_idx).sel(coords, method='nearest').values
            t_sub_old_selected = t_sub_old.isel(TIME=time_idx).sel(coords, method='nearest').values
            t_sub_new_selected = t_sub_new.isel(TIME=time_idx).sel(coords, method='nearest').values
        except KeyError as e:
            print(f"Error slicing data for {loc_name}. Check variable names: {e}")
            return

        # Need to handle NaNs for land
        mask = ~np.isnan(T_selected) & ~np.isnan(Z_selected)
        if not np.any(mask) or np.isnan(mld_selected):
            ax.text(0.5, 0.5, "No Valid Data\n(Land/NaN)", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{season_name}")
            continue

        T_selected = T_selected[mask]
        Z_selected = Z_selected[mask]

        # Plot the Temperature Profile 
        ax.plot(T_selected, Z_selected, 'k-o', markersize=3, color='gray', alpha=0.5, label='Temp Profile')

        # Plot Mixed Layer Depth (MLD)
        ax.axhline(mld_selected, color='black', linestyle='--', linewidth=2, label=f'MLD ({mld_selected:.1f}m)')

        # Plot Mixed Layer Temp (Tm)
        # We plot this at a shallow constant depth (e.g., 5m) to represent the mixed layer bulk
        ax.plot(Tm_selected, 5, 'go', markersize=10, markeredgecolor='k', label=f'Observed $T_m$ ({Tm_selected:.1f}m)')

        below_mld = Z_selected > mld_selected

        # Plot Old Tsub
        # Search for the depth of the old Tsub value
        if np.any(below_mld):
            # Find the depth in the profile that matches the Old Tsub value
            idx_match_old = np.argmin(np.abs(T_selected[below_mld] - t_sub_old_selected))
            depth_old = Z_selected[below_mld][idx_match_old]
            ax.plot(t_sub_old_selected, depth_old, 'r^', markersize=12, markeredgecolor='k', label=(f'Old $T_{{sub}}$ ({depth_old:.1f}m)'))
        else:
            # Fallback if MLD is at the very bottom
            ax.plot(t_sub_old_selected, mld_selected, 'r^', markersize=12, markeredgecolor='k', label=(f'Old $T_{{sub}}$ ({mld_selected:.1f}m)'))
        # --- 5. Plot New Tsub (Max Gradient Method) ---
        # We search below the MLD to find where the profile temperature matches our calculated t_new_val
        if np.any(below_mld):
            # Find the depth in the profile that matches the Tsub value
            idx_match = np.argmin(np.abs(T_selected[below_mld] - t_sub_new_selected))
            depth_new = Z_selected[below_mld][idx_match]
            ax.plot(t_sub_new_selected, depth_new, 'bs', markersize=12, markeredgecolor='k', label=(f'New $T_{{sub}}$ ({depth_new:.1f}m)'))

        # --- Formatting ---
        ax.set_title(f"{panel_labels[i]} {season_name}", loc='left', fontsize=12)
        ax.invert_yaxis()
        ax.set_ylim(400, 0) 
        ax.set_xlabel("Temperature (°C)")
        if i == 0: ax.set_ylabel("Depth (m)")
        ax.grid(True, linestyle=':', alpha=0.6)

        # 1. INDIVIDUAL LEGENDS: Each panel gets its own legend below the axis
        # loc='upper center': The reference point of the legend box
        # bbox_to_anchor=(0.5, -0.15): Pushes the legend box down below the x-axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
                  ncol=3, fontsize=9, frameon=True, shadow=False)

    # Finalise Legend and Layout
    fig.suptitle(f"Seasonal Vertical Profiles: {loc_name}\nLat: {coords['LATITUDE']}N, Lon: {coords['LONGITUDE']}E", 
                 fontsize=14, y=0.95)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.95, wspace=0.3)    # plt.savefig(f"Comparison_{loc_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()


#---5. Plotting -------------------------
# print("Starting profile generation...")
# for loc_name, config in locations.items():
#     print(f"Processing {loc_name}...")
#     plot_location_comparison(loc_name, config, t_full, Tm, dz_temp, mld_meters, t_sub_da, t_sub_max_grad_da)

# print("Done.")


#---6. Entrainment Flux ----------------------

# Need to define physics parameters to compute entrainment flux
RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # average seconds in a month


Tm_bar = get_monthly_mean(Tm)
Tm_prime = get_anomaly(Tm, Tm_bar)


t_sub_old_bar = get_monthly_mean(t_sub_da)
t_sub_old_prime = get_anomaly(t_sub_da, t_sub_old_bar)

t_sub_new_bar = get_monthly_mean(t_sub_max_grad_da)
t_sub_new_prime = get_anomaly(t_sub_max_grad_da, t_sub_new_bar)


ent_vel_monthly = get_monthly_mean(ent_vel_da)
ent_vel_bar = repeat_monthly_field_array(ent_vel_monthly)

delta_t_old = t_sub_old_prime - Tm_prime
delta_t_new = t_sub_new_prime - Tm_prime

Q_ent_old_prime = RHO_O * C_O * ent_vel_bar * delta_t_old
Q_ent_new_prime = RHO_O * C_O * ent_vel_bar * delta_t_new

# output_path1 = r"C:\Users\jason\MSciProject\Q_Entrainment_Anomaly_Old_T_sub.mp4"
# output_path2 = r"C:\Users\jason\MSciProject\Q_Entrainment_Anomaly_Max_Grad_T_sub.mp4"        

# make_movie(Q_ent_old_prime, -50, 50, savepath=output_path1)
# make_movie(Q_ent_new_prime, -50, 50, savepath=output_path2)

# Cross-correlation

# old_cc = xr.corr(t_sub_da, Tm, dim="TIME")
# # print(old_cc)
# t0 = old_cc

# # Copy the colormap and set NaN color
# cmap = plt.get_cmap("nipy_spectral").copy()
# cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

# plt.figure(figsize=(10,5))
# pc = plt.pcolormesh(
#     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#     cmap=cmap, shading="auto"
# )
# plt.colorbar(pc, label="Mean Temperature (°C)")
# plt.title(f"Correlation (T sub Old and Tm)")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.tight_layout()
# plt.show()


# new_cc = xr.corr(t_sub_max_grad_da, Tm, dim="TIME")

# t0 = new_cc

# # Copy the colormap and set NaN color
# cmap = plt.get_cmap("nipy_spectral").copy()
# cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

# plt.figure(figsize=(10,5))
# pc = plt.pcolormesh(
#     t0["LONGITUDE"], t0["LATITUDE"], np.ma.masked_invalid(t0),
#     cmap=cmap, shading="auto"
# )
# plt.colorbar(pc, label="Mean Temperature (°C)")
# plt.title(f"Correlation (T sub Old and Tm)")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.tight_layout()
# plt.show()


# Variance between new Tsub and old Tsub

# Varience along time
var_new_tsub = t_sub_new_prime.var(dim=["LATITUDE", "LONGITUDE"])
var_old_tsub = t_sub_old_prime.var(dim=["LATITUDE", "LONGITUDE"])

var_combined = var_old_tsub + var_new_tsub
x_list = var_combined.TIME.values
y_list = var_combined.values

fig, ax = plt.subplots(1,1, figsize=(12,7))
ax.scatter(x_list, y_list)
ax.set_ylabel(f"Variance between new and old Tsub")
ax.set_xlabel("Months since January 2004")
plt.show()

# Spatial Variances
var_new_tsub = t_sub_new_prime.var(dim=["TIME"])
var_old_tsub = t_sub_old_prime.var(dim=["TIME"])

var_combined = var_old_tsub + var_new_tsub


# Copy the colormap and set NaN color
cmap = plt.get_cmap("nipy_spectral").copy()
cmap.set_bad(color="black")   # or "white", "black", (0.5,0.5,0.5,1), etc.

plt.figure(figsize=(10,5))
pc = plt.pcolormesh(
    var_combined["LONGITUDE"], var_combined["LATITUDE"], np.ma.masked_invalid(var_combined),
    cmap=cmap, shading="auto"
)
plt.colorbar(pc, label="Variance (°C)^2")
plt.title(f"Variance between T sub Old and T sub New")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

# print(t_sub_old_prime.LATITUDE.values)
# print(t_sub_new_prime.LATITUDE.values)
# print(t_full.LATITUDE.values)
# print(Tm.LATITUDE.values)