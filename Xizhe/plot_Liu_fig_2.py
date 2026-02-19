import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from chris_utils import get_monthly_mean, get_anomaly

# --- 1. CONFIGURATION & PATHS -----------------------------------------------
observed_T_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
HEAT_FLUX_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
NEW_H_BAR_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_hbar.nc"

# Constants
rho_0 = 1025.0
c_0 = 4100.0

# --- 2. LOAD DATA -----------------------------------------------------------
print("Loading Datasets...")

# Load H (Mixed Layer Depth)
hbar_ds = xr.open_dataset(NEW_H_BAR_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

# Load T (Observed Temperature - Argo)
obs_ds = xr.open_dataset(observed_T_path, decode_times=False)
temp_raw = obs_ds["UPDATED_MIXED_LAYER_TEMP"]
temp_mean = get_monthly_mean(temp_raw)
temp_anom = get_anomaly(obs_ds, "UPDATED_MIXED_LAYER_TEMP", temp_mean)["UPDATED_MIXED_LAYER_TEMP_ANOMALY"]

# Load Q (Net Heat Flux)
q_ds = xr.open_dataset(HEAT_FLUX_PATH, decode_times=False)
q_ds['NET_HEAT_FLUX'] = q_ds['avg_slhtf'] + q_ds['avg_snlwrf'] + q_ds['avg_snswrf'] + q_ds['avg_ishf']
q_mean = get_monthly_mean(q_ds['NET_HEAT_FLUX'])
q_anom = get_anomaly(q_ds, 'NET_HEAT_FLUX', q_mean)['NET_HEAT_FLUX_ANOMALY']

# Fill NaNs
T_filled = temp_anom.fillna(0)
Q_filled = q_anom.fillna(0)

# --- 3. CALCULATE TERMS -----------------------------------------------------

# --- A. Calculate Atmospheric Damping (Lambda_a) ---
print("Calculating Damping (Lambda_a)...")

# Align Q(t) and T(t-1)
Q_t = Q_filled
T_prev = T_filled.shift(TIME=1).fillna(0) # T(t-1)
T_t = T_filled 

def calc_monthly_cov(da1, da2):
    # Group by month and calculate covariance
    
    # FIX: Extract .values to get a pure numpy array, avoiding the TypeError
    months = ((da1.TIME + 0.5) % 12).astype(int)
    months[months == 0] = 12
    
    # Pass .values to assign_coords
    da1 = da1.assign_coords(month_idx=("TIME", months.values))
    da2 = da2.assign_coords(month_idx=("TIME", months.values))
    
    # Calculate Covariance: E[XY] - E[X]E[Y]
    mean1 = da1.groupby('month_idx').mean('TIME')
    mean2 = da2.groupby('month_idx').mean('TIME')
    anom1 = da1.groupby('month_idx') - mean1
    anom2 = da2.groupby('month_idx') - mean2
    
    return (anom1 * anom2).groupby('month_idx').mean('TIME')

cov_QT = calc_monthly_cov(Q_t, T_prev)
cov_TT = calc_monthly_cov(T_t, T_prev)

# Calculate Lambda and clean up
cov_TT_safe = cov_TT.where(np.abs(cov_TT) > 1e-5)
lambda_a_monthly = -1 * (cov_QT / cov_TT_safe)

# Apply physical constraints
lambda_a_monthly = lambda_a_monthly.where(lambda_a_monthly > 0, 0)
lambda_a_monthly = lambda_a_monthly.where(lambda_a_monthly < 60, 60) 
lambda_a_monthly = lambda_a_monthly.rename({'month_idx': 'MONTH'})

# --- B. Calculate Forcing Amplitude (F') ---
print("Calculating Forcing Amplitude (F')...")

# 1. Expand Lambda to full time series
months_full = ((T_filled.TIME + 0.5) % 12).astype(int)
months_full[months_full == 0] = 12
lambda_expanded = lambda_a_monthly.sel(MONTH=xr.DataArray(months_full, coords={'TIME': T_filled.TIME}))

# 2. Calculate Residual F'
F_prime_series = Q_filled + lambda_expanded * T_filled

# 3. Calculate Monthly Standard Deviation
months_idx = ((F_prime_series.TIME + 0.5) % 12).astype(int)
months_idx[months_idx == 0] = 12
# Fix .values here as well just in case
F_prime_series = F_prime_series.assign_coords(month_idx=("TIME", months_idx.values)) 

F_std_monthly = F_prime_series.groupby('month_idx').std('TIME')
F_std_monthly = F_std_monthly.rename({'month_idx': 'MONTH'})

# --- C. Mixed Layer Depth (h) ---
# Already loaded as hbar_da

# --- 4. SEASONAL AGGREGATION ------------------------------------------------
print("Aggregating Seasons...")

def get_seasonal_mean(da_monthly):
    djf = da_monthly.sel(MONTH=[12, 1, 2]).mean(dim='MONTH')
    mam = da_monthly.sel(MONTH=[3, 4, 5]).mean(dim='MONTH')
    jja = da_monthly.sel(MONTH=[6, 7, 8]).mean(dim='MONTH')
    son = da_monthly.sel(MONTH=[9, 10, 11]).mean(dim='MONTH')
    return [djf, mam, jja, son]

seasons_F = get_seasonal_mean(F_std_monthly)
seasons_L = get_seasonal_mean(lambda_a_monthly)
seasons_H = get_seasonal_mean(hbar_da)

# --- 5. PLOTTING ------------------------------------------------------------
print("Plotting...")

fig, axes = plt.subplots(3, 4, figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree()})
season_names = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']
row_labels = ['Forcing Amplitude (F\')\n($W m^{-2}$)', 
              'Atmos. Damping ($\lambda_a$)\n($W m^{-2} K^{-1}$)', 
              'Mixed-Layer Depth (h)\n(m)']

# Settings for each row
plot_configs = [
    {'data': seasons_F, 'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 80},   # Row 1: Forcing
    {'data': seasons_L, 'cmap': 'Greens', 'vmin': 0, 'vmax': 35},   # Row 2: Damping
    {'data': seasons_H, 'cmap': 'Purples', 'vmin': 0, 'vmax': 200}  # Row 3: MLD
]

for row_idx, config in enumerate(plot_configs):
    for col_idx, season_data in enumerate(config['data']):
        ax = axes[row_idx, col_idx]
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.add_feature(cfeature.COASTLINE)
        
        mesh = ax.pcolormesh(season_data.LONGITUDE, season_data.LATITUDE, season_data, 
                             cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'],
                             transform=ccrs.PlateCarree())
        
        if row_idx == 2:
            ax.contour(season_data.LONGITUDE, season_data.LATITUDE, season_data, 
                       levels=[300, 450, 600], colors='white', linewidths=0.8,
                       transform=ccrs.PlateCarree())

        if row_idx == 0:
            ax.set_title(season_names[col_idx], fontsize=8, fontweight='bold')
        if col_idx == 0:
            ax.text(-0.25, 0.5, row_labels[row_idx], va='center', ha='center', 
                    rotation='vertical', transform=ax.transAxes, fontsize=8)

    cbar_ax = fig.add_axes([0.92, 0.65 - (row_idx * 0.27), 0.015, 0.2])
    fig.colorbar(mesh, cax=cbar_ax, label=config.get('label', ''))

plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.9)
plt.suptitle("Reproduction of Liu et al. (2023) Figure 2: Model Parameters", fontsize=14, y=0.95)
plt.show()