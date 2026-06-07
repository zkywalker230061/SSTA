import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from chris_utils import get_monthly_mean, get_anomaly, coriolis_parameter

# --- 1. CONFIGURATION & PATHS -----------------------------------------------
# Ensure these match your local machine
observed_T_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
HEAT_FLUX_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
EKMAN_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Anomaly_Full_Datasets.nc"
NEW_H_BAR_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_hbar.nc"
NEW_T_SUB_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_T_sub_prime.nc"
GEO_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated_2.nc"

rho_0 = 1025.0
c_0 = 4100.0
SECONDS_MONTH = 30.4375 * 24 * 60 * 60

# --- 2. LOAD DATA -----------------------------------------------------------
print("Loading Datasets...")

# Load MLD
hbar_ds = xr.open_dataset(NEW_H_BAR_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

# Load Temperature (Argo)
obs_ds = xr.open_dataset(observed_T_path, decode_times=False)
temp_raw = obs_ds["UPDATED_MIXED_LAYER_TEMP"]
temp_mean = get_monthly_mean(temp_raw)
temp_anom = get_anomaly(obs_ds, "UPDATED_MIXED_LAYER_TEMP", temp_mean)["UPDATED_MIXED_LAYER_TEMP_ANOMALY"]

# Load T_sub
tsub_ds = xr.open_dataset(NEW_T_SUB_PATH, decode_times=False)
tsub_anom = tsub_ds["ANOMALY_SUB_TEMPERATURE"]

# Load Heat Flux
q_ds = xr.open_dataset(HEAT_FLUX_PATH, decode_times=False)
q_ds['NET_HEAT_FLUX'] = q_ds['avg_slhtf'] + q_ds['avg_snlwrf'] + q_ds['avg_snswrf'] + q_ds['avg_ishf']
q_mean = get_monthly_mean(q_ds['NET_HEAT_FLUX'])
q_anom = get_anomaly(q_ds, 'NET_HEAT_FLUX', q_mean)['NET_HEAT_FLUX_ANOMALY']

# Load Ekman (Assuming W/m^2 based on previous simulation context)
ek_ds = xr.open_dataset(EKMAN_PATH, decode_times=False)
ek_anom = ek_ds['UPDATED_TEMP_EKMAN_ANOM'] 

# Load Geostrophic
geo_ds = xr.open_dataset(GEO_PATH, decode_times=False)
geo_anom = geo_ds["GEOSTROPHIC_ANOMALY"] 

# Fill NaNs
T_filled = temp_anom.fillna(0)
Q_filled = q_anom.fillna(0)
Ek_filled = ek_anom.fillna(0)
Geo_filled = geo_anom.fillna(0)
Tsub_filled = tsub_anom.fillna(0)

# Expand MLD to full time series
months_full = ((T_filled.TIME + 0.5) % 12).astype(int)
months_full[months_full == 0] = 12
H_expanded = hbar_da.sel(MONTH=xr.DataArray(months_full, coords={'TIME': T_filled.TIME})).fillna(20)

# --- 3. CALCULATE DAMPING (Lag-1 Covariance) --------------------------------
print("Calculating Damping (Lambda_a)...")

Q_t = Q_filled
T_prev = T_filled.shift(TIME=1).fillna(0)
T_t = T_filled

def calc_monthly_cov(da1, da2):
    months = ((da1.TIME + 0.5) % 12).astype(int)
    months[months == 0] = 12
    da1 = da1.assign_coords(month_idx=("TIME", months.values))
    da2 = da2.assign_coords(month_idx=("TIME", months.values))
    
    mean1 = da1.groupby('month_idx').mean('TIME')
    mean2 = da2.groupby('month_idx').mean('TIME')
    anom1 = da1.groupby('month_idx') - mean1
    anom2 = da2.groupby('month_idx') - mean2
    
    return (anom1 * anom2).groupby('month_idx').mean('TIME')

cov_QT = calc_monthly_cov(Q_t, T_prev)
cov_TT = calc_monthly_cov(T_t, T_prev)
cov_TT_safe = cov_TT.where(np.abs(cov_TT) > 1e-5)

lambda_a_monthly = -1 * (cov_QT / cov_TT_safe)
lambda_a_monthly = lambda_a_monthly.where(lambda_a_monthly > 0, 0) # Positive damping only
lambda_a_monthly = lambda_a_monthly.where(lambda_a_monthly < 60, 60) # Cap outliers
lambda_a_monthly = lambda_a_monthly.rename({'month_idx': 'MONTH'})

# Expand Lambda to full time series
Lambda_expanded = lambda_a_monthly.sel(MONTH=xr.DataArray(months_full, coords={'TIME': T_filled.TIME}))

# --- 4. CALCULATE INDIVIDUAL TERMS (W/m^2) ----------------------------------
print("Calculating Physical Terms...")

# A. Heat Flux Forcing (F')
# F' = Q_net + Lambda * T (Residual noise)
F_prime = Q_filled + Lambda_expanded * T_filled

# B. Ekman (Q_ek)
Q_ek = Ek_filled

# C. Geostrophic (Q_geo)
Q_geo = Geo_filled

# D. Entrainment (Q_ent)
# Physics: Q_ent = rho * Cp * w_e * (T_sub - T)
# w_e = dh/dt (only positive)
H_t = H_expanded
H_prev = H_expanded.shift(TIME=1).fillna(20)

# Calculate w_e using log difference (consistent with simulation)
# w_e/h = d(ln h)/dt  -> w_e = h * (1/dt) * ln(h_t / h_prev)
log_h_change = np.log(H_t / H_prev) / SECONDS_MONTH
w_e_over_h = log_h_change.where(log_h_change > 0, 0)
w_e = w_e_over_h * H_t

Q_ent = rho_0 * c_0 * w_e * (Tsub_filled - T_filled)

# --- 5. COMPUTE STATISTICS --------------------------------------------------
print("Computing Seasonal Stats...")

def calc_seasonal_std(da):
    months = ((da.TIME + 0.5) % 12).astype(int)
    months[months == 0] = 12
    da = da.assign_coords(month_idx=("TIME", months.values))
    return da.groupby('month_idx').std('TIME')

# Calculate Standard Deviations (Amplitudes)
F_std = calc_seasonal_std(F_prime)
Ek_std = calc_seasonal_std(Q_ek)
Geo_std = calc_seasonal_std(Q_geo)
Ent_std = calc_seasonal_std(Q_ent)

# Calculate Total Sum Std Dev (For Plot 2)
Total_Forcing = F_prime + Q_ek + Q_geo + Q_ent
Total_std = calc_seasonal_std(Total_Forcing)

# Helper to get 4 seasons
def get_seasonal_mean(da_monthly):
    if 'month_idx' in da_monthly.dims:
        da_monthly = da_monthly.rename({'month_idx': 'MONTH'})
    
    djf = da_monthly.sel(MONTH=[12, 1, 2]).mean(dim='MONTH')
    mam = da_monthly.sel(MONTH=[3, 4, 5]).mean(dim='MONTH')
    jja = da_monthly.sel(MONTH=[6, 7, 8]).mean(dim='MONTH')
    son = da_monthly.sel(MONTH=[9, 10, 11]).mean(dim='MONTH')
    return [djf, mam, jja, son]

# Prepare Data for Plot 1
Seas_F = get_seasonal_mean(F_std)
Seas_Ek = get_seasonal_mean(Ek_std)
Seas_Ent = get_seasonal_mean(Ent_std)
Seas_Geo = get_seasonal_mean(Geo_std)
Seas_Damp = get_seasonal_mean(lambda_a_monthly) # Mean, not Std

# Prepare Data for Plot 2
Seas_Total = get_seasonal_mean(Total_std)
Seas_H = get_seasonal_mean(hbar_da)

# --- 6. CORRECTED PLOTTING FUNCTION ----------------------------------------
def plot_grid(data_rows, row_labels, title):
    n_rows = len(data_rows)
    # Use figsize to give enough height per row
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 3.5*n_rows), subplot_kw={'projection': ccrs.PlateCarree()})
    season_names = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']
    
    # Ensure axes is always 2D array even if n_rows=1
    if n_rows == 1: axes = np.array([axes])

    for r, (row_data, config) in enumerate(data_rows):
        # Determine Color Limit for this row
        vmin = config.get('vmin', 0)
        vmax = config.get('vmax', 100)
        cmap = config.get('cmap', 'viridis')
        
        # Plot columns
        for c, season_map in enumerate(row_data):
            ax = axes[r, c]
            ax.add_feature(cfeature.LAND, facecolor='gray', zorder=1)
            ax.add_feature(cfeature.COASTLINE, zorder=2)
            
            # Pcolormesh
            mesh = ax.pcolormesh(season_map.LONGITUDE, season_map.LATITUDE, season_map, 
                                 cmap=cmap, vmin=vmin, vmax=vmax,
                                 transform=ccrs.PlateCarree())
            
            # Titles and Row Labels
            if r == 0: 
                ax.set_title(season_names[c], fontsize=8, fontweight='bold', pad=10)
            if c == 0: 
                # Row Label on the left
                ax.text(-0.2, 0.5, row_labels[r], va='center', ha='center', 
                        rotation='vertical', transform=ax.transAxes, fontsize=8, fontweight='bold')

        # --- FIX: Robust Colorbar Alignment ---
        # We attach the colorbar to the list of axes for this row (axes[r, :])
        # "fraction" controls width, "pad" controls distance from plot
        cbar = fig.colorbar(mesh, ax=axes[r, :], orientation='vertical', 
                            fraction=0.015, pad=0.02)
        cbar.set_label(config.get('unit', ''), fontsize=12)

    # Adjust layout to make room for Title
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    
    # Optional: tight_layout might help but sometimes conflicts with cartopy
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    plt.show()

# --- 7. GENERATE PLOTS ------------------------------------------------------

# PLOT 1: Individual Components
rows1 = [
    (Seas_F,    {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 80, 'unit': 'W m$^{-2}$'}),
    (Seas_Ek,   {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 40, 'unit': 'W m$^{-2}$'}),
    (Seas_Ent,  {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 40, 'unit': 'W m$^{-2}$'}),
    (Seas_Geo,  {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 40, 'unit': 'W m$^{-2}$'}),
    (Seas_Damp, {'cmap': 'Greens', 'vmin': 0, 'vmax': 35, 'unit': 'W m$^{-2}$ K$^{-1}$'})
]
labels1 = ["Heat Flux (F')", "Ekman", "Entrainment", "Geostrophic", "Damping"]
plot_grid(rows1, labels1, "Forcing Amplitudes (Std Dev) & Damping")

# PLOT 2: Total Sum
rows2 = [
    (Seas_Total, {'cmap': 'YlOrRd',  'vmin': 0, 'vmax': 100, 'unit': 'W m$^{-2}$'}),
    (Seas_Damp,  {'cmap': 'Greens',  'vmin': 0, 'vmax': 35,  'unit': 'W m$^{-2}$ K$^{-1}$'}),
    (Seas_H,     {'cmap': 'Purples', 'vmin': 0, 'vmax': 200, 'unit': 'm'})
]
labels2 = ["Total Forcing (Sum)", "Damping", "Mixed Layer Depth"]
plot_grid(rows2, labels2, "Total Forcing vs Parameters")