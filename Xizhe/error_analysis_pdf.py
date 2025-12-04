#%%
#--- 0. Imports ------------------------------------------------------------
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset

# Configuration for visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

#%%
#--- 1. Configuration & Data Loading ---------------------------------------
all_anomalies_path = "/Users/julia/Desktop/SSTA/datasets/all_anomalies.nc"
observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
hopohopo_path = "/Users/julia/Desktop/SSTA/datasets/chris_prev_cur_scheme_denoised.nc"


# Load Data
observed_temp_ds = xr.open_dataset(observed_path, decode_times=False) 
all_anomalies = load_and_prepare_dataset(all_anomalies_path)
hopohopo = load_and_prepare_dataset(hopohopo_path)


# Extract Variables
temperature = observed_temp_ds['__xarray_dataarray_variable__']
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

# Organize schemes into a dictionary for easy iteration
# Format: { "Label Name": DataArray }
schemes = {
    "Explicit": all_anomalies["EXPLICIT"],
    "Implicit": all_anomalies["IMPLICIT"],
    "Semi-Implicit": all_anomalies["SEMI_IMPLICIT"],
    "Chris Mean K": all_anomalies["CHRIS_MEAN_K"],
    "Chris Capped": all_anomalies["CHRIS_CAPPED_EXPONENT"],
    "Hopohopo": hopohopo["__xarray_dataarray_variable__"]

}

#%%
#--- 2. Helper Functions ---------------------------------------------------

def get_clean_error_distribution(test_da, obs_da):
    """
    Calculates error (Test - Obs), slices time, and returns 
    a flattened numpy array with NaNs removed.
    """
    # 1. Align and Slice (Time slice 1:end)
    test_sliced = test_da.isel(TIME=slice(1, None))
    obs_sliced = obs_da.isel(TIME=slice(1, None))
    
    # 2. Calculate Error (xarray handles alignment automatically)
    error_da = test_sliced - obs_sliced
    
    # 3. Flatten and drop NaNs
    flat_error = error_da.values.flatten()
    return flat_error[~np.isnan(flat_error)]

#%%
#--- 3. Calculate Errors ---------------------------------------------------

# Dictionary to store processed error arrays
error_distributions = {}

print("Calculating errors...")
for name, data in schemes.items():
    print(f"Processing {name}...")
    error_distributions[name] = get_clean_error_distribution(data, temperature_anomaly)

#%%
#--- 4. Plotting: Histograms (Grid Layout) ---------------------------------

#--- 4. Plotting: Histograms with Middle 50% Region (IQR) ------------------

# Create a figure with enough subplots
num_schemes = len(schemes)
cols = 2
rows = int(np.ceil(num_schemes / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows), constrained_layout=True)
axes_flat = axes.flatten()

for i, (name, err_data) in enumerate(error_distributions.items()):
    ax = axes_flat[i]
    
    # 1. Plot Histogram
    sns.histplot(err_data, bins=100, ax=ax, stat="count", 
                 color="steelblue", alpha=0.4, edgecolor=None)
    
    # 2. Calculate Percentiles
    # The middle 50% lies between the 25th and 75th percentiles
    q25, q75 = np.percentile(err_data, [25, 75])
    
    # 3. Add Shaded Region for Middle 50%
    # axvspan draws a vertical span (rectangle) across the plot
    ax.axvspan(q25, q75, color='green', alpha=0.2, label='Middle 50% (IQR)')
    
    # Optional: Add boundary lines for the region
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5)

    ax.axvline(np.min(err_data), color='firebrick', linestyle='--', alpha=0.6, label=f'Min: {np.min(err_data):.2f}')
    ax.axvline(np.max(err_data), color='darkorange', linestyle='--', alpha=0.6, label=f'Max: {np.max(err_data):.2f}')
    
    # 4. Add Reference Lines
    ax.axvline(0, color='black', linewidth=1, alpha=0.5, label="Zero Error")

    # 5. Formatting & Annotations
    ax.set_title(f"{name} Scheme Error Distribution")
    ax.set_xlabel("Error (K)")
    ax.set_ylabel("Frequency")

    # Add text annotation to show the width of this region
    iqr_width = q75 - q25
    # Place text near the top of the range
    y_limits = ax.get_ylim()
    ax.text(5, y_limits[1]*0.5, f"IQR Width:\n{iqr_width:.3f} K", 
            horizontalalignment='center', color='darkgreen', fontweight='bold')

    ax.legend(loc='upper right')

# Turn off unused subplots
if len(axes_flat) > num_schemes:
    for j in range(num_schemes, len(axes_flat)):
        axes_flat[j].axis('off')

plt.show()

#%%
#--- 5. Plotting: Combined KDE (Log Scale) ---------------------------------

plt.figure(figsize=(12, 7))

for name, err_data in error_distributions.items():
    # KDE plot
    sns.kdeplot(err_data, bw_adjust=1.0, label=name)

# Formatting
plt.xlim(-10, 10)        
plt.yscale("log")      
plt.ylim(1e-4, 1e2)
plt.title("Comparison of Temporal Error Distributions (Log-Scale)")
plt.xlabel("Error (K)")
plt.ylabel("Probability Density (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)

plt.show()



#%%
#--- 6. Plotting: Cumulative Distribution Function (CDF) with 50% Box ------

plt.figure(figsize=(12, 7))

# Iterate through schemes and plot their CDF
for name, err_data in error_distributions.items():
    sns.ecdfplot(data=err_data, label=name, linewidth=2, alpha=0.5)

# Add reference lines
plt.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1, label="Zero Error")
plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label="50% Probability")

# Add Guidelines for the 25% - 75% probability zone
plt.axhline(0.25, color='gray', linestyle=':', alpha=0.5)
plt.axhline(0.75, color='gray', linestyle=':', alpha=0.5)
plt.text(plt.xlim()[0], 0.25, " 25%", verticalalignment='bottom', color='gray')
plt.text(plt.xlim()[0], 0.75, " 75%", verticalalignment='bottom', color='gray')

plt.axhspan(0.25, 0.75, color='grey' , alpha=0.2)

# Formatting
plt.title("Cumulative Distribution Function (CDF) of Errors")
plt.xlabel("Error (K)")
plt.ylabel("Proportion")
plt.xlim(-5, 5) # Adjust this limit based on your data range
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()
# %%
