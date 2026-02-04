#%%
# --- 1. Running Implicit Scheme ---------------------------------- 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from chris_utils import make_movie, get_eof_with_nan_consideration
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.stats import kurtosis, skew

matplotlib.use('TkAgg')

INCLUDE_SURFACE = True
INCLUDE_EKMAN = True
INCLUDE_ENTRAINMENT = True
CLEAN_CHRIS_PREV_CUR = True        # only really useful when entrainment is turned on

observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = r"C:\Users\jason\MSciProject\heat_flux_interpolated_all_contributions.nc"
# HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = r"C:\Users\jason\MSciProject\Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-(2004-2018).nc"
ENTRAINMENT_VEL_DATA_PATH = r"C:\Users\jason\MSciProject\Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"
# H_BAR_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\t_sub.nc"
rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 30

temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
observed_temp_ds = xr.open_dataset(observed_path, decode_times=False)

heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + \
                                heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
ekman_anomaly_da = ekman_anomaly_ds['Q_Ek_anom']
ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD_PRESSURE"]

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
t_sub_da = t_sub_ds["T_sub_ANOMALY"]

entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']


def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60


delta_t = month_to_second(1)

# initialise lists for temperature anomalies for implicit scheme
implicit_model_anomalies = []

# initialise lists for entrainment fluxes for the implicit scheme
entrainment_fluxes_implicit = []

added_baseline = False
for month in heat_flux_anomaly_ds.TIME.values:
    # find the previous and current month from 1 to 12 to access the monthly-averaged data (hbar, entrainment vel.)
    prev_month = month - 1
    month_in_year = int((month + 0.5) % 12)
    if month_in_year == 0:
        month_in_year = 12
    prev_month_in_year = month_in_year - 1
    if prev_month_in_year == 0:
        prev_month_in_year = 12

    if not added_baseline:  # just adds the baseline of a whole bunch of zero
        base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY'] - \
               temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
        base = base.expand_dims(TIME=[month])
        implicit_model_anomalies.append(base)
        added_baseline = True

    else:
        # store previous readings Tm(n-1)
        prev_implicit_k_tm_anom = implicit_model_anomalies[-1].isel(TIME=-1)

        # get previous data
        prev_tsub_anom = t_sub_da.sel(TIME=prev_month)
        prev_heat_flux_anom = surface_flux_da.sel(TIME=prev_month)
        prev_ekman_anom = ekman_anomaly_da.sel(TIME=prev_month)
        prev_entrainment_vel = entrainment_vel_da.sel(MONTH=prev_month_in_year)
        prev_hbar = hbar_da.sel(MONTH=prev_month_in_year)

        # get current data
        cur_tsub_anom = t_sub_da.sel(TIME=month)
        cur_heat_flux_anom = surface_flux_da.sel(TIME=month)
        cur_ekman_anom = ekman_anomaly_da.sel(TIME=month)
        cur_entrainment_vel = entrainment_vel_da.sel(MONTH=month_in_year)
        cur_hbar = hbar_da.sel(MONTH=month_in_year)

        # generate the right dataset depending on whether surface flux and/or Ekman terms are desired
        if INCLUDE_SURFACE and INCLUDE_EKMAN:
            cur_surf_ek = cur_heat_flux_anom + cur_ekman_anom
            prev_surf_ek = prev_heat_flux_anom + prev_ekman_anom

        elif INCLUDE_SURFACE:
            cur_surf_ek = cur_heat_flux_anom
            prev_surf_ek = prev_heat_flux_anom

        elif INCLUDE_EKMAN:
            cur_surf_ek = cur_ekman_anom
            prev_surf_ek = prev_ekman_anom

        else:       # just a way to get a zero dataset
            cur_surf_ek = cur_ekman_anom - cur_ekman_anom
            prev_surf_ek = prev_ekman_anom - prev_ekman_anom

        if INCLUDE_ENTRAINMENT:
            cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar) + cur_entrainment_vel / cur_hbar * cur_tsub_anom
            cur_a = cur_entrainment_vel / cur_hbar + gamma_0 / (rho_0 * c_0 * cur_hbar)
            cur_k = (gamma_0 / (rho_0 * c_0) + cur_entrainment_vel) / cur_hbar

            prev_b = prev_surf_ek / (rho_0 * c_0 * prev_hbar) + prev_entrainment_vel / prev_hbar * prev_tsub_anom
            prev_a = prev_entrainment_vel / prev_hbar + gamma_0 / (rho_0 * c_0 * prev_hbar)
            prev_k = (gamma_0 / (rho_0 * c_0) + prev_entrainment_vel) / prev_hbar
        else:
            cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar)
            cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)
            cur_k = cur_a

            prev_b = prev_surf_ek / (rho_0 * c_0 * prev_hbar)
            prev_a = gamma_0 / (rho_0 * c_0 * prev_hbar)
            prev_k = prev_a

        

        # update anomalies
        cur_implicit_k_tm_anom = (prev_implicit_k_tm_anom + delta_t * cur_b) / (1 + delta_t * cur_a)

        # reformat and save model
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
        cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.expand_dims(TIME=[month])
        implicit_model_anomalies.append(cur_implicit_k_tm_anom)

        # get entrainment flux components; for categorising each component
        if INCLUDE_ENTRAINMENT:
            entrainment_flux_implicit = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_implicit_k_tm_anom)
            entrainment_fluxes_implicit.append(entrainment_flux_implicit)


# concatenate anomalies into a ds
implicit_model_anomaly_ds = xr.concat(implicit_model_anomalies, 'TIME')

# rename all models
implicit_model_anomaly_ds = implicit_model_anomaly_ds.rename("IMPLICIT")

# # clean up prev_cur model
# if CLEAN_CHRIS_PREV_CUR:
#     all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = all_anomalies_ds["CHRIS_PREV_CUR"].where((all_anomalies_ds["CHRIS_PREV_CUR"] > -10) & (all_anomalies_ds["CHRIS_PREV_CUR"] < 10))
#     n_modes = 20
#     monthly_mean = get_monthly_mean(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"])
#     map_mask = temperature_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5)
#     eof_ds, variance, PCs = get_eof_with_nan_consideration(all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"], map_mask, modes=n_modes, monthly_mean_ds=None, tolerance=1e-4)
#     all_anomalies_ds["CHRIS_PREV_CUR_CLEAN"] = eof_ds.rename("CHRIS_PREV_CUR_CLEAN")

# save
# all_anomalies_ds.to_netcdf("../datasets/all_anomalies_10.nc")

# format entrainment flux datasets
if INCLUDE_ENTRAINMENT:
    entrainment_flux_implicit_ds = xr.concat(entrainment_fluxes_implicit, 'TIME')
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.drop_vars(["MONTH", "PRESSURE"])
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    entrainment_flux_implicit_ds = entrainment_flux_implicit_ds.rename("ENTRAINMENT_FLUX_IMPLICIT_ANOMALY")


# merge the relevant fluxes into a single dataset
flux_components_to_merge = []
if INCLUDE_SURFACE:
    surface_flux_da = surface_flux_da.rename("SURFACE_FLUX_ANOMALY")
    flux_components_to_merge.append(surface_flux_da)
if INCLUDE_EKMAN:
    ekman_anomaly_da = ekman_anomaly_da.rename("EKMAN_FLUX_ANOMALY")
    flux_components_to_merge.append(ekman_anomaly_da)
if INCLUDE_ENTRAINMENT:
    flux_components_to_merge.append(entrainment_flux_implicit_ds)

flux_components_ds = xr.merge(flux_components_to_merge)

#%%
#--- 2. Prepare Observed Temperature Anomaly --------------------

# Extract Variables
observed_temperature_monthly_average = get_monthly_mean(observed_temp_ds['__xarray_dataarray_variable__'])
observed_temperature_anomaly = get_anomaly(observed_temp_ds, '__xarray_dataarray_variable__', observed_temperature_monthly_average)
observed_temperature_anomaly = observed_temperature_anomaly['__xarray_dataarray_variable___ANOMALY']


#%%
#--- 3. Helper Functions ---------------------------------------------------

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


def plot_pdf(err_data, name="Implicit"):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.kdeplot(err_data, bw_adjust=1.0, label=name, ax=ax)
    
    # 2. Calculate Percentiles
    # The middle 50% lies between the 25th and 75th percentiles
    q25, q75 = np.percentile(err_data, [25, 75])
    
    # 3. Add Shaded Region for Middle 50%
    # axvspan draws a vertical span (rectangle) across the plot
    ax.axvspan(q25, q75, color='green', alpha=0.2, label='Middle 50% (IQR)')
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5)

    ax.axvline(np.min(err_data), color='firebrick', linestyle='--', alpha=0.6, label=f'Min: {np.min(err_data):.2e}')
    ax.axvline(np.max(err_data), color='darkorange', linestyle='--', alpha=0.6, label=f'Max: {np.max(err_data):.2e}')
    
    # 4. Add Reference Lines
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

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
    ax.legend(loc='upper right', fontsize = 'x-small')

    #skewness and kurtosis
    err_data_skew = skew(err_data, axis=0, bias=True)
    err_data_kurt = kurtosis(err_data, axis=0, bias=True)
    ax.text(0.9,0.3, f"Skewness = {err_data_skew:.3f} \n Kurtosis = {err_data_kurt:.3f}")

    fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
    )

    plt.show()

def plot_log_pdf(err_data, name="Implicit"):
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    sns.kdeplot(err_data, bw_adjust=1.0, label=name, ax=ax)
    
    # 2. Calculate Percentiles
    # The middle 50% lies between the 25th and 75th percentiles
    q25, q75 = np.percentile(err_data, [25, 75])
    
    # 3. Add Shaded Region for Middle 50%
    # axvspan draws a vertical span (rectangle) across the plot
    ax.axvspan(q25, q75, color='green', alpha=0.2, label='Middle 50% (IQR)')
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5)

    ax.axvline(np.min(err_data), color='firebrick', linestyle='--', alpha=0.6, label=f'Min: {np.min(err_data):.2e}')
    ax.axvline(np.max(err_data), color='darkorange', linestyle='--', alpha=0.6, label=f'Max: {np.max(err_data):.2e}')
    
    # 4. Add Reference Lines
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

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
    ax.legend(loc='upper right', fontsize = 'x-small')

    #skewness and kurtosis
    err_data_skew = skew(err_data, axis=0, bias=True)
    err_data_kurt = kurtosis(err_data, axis=0, bias=True)
    ax.text(0.9,0.3, f"Skewness = {err_data_skew:.3f} \n Kurtosis = {err_data_kurt:.3f}")
    plt.xlim(-10, 10)        
    plt.yscale("log")      
    plt.ylim(1e-4, 1e2)
    plt.title("Implicit Scheme Error Distribution (Log-Scale)")
    plt.xlabel("Error (K)")
    plt.ylabel("Probability Density (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
    )

    plt.show()

#%% 
#--- 4. Getting Seasonal Datasets-------------------------------------------------

summer_months_north_index = []
for i in range(13):
    summer_months_north = (17.5 + i*12, 18.5 + i*12, 19.5 + i*12)
    summer_months_north_index.extend(summer_months_north)



obs_summer_north_ds = observed_temperature_anomaly.sel(TIME=summer_months_north_index, method="nearest")
obs_summer_north_ds = obs_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

imp_summer_north_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_north_index, method="nearest")
imp_summer_north_ds = imp_summer_north_ds.sel(LATITUDE=slice(0, 79.5))

# Summer months for the Southern Hemisphere 

summer_months_south_index = []
for i in range(13):
    summer_months_south = (11.5 + i*12, 12.5 + i*12, 13.5 +i*12)
    summer_months_south_index.extend(summer_months_south)


obs_summer_south_ds = observed_temperature_anomaly.sel(TIME=summer_months_south_index, method="nearest")
obs_summer_south_ds = obs_summer_south_ds.sel(LATITUDE=slice(-64.5,0))

imp_summer_south_ds = implicit_model_anomaly_ds.sel(TIME=summer_months_south_index, method="nearest")
imp_summer_south_ds = imp_summer_south_ds.sel(LATITUDE=slice(-64.5, 0))


# Winter Seasonal Analysis 

winter_months_north_index = summer_months_south_index

obs_winter_north_ds = observed_temperature_anomaly.sel(TIME=winter_months_north_index, method="nearest")
obs_winter_north_ds = obs_winter_north_ds.sel(LATITUDE=slice(0,79.5))

imp_winter_north_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_north_index, method="nearest")
imp_winter_north_ds = imp_winter_north_ds.sel(LATITUDE=slice(0,79.5))

winter_months_south_index = summer_months_north_index

obs_winter_south_ds = observed_temperature_anomaly.sel(TIME=winter_months_south_index, method="nearest")
obs_winter_south_ds = obs_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))

imp_winter_south_ds = implicit_model_anomaly_ds.sel(TIME=winter_months_south_index, method="nearest")
imp_winter_south_ds = imp_winter_south_ds.sel(LATITUDE=slice(-64.5, 0))


#%%

#--- 5. Full Distribution Error Analysis ---------------------------------------------

imp_error = get_clean_error_distribution(implicit_model_anomaly_ds, observed_temperature_anomaly)
print(imp_error)
plot_pdf(imp_error)
plot_log_pdf(imp_error)

#%%
#--- 6. Seasonal Distribution Error Analysis ---------------------------------------------------

# Dictionary to store processed error arrays
error_distributions = {}

print("Calculating errors...")

schemes = {
    "Summer (Northern Hemisphere)": imp_summer_north_ds,
    "Summer (Southern Hemisphere)": imp_summer_south_ds,
    "Winter (Northern Hemisphere)": imp_winter_north_ds,
    "Winter (Southern Hemisphere)": imp_winter_south_ds
}

observations = {
    "Summer (Northern Hemisphere)": obs_summer_north_ds,
    "Summer (Southern Hemisphere)": obs_summer_south_ds,
    "Winter (Northern Hemisphere)": obs_winter_north_ds,
    "Winter (Southern Hemisphere)": obs_winter_south_ds
}

for name, data in schemes.items():
    print(f"Processing {name}...")
    error_dataset = data - observations[name]
    flat_error = error_dataset.values.flatten()
    flat_error = flat_error[~np.isnan(flat_error)]
    error_distributions[name] = flat_error


# Create a figure with enough subplots
num_schemes = len(schemes)
cols = 2
rows = int(np.ceil(num_schemes / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows), constrained_layout=True)
axes_flat = axes.flatten()

for i, (name, err_data) in enumerate(error_distributions.items()):
    ax = axes_flat[i]
    
    # 1. Plot Histogram
    sns.kdeplot(err_data, bw_adjust=1.0, label=name, ax=ax)
    
    # 2. Calculate Percentiles
    # The middle 50% lies between the 25th and 75th percentiles
    q25, q75 = np.percentile(err_data, [25, 75])
    
    # 3. Add Shaded Region for Middle 50%
    # axvspan draws a vertical span (rectangle) across the plot
    ax.axvspan(q25, q75, color='green', alpha=0.2, label='Middle 50% (IQR)')
    ax.axvline(q25, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(q75, color='green', linestyle=':', linewidth=1.5)

    ax.axvline(np.min(err_data), color='firebrick', linestyle='--', alpha=0.6, label=f'Min: {np.min(err_data):.2e}')
    ax.axvline(np.max(err_data), color='darkorange', linestyle='--', alpha=0.6, label=f'Max: {np.max(err_data):.2e}')
    
    # 4. Add Reference Lines
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)

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
    ax.legend(loc='upper right', fontsize = 'x-small')

    #skewness and kurtosis
    err_data_skew = skew(err_data, axis=0, bias=True)
    err_data_kurt = kurtosis(err_data, axis=0, bias=True)
    ax.text(0.7,0.2, f"Skewness = {err_data_skew:.3f} \n Kurtosis = {err_data_kurt:.3f}")


# Turn off unused subplots
if len(axes_flat) > num_schemes:
    for j in range(num_schemes, len(axes_flat)):
        axes_flat[j].axis('off')

fig.text(
    0.99, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
)
plt.show()

#%%
#--- 7. Seasonal Distribution Error Analysis Combined PDF Plot (Log Scale) ---------------------------------

plt.figure(figsize=(12, 7))

for name, err_data in error_distributions.items():
    # KDE plot
    sns.kdeplot(err_data, bw_adjust=1, label=name)

# Formatting
plt.xlim(-7.5, 7.5)        
plt.yscale("log")      
plt.ylim(1e-4, 1e1)
plt.title("Comparison of Error Distributions (Log-Scale)")
plt.xlabel("Error (K)")
plt.ylabel("Probability Density (Log Scale)")
plt.legend()
plt.text(
    7.5, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=12
)
plt.show()



#%%
#--- 8. Seasonal Distribution Error Analysis (CDF) ------

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
plt.title("Cumulative Distribution Function (CDF) of Seasonal Errors")
plt.xlabel("Error (K)")
plt.ylabel("Proportion")
plt.xlim(-3, 3) # Adjust this limit based on your data range
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(
    2, 0.01,
    f"Gamma = {gamma_0}\n"
    f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    f"INCLUDE_EKMAN = {INCLUDE_EKMAN}\n"
    f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}",
    ha='right', va='bottom', fontsize=10
)
plt.show()

# %%
