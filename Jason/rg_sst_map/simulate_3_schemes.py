#%%
# --- 1. Running Implicit Scheme ---------------------------------- 
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from chris_utils import get_anomaly, coriolis_parameter, get_month_from_time, compute_upwind_advection
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from chris_utils import remove_empty_attributes, make_movie, get_eof_with_nan_consideration
# from utils_read_nc import get_monthly_mean, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation
from utils_ekman import repeat_monthly_field_array
from tqdm import tqdm
from chris_utils import calculate_RMSE_normalised, get_clean_error_distribution, decompose_mse
from scipy.stats import skew, kurtosis, norm


plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

matplotlib.use('TkAgg')


USE_DOWNLOADED_SSH = False




added_baseline = False
testparam = False

# Compute Simulation & Simulation Variable Settings 
COMPUTE_SIM = True 
COMPUTE_IMP = True 
COMPUTE_EXP = False
COMPUTE_SIMP = False

USE_DOWNLOADED_SSH = False

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True


# Display And/Or Save Simulation Settings 
SHOW_SIMULATIONS = False
SAVE_SIMULATIONS = False

# Statistical Analysis Settings
COMPUTE_NRMSE = False
COMPUTE_PDF_ANALYSIS = False
COMPUTE_MSE_DECOMPOSITION = True
COMPUTE_PRINCIPAL_COMPONENT = True

# ----------1. Defining Physical Parameters for Simulations ---------------------
explicit_stability_limit = 1e-7

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15
g = 9.81

def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60

delta_t = month_to_second(1)



#---------- 2. Define File Paths -------------------------------------

# Observed (ARGO & Reynolds) File Paths
TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Datasets.nc"
observed_T_path_Reynold_anom = r"C:\Users\jason\MSciProject\sst_anomalies-(2004-2018).nc"
observed_T_path_Reynold = r"C:\Users\jason\MSciProject\sst_ltm.nc"

# Surface Heat Flux File Path 
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = r"C:\Users\jason\MSciProject\heat_flux_interpolated_all_contributions.nc"

# Ekman Heat Fluxes File Paths
EKMAN_ANOMALY_DATA_PATH = r"C:\Users\jason\MSciProject\Ekman_Anomaly_Full_Datasets.nc"
EKMAN_MEAN_ADVECTION_DATA_PATH = r"C:\Users\jason\MSciProject\ekman_mean_advection.nc"

# Entrainment Velocity File Path 
ENTRAINMENT_VEL_DATA_PATH = r"C:\Users\jason\MSciProject\Entrainment_Vel_h.nc"
#NEW_ENTRAINMENT_VEL_DATA_PATH = r"C:\Users\jason\MSciProject\Entrainment_Vel_New_h.nc"

# 
T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\Tsub_Max_Gradient_Method_h.nc"
NEW_T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\Tsub_Max_Gradient_Method_New_h.nc"

# MLD File Path 
H_DATA_PATH = r"C:\Users\jason\MSciProject\h.nc"

# Geostrophic Anomaly File Path
GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_calculated.nc"
SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_interpolated_grad.nc"
SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_monthly_mean_calculated_grad.nc"


#----------3. Loading Data and Defining Key Variables -----------------------------------


# Loading h bar (Mean MLD)
h_ds = xr.open_dataset(H_DATA_PATH, decode_times=False)
h_da = h_ds["MLD"]
hbar_da = get_monthly_mean(h_da) # (TIME=12, LAT=145, LON=360)

# Loading T sub
t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
t_sub_da = t_sub_ds["SUB_TEMPERATURE"] 

t_sub_mean = get_monthly_mean(t_sub_da)
t_sub_anom = get_anomaly(t_sub_ds, "SUB_TEMPERATURE", t_sub_mean)
t_sub_anom = t_sub_anom["SUB_TEMPERATURE_ANOMALY"]

t_sub_da = t_sub_anom # (TIME=180, LAT=145, LON=360)

# Observed Data (Tm) using ARGO
observed_temp_ds_full = xr.open_dataset(observed_path, decode_times=False)
observed_temp_ds = observed_temp_ds_full["MIXED_LAYER_TEMP"]
obs_temp_mean = get_monthly_mean(observed_temp_ds)
obs_temp_anom = get_anomaly(observed_temp_ds_full, "MIXED_LAYER_TEMP", obs_temp_mean)
obs_temp_anom = obs_temp_anom["MIXED_LAYER_TEMP_ANOMALY"] # (TIME=180, LAT=145, LON=360)

# Reynolds SST mean and anom
observed_temp_ds_reynold = xr.open_dataset(observed_T_path_Reynold_anom, decode_times=False)
observed_temperature_anomaly_reynold = observed_temp_ds_reynold['anom']

observed_temp_ds_reynold = xr.open_dataset(observed_T_path_Reynold, decode_times=False)
observed_temperature_mean_reynold = observed_temp_ds_reynold['sst']

# Ekman Anomaly 
ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
ekman_anomaly_da = ekman_anomaly_ds["TEMP_EKMAN_ANOM"] # (TIME=180, LAT=145, LON=360)
ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

# Ekman Mean Advection
ekman_mean_advection = xr.open_dataset(EKMAN_MEAN_ADVECTION_DATA_PATH, decode_times=False)

# Entrainment Velocity 
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] # (TIME=12, LAT=145, LON=360)
print(entrainment_vel_da)


# Unchanged Parameters for the simulation 
temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)

# Surface Heat Flux 
heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + \
                                heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'] # (TIME=180, LAT=145, LON=360)


# Geostrophic Anomaly
geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

if USE_DOWNLOADED_SSH:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_interpolated_grad.nc"
    ssh_var_name = "sla"
else:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_calculated_grad.nc"
    ssh_var_name = "ssh"
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

# Sea surface height
sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)
sea_surface_monthlymean_ds = xr.open_dataset(SEA_SURFACE_MONTHLY_MEAN_DATA_PATH, decode_times=False)


# Preprocessing Reynolds Anomaly 
mean_obs = repeat_monthly_field_array(observed_temperature_mean_reynold)

reynolds_sst = mean_obs + observed_temperature_anomaly_reynold
# reynolds_sst_y = reynolds_sst.isel(TIME=slice(12,None))

reynolds_mean = get_monthly_mean(reynolds_sst)
reynolds_mean_stacked = repeat_monthly_field_array(reynolds_mean, n_repeats=15)

reynolds_anom = reynolds_sst - reynolds_mean_stacked # mean anomalies of order 1e-6


#----------4. Initialising Lists for Different Schemes for Simulations -----------------------------

# initialise lists for temperature anomalies and entrainment fluxes for each model

implicit_model_anomalies = []
explicit_model_anomalies = []
semi_implicit_model_anomalies = []

entrainment_fluxes_implicit = []
entrainment_fluxes_explicit = []
entrainment_fluxes_semi_implicit = []
# Diagnostic storage for Fluxes (W/m^2)

flux_diag_surface = [] # I will always have surface flux set to true so no adjustment needed
if INCLUDE_EKMAN_ANOM_ADVECTION:
    flux_diag_ekman_anom = []
if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
    flux_diag_geo_anom = []
if INCLUDE_ENTRAINMENT:
    flux_diag_entrainment = []
if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
    flux_diag_total_mean_adv = []


#-----------5. Testing Drift Before Simulations -----------------------

components = {
    "Heat Flux Anomaly": surface_flux_da,
    "Ekman Anomaly": ekman_anomaly_da,
    "T Sub Anomaly": t_sub_da  
}

print("---Mean Anomaly Diagnostic----")

for name, da in components.items():
    mean_da = da.fillna(0).mean().values
    print(f"{name} Mean: {mean_da}")

#----------6. Net Velocity Setup for Total Horizontal Transport-----------------------
# Useful for Ekman Mean Advection and Geo Mean Advection
# Putting outside big for loop to combine fields so that the tiny substepping only needs to be done once

earth_radius = 6371000
latitudes = np.deg2rad(hbar_da['LATITUDE'])

dx = (2 * np.pi * earth_radius / 360) * np.cos(latitudes)
dy = (2 * np.pi * earth_radius / 360) * np.ones_like(latitudes)
dx = xr.DataArray(dx, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values}, dims=['LATITUDE'])  # convert dx, dy to xarray for use below
dy = xr.DataArray(dy, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values}, dims=['LATITUDE'])

# Combined Mean Vel Field (GEO + EKMAN)
alpha_mean = xr.zeros_like(hbar_da)
beta_mean = xr.zeros_like(hbar_da)
div_v_mean = xr.zeros_like(hbar_da)
print(alpha_mean.dims)
print(sea_surface_monthlymean_ds.dims)

if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
    alpha_mean += sea_surface_monthlymean_ds['alpha']
    beta_mean += sea_surface_monthlymean_ds['beta']
    div_v_mean += (sea_surface_monthlymean_ds["alpha_grad_long"] + 
                   sea_surface_monthlymean_ds["beta_grad_lat"])

if INCLUDE_EKMAN_MEAN_ADVECTION:
    alpha_mean += ekman_mean_advection["ekman_alpha"]
    beta_mean += ekman_mean_advection["ekman_beta"]
    div_v_mean += (ekman_mean_advection["ekman_alpha_grad_long"] + 
                   ekman_mean_advection["ekman_beta_grad_lat"])


print(f"Stability Limit for Explicit (0.9/dt): {0.9/delta_t:.2e} s^-1")
print("Testing how many points will be too fast for the implicit scheme")
stable_lim = 0.9 / delta_t
test_a = (gamma_0 / (rho_0 * c_0 * hbar_da)) + div_v_mean
exceeded_points = test_a.where((test_a > stable_lim) & (hbar_da >0)).count().values
total_points = hbar_da.where(hbar_da > 0).count().values
print(f"Percentage of Unstable Ocean: {(exceeded_points/total_points) * 100:2f}%")

#----------7. Time Stepping (Actual Running of Simulations)----------------
if COMPUTE_SIM:
    b_prev = None

    pbar = tqdm(heat_flux_anomaly_ds.TIME.values, desc="Simulating Schemes...")
    for month in pbar:
            month_in_year = get_month_from_time(month)
            current_year = 2004 + int(month // 12)
            pbar.set_description(f"Simulating Year {current_year} Month {month_in_year}") # Gemini aided (this line and previous 2 lines)
            month_in_year = get_month_from_time(month)
            if not added_baseline:  # just adds the baseline of a whole bunch of zero
                base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY'] - \
                    temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
                base = base.expand_dims(TIME=[month])
                implicit_model_anomalies.append(base) if COMPUTE_IMP == True else None
                explicit_model_anomalies.append(base) if COMPUTE_EXP == True else None
                semi_implicit_model_anomalies.append(base) if COMPUTE_SIMP == True else None
                added_baseline = True
            else:
                # Previous Month States 
                prev_imp = implicit_model_anomalies[-1].isel(TIME=-1) if COMPUTE_IMP == True else None
                prev_exp = explicit_model_anomalies[-1].isel(TIME=-1) if COMPUTE_EXP == True else None
                prev_semi = semi_implicit_model_anomalies[-1].isel(TIME=-1) if COMPUTE_SIMP == True else None

                # Current Velocities and Monthly Data 
                alpha_curr = alpha_mean.sel(MONTH=month_in_year)
                beta_curr = beta_mean.sel(MONTH=month_in_year)
                cur_hbar = hbar_da.sel(MONTH=month_in_year)
                cur_ent_vel = entrainment_vel_da.sel(MONTH=month_in_year)
                cur_tsub = t_sub_da.sel(TIME=month)

                if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
                    if COMPUTE_IMP:
                        div_imp = compute_upwind_advection(prev_imp, alpha_curr, beta_curr, dx, dy, delta_t)
                    if COMPUTE_EXP:
                        div_exp = compute_upwind_advection(prev_exp, alpha_curr, beta_curr, dx, dy, delta_t)
                    if COMPUTE_SIMP:
                        div_semi = compute_upwind_advection(prev_semi, alpha_curr, beta_curr, dx, dy, delta_t)
                else:
                    div_imp = div_exp = div_semi = xr.zeros_like(prev_imp)
                
                cur_static = xr.zeros_like(prev_imp)

                if INCLUDE_SURFACE:
                    cur_static += surface_flux_da.sel(TIME=month)
                if INCLUDE_EKMAN_ANOM_ADVECTION:
                    cur_static += ekman_anomaly_da.sel(TIME=month)
                if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
                    cur_static += geostrophic_anomaly_da.sel(TIME=month)
                if INCLUDE_ENTRAINMENT:
                    cur_static += (cur_ent_vel * cur_tsub * rho_0 * c_0)
                
                cur_b = cur_static / (rho_0 * c_0 * cur_hbar)
                cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)
                
                if INCLUDE_ENTRAINMENT:
                    cur_a += (cur_ent_vel / cur_hbar)

                div_curr = div_v_mean.sel(MONTH=month_in_year)
                cur_a_stable = (cur_a + div_curr).clip(min=0)
                # For Semi-Imp Scheme
                if b_prev is None:
                    b_prev = cur_b
                
                cur_a_exp = cur_a + div_curr.clip(-explicit_stability_limit, explicit_stability_limit)
                cur_a_exp = cur_a_exp.clip(max= 0.9/ delta_t)

                
                if COMPUTE_IMP:
                    new_imp = (prev_imp + delta_t * (cur_b - div_imp)) / (1 + delta_t * cur_a_stable)
                if COMPUTE_EXP:
                    new_exp = prev_exp + delta_t * (cur_b - cur_a_exp * prev_exp - div_exp)
                if COMPUTE_SIMP:
                    new_semi = (prev_semi + 0.5 * delta_t * (cur_b + b_prev - div_semi)) / (1 + delta_t * cur_a_stable)

                energy_factor = rho_0 * c_0 * cur_hbar

                if INCLUDE_SURFACE:
                    flux_diag_surface.append(surface_flux_da.sel(TIME=month).expand_dims(TIME=[month]))
                if INCLUDE_EKMAN_ANOM_ADVECTION:
                    flux_diag_ekman_anom.append(ekman_anomaly_da.sel(TIME=month).expand_dims(TIME=[month]))
                if INCLUDE_ENTRAINMENT:
                    q_ent = rho_0 * c_0 * cur_ent_vel * (cur_tsub - new_imp)
                    flux_diag_entrainment.append(q_ent.expand_dims(TIME=[month]))
                if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
                    flux_diag_geo_anom.append(geostrophic_anomaly_da.sel(TIME=month).expand_dims(TIME=[month]))
                if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
                    q_mean_adv = -(energy_factor * div_imp)
                    flux_diag_total_mean_adv.append(q_mean_adv.expand_dims(TIME=[month]))
                
                b_prev = cur_b
            
                if COMPUTE_IMP:
                    implicit_model_anomalies.append(new_imp.expand_dims(TIME=[month]))
                
                if COMPUTE_EXP:
                    explicit_model_anomalies.append(new_exp.expand_dims(TIME=[month]))
                if COMPUTE_SIMP:
                    semi_implicit_model_anomalies.append(new_semi.expand_dims(TIME=[month]))

    # Gemini aided 
    # --- 8. Post-Processing: Aligning to Reynolds Benchmark ---
    print("Concatenating active model results...")

    datasets_to_merge = []
    # Merge the 3 primary numerical schemes
    scheme_configs = [
    (COMPUTE_IMP, implicit_model_anomalies, "IMPLICIT"),
    (COMPUTE_EXP, explicit_model_anomalies, "EXPLICIT"),
    (COMPUTE_SIMP, semi_implicit_model_anomalies, "SEMI_IMPLICIT")
]

    for active, anomaly_list, name in scheme_configs:
        if active and anomaly_list:
            print(f"Applying {name} baseline coordinate shift (Argo -> Reynolds)...")
            
            # 1. Convert list to a temporary 3D DataArray for calculation
            raw_ds = xr.concat(
            [da.drop_vars('MONTH', errors='ignore') for da in anomaly_list], 
            dim='TIME')
            
            # 2. Step A: Internal Recenter (Removes numerical drift)
            # We calculate the climatology of the model's own output
            model_climatology = get_monthly_mean(raw_ds)
            
            # 3. Step B: Apply the shift using repeat_monthly_field_array
            # This aligns the model's 0 to the physical 0
            climatology_stacked = repeat_monthly_field_array(model_climatology, n_repeats=15)
            recentered_da = raw_ds - climatology_stacked
            
            # 4. Prepare for final merge
            final_ds = recentered_da.rename(name).to_dataset()
            datasets_to_merge.append(final_ds)
            
            # Free memory of temporary lists
            del raw_ds, recentered_da

    if datasets_to_merge:
        print(f"Merging {len(datasets_to_merge)} active schemes...")
        all_models = xr.merge(datasets_to_merge)
        all_models = remove_empty_attributes(all_models)
        del datasets_to_merge 
    else:
        print("No schemes were computed.")

    # End of Gemini aided code

    if SHOW_SIMULATIONS:
        print("Displaying Reynolds Benchmark...")
        if SAVE_SIMULATIONS:
            make_movie(observed_temperature_anomaly_reynold, -3, 3, 
                    savepath=r"C:\Users\jason\MSciProject\Observed_Reynolds_Anomaly.mp4")
        else:
            make_movie(observed_temperature_anomaly_reynold, -3, 3)

        for scheme_name in all_models.data_vars:
            data_array = all_models[scheme_name]
            print(f"Processing video for: {scheme_name}")
            
            if SAVE_SIMULATIONS:
                save_path = fr"C:\Users\jason\MSciProject\{scheme_name.capitalize()}_Scheme_Temp_Anomaly.mp4"
                print(f"Saving to {save_path}...")
                make_movie(data_array, -3, 3, savepath=save_path)
            else:
                make_movie(data_array, -3, 3)
            
    # all_models.to_netcdf(r"C:\Users\jason\MSciProject\SSTA_All_Schemes_Final.nc")

if not COMPUTE_SIM:
    test_file_path = r"C:\Users\jason\MSciProject\SSTA_All_Schemes_Final.nc"

    all_models = xr.open_dataset(test_file_path, decode_times=False)

    if SHOW_SIMULATIONS:
        print("Displaying Reynolds Benchmark...")
        if SAVE_SIMULATIONS:
            make_movie(observed_temperature_anomaly_reynold, -3, 3, 
                    savepath=r"C:\Users\jason\MSciProject\Observed_Reynolds_Anomaly.mp4")
        else:
            make_movie(observed_temperature_anomaly_reynold, -3, 3)

        for scheme_name in all_models.data_vars:
            data_array = all_models[scheme_name]
            print(f"Processing video for: {scheme_name}")
            
            if SAVE_SIMULATIONS:
                save_path = fr"C:\Users\jason\MSciProject\{scheme_name.capitalize()}_Scheme_Temp_Anomaly_PreSaved_File.mp4"
                print(f"Saving to {save_path}...")
                make_movie(data_array, -3, 3, savepath=save_path)
            else:
                make_movie(data_array, -3, 3)




# --------------------------- Statistical Analysis ----------------------------

# Redefine Scheme Dictionary 
# schemes = {
#         "Explicit": all_models["EXPLICIT"],
#         "Implicit": all_models["IMPLICIT"],
#         "Semi-Implicit": all_models["SEMI_IMPLICIT"]
#     }

if COMPUTE_NRMSE:
    for scheme_name in all_models.data_vars:
        model_da = all_models[scheme_name]
        fig, axes = plt.subplots(1, 1, figsize=(12,7))
        rmse_map, global_score = calculate_RMSE_normalised(reynolds_anom, model_da, dim='TIME')
        
        # Plotting
        # ax = plt.subplot(3, 2, i + 1)
        rmse_map.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'Arbitrary Units'}, vmin = 0, vmax = 2)
        axes.set_xlabel("Longitude")
        axes.set_ylabel("Lattitude")
        axes.set_title(f'{scheme_name} Scheme - Normalised RMSE')
        print(scheme_name, global_score)
        fig.text(
            0.99, 0.01,
            f"Gamma = {gamma_0}\n"
            f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
            f"INCLUDE_EKMAN = {INCLUDE_EKMAN_ANOM_ADVECTION}\n"
            f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
            f"INCLUDE_GEOSTROPHIC = {INCLUDE_GEOSTROPHIC_ANOM_ADVECTION}\n"
            f"INCLUDE_GEOSTROPHIC_DISPLACEMENT = {INCLUDE_GEOSTROPHIC_MEAN_ADVECTION}\n"
            f"INCLUDE_EKMAN = {INCLUDE_EKMAN_MEAN_ADVECTION}",
            ha='right', va='bottom', fontsize=18
        )
        plt.show()

if COMPUTE_PDF_ANALYSIS:
    error_distributions = {}
    print("Calculating errors for all schemes...")

    for scheme_name in all_models.data_vars:
        data = all_models[scheme_name]
        print(f"Processing {scheme_name}...")
        error_distributions[scheme_name] = get_clean_error_distribution(data, reynolds_anom)

    num_schemes = len(all_models.data_vars)
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
        mean, std = norm.fit(err_data) 
        print(name, mean, std)
        ax.text(0.9,0.01, f"Skewness = {err_data_skew:.3f} \n Kurtosis = {err_data_kurt:.3f}")


    # Turn off unused subplots
    if len(axes_flat) > num_schemes:
        for j in range(num_schemes, len(axes_flat)):
            axes_flat[j].axis('off')

    fig.text(
            0.99, 0.01,
            f"Gamma = {gamma_0}\n"
            f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
            f"INCLUDE_EKMAN = {INCLUDE_EKMAN_ANOM_ADVECTION}\n"
            f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
            f"INCLUDE_GEOSTROPHIC = {INCLUDE_GEOSTROPHIC_ANOM_ADVECTION}\n"
            f"INCLUDE_GEOSTROPHIC_DISPLACEMENT = {INCLUDE_GEOSTROPHIC_MEAN_ADVECTION}\n",
            ha='right', va='bottom', fontsize=18
        )
    plt.show()

if COMPUTE_MSE_DECOMPOSITION:
    print("Calculating MSE....")
    for scheme_name in all_models.data_vars:
        data = all_models[scheme_name]
        results = decompose_mse(reynolds_anom, data)

        bias_da = results['bias'] 
        bias_pct = results['bias_pct'] 

        var_da = results['variance']
        var_pct = results['variance_pct']

        phase_da = results['phase']
        phase_pct = results['phase_pct']

        total_mse_da = results['total']

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()  

        plot_names = ["Bias Error", "Variance Error", "Phase Error", "Total MSE"]
        maps_to_plot = [bias_da, var_da, phase_da, total_mse_da]

        mse_configs = [
            ("Bias Error", bias_da, -1e-12, 1e-12),
            ("Variance Error", var_da, 0, 1),
            ("Phase Error", phase_da, 0, 1),
            ("Total MSE", total_mse_da, 0, 1)
        ]

        mse_pct_configs = [
            ("Bias Percentage Error", bias_pct, 0, 100),
            ("Variance Percentage Error", var_pct, 0, 100),
            ("Phase Percentage Error", phase_pct, 0, 100),
            ("Total MSE", total_mse_da, 0, 1)
        ]

        for i, (name, item, vmin_val, vmax_val) in enumerate(mse_configs):
            im = item.plot(ax=axes[i], cmap='nipy_spectral', add_colorbar=True, 
                        cbar_kwargs={'label': f'{name}'},
                        vmin=vmin_val, vmax=vmax_val)
            
            axes[i].set_title(f'{plot_names[i]}')
            
        
            mean_val = item.mean().item()
            print(f"{plot_names[i]} Mean: {mean_val:.4f}")

        plt.tight_layout()
        fig.suptitle(f"{scheme_name} Scheme MSE Decomposition (Gamma={gamma_0})", fontsize=20)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()
        for i, (name, item, vmin_val, vmax_val) in enumerate(mse_pct_configs):
            im = item.plot(ax=axes[i], cmap='nipy_spectral', add_colorbar=True, 
                        cbar_kwargs={'label': f'{name}'},
                        vmin=vmin_val, vmax=vmax_val)
            
            axes[i].set_title(f'{plot_names[i]}')
            
        
            # mean_val = item.mean().item()
            # print(f"{plot_names[i]} Mean: {mean_val:.4f}")

        plt.tight_layout()
        fig.suptitle(f"{scheme_name} Scheme MSE Decomposition (Gamma={gamma_0})", fontsize=20)
        plt.tight_layout()
        plt.show() 


if COMPUTE_PRINCIPAL_COMPONENT:
    n_modes = 5
    ocean_mask = reynolds_anom.notnull().all(dim='TIME')
    
    print("Calculating EOFs for Reynolds Observations...")
    # Using Chris function to get the eof of the obs
    reconstructed_obs, exp_var_obs, pc_obs, eof_obs_da = get_eof_with_nan_consideration(
        reynolds_anom, mask=ocean_mask, modes=n_modes
    )

    for scheme_name in all_models.data_vars:
        data = all_models[scheme_name]
        
        fig, axes = plt.subplots(n_modes, 1, figsize=(12, 4 * n_modes), sharex=True)
        if n_modes == 1:
            axes = [axes]

        for m in range(n_modes):
            ax = axes[m]
            
            # psuedo pc calculation (dot product)
            model_pseudo_pc = xr.dot(
                data.fillna(0), 
                eof_obs_da.sel(MODE=m).fillna(0), 
                dims=['LATITUDE', 'LONGITUDE']
            )

            # observed pc 
            obs_pc_raw = pc_obs[:, m]

            # 4. Normalise (Z-score)
            # model_pc_norm = (model_pseudo_pc - model_pseudo_pc.mean()) / model_pseudo_pc.std()
            # obs_pc_norm = (obs_pc_raw - obs_pc_raw.mean()) / obs_pc_raw.std()

            # 5. Plotting
            time_coords = data.TIME.values 
            
            ax.plot(time_coords, obs_pc_raw, label="Observed (Reynolds)", color='black', alpha=0.6, linewidth=1.5)
            ax.plot(time_coords, model_pseudo_pc, label=f"Model ({scheme_name})", color='crimson', linestyle='--', linewidth=1.5)
            
            # 6. Correlation Statistics
            correlation = np.corrcoef(obs_pc_raw, model_pseudo_pc)[0, 1] # Use pseudo_pc directly for corr
            ax.text(0.02, 0.9, f"Mode {m} | Correlation: {correlation:.2f}", transform=ax.transAxes, fontweight='bold')
            
            # RMSE Statistics 
            raw_rmse = np.sqrt(np.mean((obs_pc_raw - model_pseudo_pc)**2))
            obs_rms = np.sqrt(np.mean(obs_pc_raw**2))
            nrmse_pc = raw_rmse / obs_rms
            ax.text(0.02, 0.8, f"PC-RMSE: {nrmse_pc:.3f}", transform=ax.transAxes)
            # Aesthetics
            ax.set_ylabel("Amplitude (σ)")
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
            ax.legend(loc='upper right', fontsize='small')
            ax.set_title(f"{scheme_name} | Principal Component Comparison: Mode {m}")

        axes[-1].set_xlabel("Time (Months from 2004)")
        plt.tight_layout()
        plt.show()
# print(observed_temperature_anomaly_reynold)
# print(observed_temperature_mean_reynold)



print(reynolds_sst.dims)
print(reynolds_mean.dims)
make_movie(reynolds_anom, -3,3)
# psuedo_mean = get_monthly_mean(observed_temperature_anomaly_reynold)
# real_anom = get_anomaly(observed_temp_ds_reynold, "SST", psuedo_mean)
# observed_temperature_anomaly_reynold = observed_temp_ds_reynold["SST_ANOMALY"]
mean_obs = reynolds_anom.mean(dim="TIME")
fig, axes = plt.subplots(1, 1, figsize=(8,5))
cmap = plt.get_cmap('RdBu_r').copy()
cmap.set_bad(color="black")
mean_obs.plot(ax=axes, cmap=cmap, cbar_kwargs={'label': '(K)'}, vmin = -1e-7, vmax = 1e-7)
axes.set_xlabel("Longitude")
axes.set_ylabel("Lattitude")
axes.set_title(f'Mean Reynolds Anomaly')
plt.tight_layout()
plt.show()


# mean_model = schemes["Implicit"].mean(dim="TIME")
# fig, axes = plt.subplots(1, 1, figsize=(8,5))
# cmap = plt.get_cmap('RdBu_r').copy()
# cmap.set_bad(color="black")
# mean_model.plot(ax=axes, cmap=cmap, cbar_kwargs={'label': '(K)'}, vmin = -0.05, vmax = 0.05)
# axes.set_xlabel("Longitude")
# axes.set_ylabel("Lattitude")
# axes.set_title(f'Implicit Scheme Anomaly')
# plt.tight_layout()
# plt.show()