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
from chris_utils import calculate_RMSE_normalised, get_clean_error_distribution
from scipy.stats import skew, kurtosis, norm

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

matplotlib.use('TkAgg')


USE_DOWNLOADED_SSH = False


INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True

USE_NEW_H_BAR_NEW_T_SUB = False

added_baseline = False
testparam = False


COMPUTE_SIM = True
COMPUTE_NRMSE = True
COMPUTE_PDF_ANALYSIS = True
SHOW_SIMULATIONS = True
SAVE_SIMULATIONS = True

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

#----------4. Initialising Lists for Different Schemes for Simulations -----------------------------

# initialise lists for temperature anomalies for each model
implicit_model_anomalies = []
explicit_model_anomalies = []
semi_implicit_model_anomalies = []

# initialise lists for entrainment fluxes for each model; for categorising each component
entrainment_fluxes_implicit = []
entrainment_fluxes_explicit = []
entrainment_fluxes_semi_implicit = []

# Diagnostic storage for Fluxes (W/m^2)
flux_diag_surface = []
flux_diag_ekman_anom = []
flux_diag_geo_anom = []
flux_diag_entrainment = []
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
                implicit_model_anomalies.append(base)
                explicit_model_anomalies.append(base)
                semi_implicit_model_anomalies.append(base)
                added_baseline = True
            else:
                # Previous Month States 
                prev_imp = implicit_model_anomalies[-1].isel(TIME=-1)
                prev_exp = explicit_model_anomalies[-1].isel(TIME=-1)
                prev_semi = semi_implicit_model_anomalies[-1].isel(TIME=-1)

                # Current Velocities and Monthly Data 
                alpha_curr = alpha_mean.sel(MONTH=month_in_year)
                beta_curr = beta_mean.sel(MONTH=month_in_year)
                cur_hbar = hbar_da.sel(MONTH=month_in_year)
                cur_ent_vel = entrainment_vel_da.sel(MONTH=month_in_year)
                cur_tsub = t_sub_da.sel(TIME=month)

                if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
                    div_imp = compute_upwind_advection(prev_imp, alpha_curr, beta_curr, dx, dy, delta_t)
                    div_exp = compute_upwind_advection(prev_exp, alpha_curr, beta_curr, dx, dy, delta_t)
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

                new_imp = (prev_imp + delta_t * (cur_b - div_imp)) / (1 + delta_t * cur_a_stable)
                new_exp = prev_exp + delta_t * (cur_b - cur_a_exp * prev_exp - div_exp)
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
                implicit_model_anomalies.append(new_imp.expand_dims(TIME=[month]))
                explicit_model_anomalies.append(new_exp.expand_dims(TIME=[month]))
                semi_implicit_model_anomalies.append(new_semi.expand_dims(TIME=[month]))

    # Gemini aided 
    # --- 8. Post-Processing: Aligning to Reynolds Benchmark ---
    print("Concatenating model results...")

    # Merge the 3 primary numerical schemes
    imp_ds = xr.concat([da.drop_vars('MONTH', errors='ignore') for da in implicit_model_anomalies], 'TIME').rename("IMPLICIT")
    exp_ds = xr.concat([da.drop_vars('MONTH', errors='ignore') for da in explicit_model_anomalies], 'TIME').rename("EXPLICIT")
    semi_ds = xr.concat([da.drop_vars('MONTH', errors='ignore') for da in semi_implicit_model_anomalies], 'TIME').rename("SEMI_IMPLICIT")

    print("Merging schemes...")
    all_models = xr.merge([imp_ds, exp_ds, semi_ds])

    del imp_ds, exp_ds, semi_ds

    # Calculate the climatological offset: Argo Mean - Reynolds Mean
    # This represents the fixed physical difference between the two datasets

    if SHOW_SIMULATIONS:
        make_movie(observed_temperature_anomaly_reynold, -3, 3)
        obs_temp_mean_repeated = repeat_monthly_field_array(obs_temp_mean)
        observed_temperature_mean_reynold_repeated = repeat_monthly_field_array(observed_temperature_mean_reynold)

        delta_t_mean = obs_temp_mean_repeated - observed_temperature_mean_reynold_repeated


        print("Applying baseline coordinate shift (Argo -> Reynolds)...")
        for scheme in ["IMPLICIT", "EXPLICIT", "SEMI_IMPLICIT"]:
            # Step A: Internal Recenter (removes any model-specific numerical drift)
            model_climatology = get_monthly_mean(all_models[scheme])
            recentered_anom_ds = get_anomaly(all_models, scheme, model_climatology)
            recentered_anom = recentered_anom_ds[f"{scheme}_ANOMALY"]
            
            # Step B: Baseline shift (aligns the model's 0 to the Reynolds 0)
            # all_models[scheme] = recentered_anom + delta_t_mean
            if SAVE_SIMULATIONS:
                save_path = fr"C:\Users\jason\MSciProject\{scheme.capitalize()}_Scheme_Temp_Anomaly.mp4"
                print(f"Saving movie for {scheme} to {save_path}...")
                make_movie(recentered_anom, -3, 3, savepath=save_path)
            else:
                make_movie(all_models[f"{scheme}_ANOMALY"], -3,3)
            
        all_models = remove_empty_attributes(all_models)
    # all_models.to_netcdf(r"C:\Users\jason\MSciProject\SSTA_All_Schemes_Final.nc")

if not COMPUTE_SIM:
    test_file_path = r"C:\Users\jason\MSciProject\SSTA_All_Schemes_Final.nc"

    all_models = xr.open_dataset(test_file_path, decode_times=False)

    if SHOW_SIMULATIONS:

        if SAVE_SIMULATIONS: 
            make_movie(observed_temperature_anomaly_reynold, -3, 3, savepath=r"C:\Users\jason\MSciProject\Observed_Reynolds_Anomaly.mp4")
            make_movie(all_models["EXPLICIT_ANOMALY"], -3, 3, savepath=r"C:\Users\jason\MSciProject\Explicit_Scheme_Temp_Anomaly.mp4")
            make_movie(all_models["SEMI_IMPLICIT_ANOMALY"], -3, 3, savepath=r"C:\Users\jason\MSciProject\Semi_Implicit_Scheme_Temp_Anomaly.mp4")
            make_movie(all_models["IMPLICIT_ANOMALY"], -3, 3, savepath=r"C:\Users\jason\MSciProject\Implicit_Scheme_Temp_Anomaly.mp4")
        
        else:
            make_movie(observed_temperature_anomaly_reynold, -3, 3)
            make_movie(all_models["EXPLICIT_ANOMALY"], -3, 3)
            make_movie(all_models["SEMI_IMPLICIT_ANOMALY"], -3, 3)
            make_movie(all_models["IMPLICIT_ANOMALY"], -3, 3)

schemes = {
        "Explicit": all_models["EXPLICIT_ANOMALY"],
        "Implicit": all_models["IMPLICIT_ANOMALY"],
        "Semi-Implicit": all_models["SEMI_IMPLICIT_ANOMALY"]
    }

if COMPUTE_NRMSE:
    for scheme_name, model_da in schemes.items():
        fig, axes = plt.subplots(1, 1, figsize=(12,7))
        rmse_map, global_score = calculate_RMSE_normalised(observed_temperature_anomaly_reynold, model_da, dim='TIME')
        
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

    for name, data in schemes.items():
        print(f"Processing {name}...")
        error_distributions[name] = get_clean_error_distribution(data, observed_temperature_anomaly_reynold)

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