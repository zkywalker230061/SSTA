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
from chris_utils import format_cartopy_axis, make_movie_all_models
# from utils_read_nc import get_monthly_mean, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation
from utils_ekman import repeat_monthly_field_array
from tqdm import tqdm
from chris_utils import calculate_RMSE_normalised, get_clean_error_distribution, decompose_mse
from scipy.stats import skew, kurtosis, norm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from chris_utils import block_bootstrap_kurtosis, autocorrelation_map
import matplotlib.colors as mcolors

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

matplotlib.use('TkAgg')


USE_DOWNLOADED_SSH = False




added_baseline = False
testparam = False

# Compute Simulation & Simulation Variable Settings 
COMPUTE_SIM = True 
COMPUTE_IMP = True 
COMPUTE_EXP = True
COMPUTE_SIMP = True

USE_DOWNLOADED_SSH = False

INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = True
INCLUDE_EKMAN_MEAN_ADVECTION = True
INCLUDE_ENTRAINMENT = True
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = True
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = True


# Display And/Or Save Simulation Settings 
EQUATOR_MASK = False
SHOW_SIMULATIONS = False
SAVE_VIDEO_SIMULATIONS = False
SAVE_SIMULATION_DATA = True

# Statistical Analysis Settings
COMPUTE_NRMSE = False
COMPUTE_PDF_ANALYSIS = False
COMPUTE_MSE_DECOMPOSITION = False
COMPUTE_PRINCIPAL_COMPONENT = False
COMPUTE_SPIN_UP_PERIOD = False
COMPUTE_PDF_ANALYSIS_REPORT = False

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
# TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\RG_ArgoClim_Temperature_2019.nc"
TEMP_DATA_PATH = r"C:\Users\jason\MSciProject\Temperature-(2004-2025).nc"

# observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Datasets.nc"
observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature_Anomalies-(2004-2025).nc"

# observed_T_path_Reynold_anom = r"C:\Users\jason\MSciProject\sst_anomalies-(2004-2018).nc"
# observed_T_path_Reynold = r"C:\Users\jason\MSciProject\sst_ltm.nc"


observed_T_path_Reynold_anom = r"C:\Users\jason\MSciProject\reynolds_sst_Anomalies-(2004-2025).nc"
observed_T_path_Reynold = r"C:\Users\jason\MSciProject\reynolds_sst-(2004-2025).nc"

# Surface Heat Flux File Path 
# HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = r"C:\Users\jason\MSciProject\heat_flux_interpolated_all_contributions.nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = r"C:\Users\jason\MSciProject\Surface_Heat_Flux-(2004-2025).nc"

# Ekman Heat Fluxes File Paths
# EKMAN_ANOMALY_DATA_PATH = r"C:\Users\jason\MSciProject\Ekman_Anomaly_Full_Datasets.nc"
# EKMAN_MEAN_ADVECTION_DATA_PATH = r"C:\Users\jason\MSciProject\ekman_mean_advection.nc"

EKMAN_ANOMALY_DATA_PATH = r"C:\Users\jason\MSciProject\Simulation-Ekman_Heat_Flux-(2004-2025).nc"
EKMAN_MEAN_ADVECTION_DATA_PATH = r"C:\Users\jason\MSciProject\2025_ekman_mean_advection.nc"


# Entrainment Velocity File Path 
# ENTRAINMENT_VEL_DATA_PATH = r"C:\Users\jason\MSciProject\Entrainment_Vel_h.nc"
ENTRAINMENT_VEL_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Entrainment_Velocity-(2004-2025).nc"

# 
# T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\Tsub_Max_Gradient_Method_h.nc"
# NEW_T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\Tsub_Max_Gradient_Method_New_h.nc"
T_SUB_DATA_PATH = r"C:\Users\jason\MSciProject\Sub_Layer_Temperature_Max_Gradient_Method-(2004-2025).nc"


# MLD File Path 
# H_DATA_PATH = r"C:\Users\jason\MSciProject\h.nc"
H_DATA_PATH = r"C:\Users\jason\MSciProject\Mixed_Layer_Depth-(2004-2025).nc"


# Geostrophic Anomaly File Path
# GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_downloaded.nc"
# GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = r"C:\Users\jason\MSciProject\geostrophic_anomaly_calculated.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = r"C:\Users\jason\MSciProject\Simulation-Geostrophic_Heat_Flux-(2004-2025).nc"

# SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_interpolated_grad.nc"
# SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_monthly_mean_calculated_grad.nc"
SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = r"C:\Users\jason\MSciProject\2025_sea_surface_monthly_mean_calculated_grad.nc"

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
# observed_temp_ds = observed_temp_ds_full["MIXED_LAYER_TEMP"]
# obs_temp_mean = get_monthly_mean(observed_temp_ds)
# obs_temp_anom = get_anomaly(observed_temp_ds_full, "MIXED_LAYER_TEMP", obs_temp_mean)
# obs_temp_anom = obs_temp_anom["MIXED_LAYER_TEMP_ANOMALY"] # (TIME=180, LAT=145, LON=360)
obs_temp_anom = observed_temp_ds_full["ANOMALY_ML_TEMPERATURE"]

# Reynolds SST mean and anom
observed_temp_ds_reynold = xr.open_dataset(observed_T_path_Reynold_anom, decode_times=False)
observed_temperature_anomaly_reynold = observed_temp_ds_reynold['ANOMALY_SST'].rename("Reynolds Anomaly SST Benchmark")
observed_temperature_anomaly_reynold = observed_temperature_anomaly_reynold.where(obs_temp_anom.notnull()) # Masks extra Reynolds values 

observed_temp_ds_reynold = xr.open_dataset(observed_T_path_Reynold, decode_times=False)
observed_temperature_mean_reynold = observed_temp_ds_reynold['SST']

# Ekman Anomaly 
ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
ekman_anomaly_da = ekman_anomaly_ds["ANOMALY_EKMAN_HEAT_FLUX"] # (TIME=180, LAT=145, LON=360)
ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

# Ekman Mean Advection
ekman_mean_advection = xr.open_dataset(EKMAN_MEAN_ADVECTION_DATA_PATH, decode_times=False)

# Entrainment Velocity 
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['w_e'])
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
geostrophic_anomaly_da = geostrophic_anomaly_ds["ANOMALY_GEOSTROPHIC_HEAT_FLUX"]

# if USE_DOWNLOADED_SSH:
#     geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH, decode_times=False)
#     SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_interpolated_grad.nc"
#     ssh_var_name = "sla"
# else:
#     geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
#     SEA_SURFACE_GRAD_DATA_PATH = r"C:\Users\jason\MSciProject\sea_surface_calculated_grad.nc"
#     ssh_var_name = "ssh"
# geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

# # Sea surface height
# sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)

sea_surface_monthlymean_ds = xr.open_dataset(SEA_SURFACE_MONTHLY_MEAN_DATA_PATH, decode_times=False)


# Preprocessing Reynolds Anomaly 
# mean_obs = repeat_monthly_field_array(observed_temperature_mean_reynold)

# reynolds_sst = mean_obs + observed_temperature_anomaly_reynold
# reynolds_sst_y = reynolds_sst.isel(TIME=slice(12,None))

# reynolds_mean = get_monthly_mean(reynolds_sst)
# reynolds_mean_stacked = repeat_monthly_field_array(reynolds_mean, n_repeats=15)

# reynolds_anom = reynolds_sst - reynolds_mean_stacked # mean anomalies of order 1e-6

reynolds_anom = observed_temperature_anomaly_reynold
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


# print(f"Stability Limit for Explicit (0.9/dt): {0.9/delta_t:.2e} s^-1")
# print("Testing how many points will be too fast for the implicit scheme")
# stable_lim = 0.9 / delta_t
# test_a = (gamma_0 / (rho_0 * c_0 * hbar_da)) + div_v_mean
# exceeded_points = test_a.where((test_a > stable_lim) & (hbar_da >0)).count().values
# total_points = hbar_da.where(hbar_da > 0).count().values
# print(f"Percentage of Unstable Ocean: {(exceeded_points/total_points) * 100:2f}%")

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
                base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['TEMPERATURE'] - \
                    temperature_ds.sel(PRESSURE=2.5, TIME=month)['TEMPERATURE']
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
    (COMPUTE_EXP, explicit_model_anomalies, "Explicit Scheme"),
    (COMPUTE_SIMP, semi_implicit_model_anomalies, "Semi-Implicit Scheme"),
    (COMPUTE_IMP, implicit_model_anomalies, "Implicit Scheme")
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
            climatology_stacked = repeat_monthly_field_array(model_climatology, n_repeats=22)
            recentered_da = raw_ds - climatology_stacked
            
            # 4. Prepare for final merge
            final_ds = recentered_da.rename(name).to_dataset()

            if EQUATOR_MASK:
                mask = abs(final_ds.LATITUDE) > 15
                final_ds = final_ds.where(mask)

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

    if SAVE_SIMULATION_DATA:
        file_path = r"C:\Users\jason\MSciProject\All_Schemes_Final.nc"
        all_models.to_netcdf(path=file_path)

    if SHOW_SIMULATIONS:
        print("Displaying Reynolds Benchmark...")
        if SAVE_VIDEO_SIMULATIONS:
            make_movie(observed_temperature_anomaly_reynold, -3, 3, 
                    savepath=r"C:\Users\jason\MSciProject\Viva\Observed_Reynolds_Anomaly.mp4",
                    colorbar_label="Temperature Anomaly (K)")
        else:
            make_movie(observed_temperature_anomaly_reynold, -3, 3, colorbar_label="Temperature Anomaly (K)")

        for scheme_name in all_models.data_vars:
            data_array = all_models[scheme_name]
            print(f"Processing video for: {scheme_name}")
            
            if SAVE_VIDEO_SIMULATIONS:
                save_path = fr"C:\Users\jason\MSciProject\Viva\{scheme_name}_Scheme_Temp_Anomaly.mp4"
                print(f"Saving to {save_path}...")
                make_movie(data_array, -3, 3, savepath=save_path, colorbar_label="Temperature Anomaly (K)")
            else:
                make_movie(data_array, -3, 3, colorbar_label="Temperature Anomaly (K)")
        

        # if len(all_models.data_vars) == 3:
        #     path_to_save = r"C:\Users\jason\MSciProject\Viva\All_Schemes_Temp_Anomaly_Mask.mp4"
        #     reynolds_anom = reynolds_anom.where(abs(reynolds_anom.LATITUDE) > 15)
        #     make_movie_all_models(reynolds_anom, all_models, -3, 3, savepath=path_to_save)
    # all_models.to_netcdf(r"C:\Users\jason\MSciProject\SSTA_All_Schemes_Final.nc")

if not COMPUTE_SIM:
    test_file_path = r"C:\Users\jason\MSciProject\All_Schemes_Final.nc"

    all_models = xr.open_dataset(test_file_path, decode_times=False)

    if SHOW_SIMULATIONS:
        print("Displaying Reynolds Benchmark...")
        if SAVE_VIDEO_SIMULATIONS:
            make_movie(observed_temperature_anomaly_reynold, -3, 3, 
                    savepath=r"C:\Users\jason\MSciProject\Observed_Reynolds_Anomaly.mp4")
        else:
            make_movie(observed_temperature_anomaly_reynold, -3, 3)

        for scheme_name in all_models.data_vars:
            data_array = all_models[scheme_name]
            print(f"Processing video for: {scheme_name}")
            
            if SAVE_VIDEO_SIMULATIONS:
                save_path = fr"C:\Users\jason\MSciProject\{scheme_name}_Scheme_Temp_Anomaly_PreSaved_File.mp4"
                print(f"Saving to {save_path}...")
                make_movie(data_array, -3, 3, savepath=save_path)
            else:
                make_movie(data_array, -3, 3)

# print(all_models.sizes)
# print(f"Heat Flux Time: {len(surface_flux_da.TIME)}")
# print(f"Ekman Time: {len(ekman_anomaly_da.TIME)}")
# print(f"T-Sub Time: {len(t_sub_da.TIME)}")

# --------------------------- Statistical Analysis ----------------------------

# Redefine Scheme Dictionary 
# schemes = {
#         "Explicit": all_models["EXPLICIT"],
#         "Implicit": all_models["IMPLICIT"],
#         "Semi-Implicit": all_models["SEMI_IMPLICIT"]
#     }

if COMPUTE_NRMSE:
    num_schemes = len(all_models.data_vars)
    fig, axes = plt.subplots(num_schemes, 1,
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             figsize=(10,12), constrained_layout=True, dpi=300)
    
    div_norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)

    for i, scheme_name in enumerate(all_models.data_vars):
        model_da = all_models[scheme_name]
        ax = axes[i]
        # fig, axes = plt.subplots(1, 1, figsize=(12,7))
        # fig, axes = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10,6))
        format_cartopy_axis(ax)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
        ax.set_facecolor('#dcdcdc')
        ax.coastlines(resolution='110m', color='black', linewidth=0.8, zorder=3)
        rmse_map, global_score = calculate_RMSE_normalised(reynolds_anom, model_da, dim='TIME', ENSO_mask=False)
        mesh = rmse_map.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            norm=div_norm,
            add_labels=False,
            add_colorbar=False
            )
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title(f'{scheme_name}', fontsize=18, fontweight='bold', pad=10)
        # Stack 2D into 1D to find the global index
        stacked = rmse_map.stack(point=['LATITUDE', 'LONGITUDE'])

        # Find the point index for min and max
        min_point = stacked.idxmin().item()
        max_point = stacked.idxmax().item()

        # min_point and max_point are now tuples: (lat_val, lon_val)
        min_lat, min_lon = min_point
        max_lat, max_lon = max_point

        print(f"Min RMSE: {rmse_map.min().item():.4f} at Lat: {min_lat}, Lon: {min_lon}")
        print(f"Max RMSE: {rmse_map.max().item():.4f} at Lat: {max_lat}, Lon: {max_lon}")
        print(global_score)
        
        # Plotting
        # ax = plt.subplot(3, 2, i + 1)
        # rmse_map.plot(ax=axes, cmap='nipy_spectral', cbar_kwargs={'label': 'Arbitrary Units'}, vmin = 0, vmax = 2)
        # axes.set_xlabel("Longitude")
        # axes.set_ylabel("Lattitude")
        # axes.set_title(f'{scheme_name} Scheme - Normalised RMSE')
        
        # metadata_str = (
        #     f"Gamma = {gamma_0}\n"
        #     f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
        #     f"INCLUDE_EKMAN_ANOM = {INCLUDE_EKMAN_ANOM_ADVECTION}\n"
        #     f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
        #     f"INCLUDE_GEOSTROPHIC_ANOM = {INCLUDE_GEOSTROPHIC_ANOM_ADVECTION}\n"
        #     f"INCLUDE_GEOSTROPHIC_MEAN_ADV = {INCLUDE_GEOSTROPHIC_MEAN_ADVECTION}\n"
        #     f"INCLUDE_EKMAN_MEAN_ADV = {INCLUDE_EKMAN_MEAN_ADVECTION}"
        # )

        # fig.text(0.98, 0.02, metadata_str, ha='right', va='bottom',
        #          fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7]) # [left, bottom, width, height]
    # cbar = fig.colorbar(mesh, cax=cbar_ax)
    # cbar.set_label('Normalised RMSE (a.u.)', fontsize=12, fontweight='bold')

    cbar = fig.colorbar(mesh, ax=axes, orientation='vertical', 
                        fraction=0.046, pad=0.04, shrink=0.8, extend='max')
    cbar.set_label('Normalised RMSE (a.u.)', fontsize=14, fontweight='bold')
    #plt.savefig(r'C:\Users\jason\MSciProject\Viva\Centered_NRMSE_Maps_2.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(r'C:\Users\jason\MSciProject\Report Figures\NRMSE_All_Terms_Autosave_Border_Adjusted_2.png', bbox_inches='tight')
    plt.show()

        
if COMPUTE_PDF_ANALYSIS:
    error_distributions = {}
    print("Calculating errors for all schemes...")

    for scheme_name in all_models.data_vars:
        data = all_models[scheme_name]
        print(f"Processing {scheme_name}...")
        error_distributions[scheme_name] = get_clean_error_distribution(data, reynolds_anom, exclude_tropics=True)

    num_schemes = len(all_models.data_vars)
    cols = 1
    rows = int(np.ceil(num_schemes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows), sharex=True, constrained_layout=True)
    axes_flat = axes.flatten()

    for i, (name, err_data) in enumerate(error_distributions.items()):
        ax = axes_flat[i]
        
        # 1. High-resolution KDE
        sns.kdeplot(err_data, bw_adjust=1.5, gridsize=1000, 
                    color='royalblue', fill=True, alpha=0.2, ax=ax, label='Error PDF')

        mean, std = np.mean(err_data), np.std(err_data)
        x_range = np.linspace(-3, 3, 200)
        ax.plot(x_range, norm.pdf(x_range, mean, std), color='black', linestyle='--', alpha=0.6, label='Normal Distrubution Fit')
        # 2. Statistics
        q10, q90 = np.percentile(err_data, [10, 90])
        width = q90 - q10

        # 3. Shading and Labeling
        ax.axvspan(q10, q90, color='forestgreen', alpha=0.15, label=f'90% Range ({width:.3f} K)')
        ax.axvline(q10, color='forestgreen', linestyle='--', linewidth=1)
        ax.axvline(q90, color='forestgreen', linestyle='--', linewidth=1)

        # 4. Center the text horizontally between q10 and q90
        # Placing it slightly above the x-axis for visibility
        ax.text((q10 + q90) / 2, ax.get_ylim()[1] * 0.05, f"Range: {width:.3f} K", 
                ha='center', va='bottom', color='forestgreen', fontweight='bold', fontsize=12)

        ax.legend(loc='upper right', fontsize='small')
        ax.set_xlim(-3, 3) # Zooming in helps show off that high kurtosis!
        ax.set_title(f"Error Distribution: {name} Scheme", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        #ax.grid(axis='y', alpha=0.3)
        stats_text = (
        f"$\mu$: {mean:.3f} K\n"
        f"$\sigma$: {std:.3f} K\n"
        f"Skew: {skew(err_data):.3f}\n"
        f"Kurtosis: {kurtosis(err_data):.3f}"
        )
        print(stats_text)

    axes_flat[-1].set_xlabel("Error (K)", fontsize=16)
    # Turn off unused subplots
    if len(axes_flat) > num_schemes:
        for j in range(num_schemes, len(axes_flat)):
            axes_flat[j].axis('off')

    # fig.text(
    #         0.99, 0.01,
    #         f"Gamma = {gamma_0}\n"
    #         f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    #         f"INCLUDE_EKMAN_ANOM = {INCLUDE_EKMAN_ANOM_ADVECTION}\n"
    #         f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
    #         f"INCLUDE_GEOSTROPHIC_ANOM = {INCLUDE_GEOSTROPHIC_ANOM_ADVECTION}\n"
    #         f"INCLUDE_GEOSTROPHIC_MEAN_ADV = {INCLUDE_GEOSTROPHIC_MEAN_ADVECTION}\n"
    #         f"INCLUDE_EKMAN_MEAN_ADV = {INCLUDE_EKMAN_MEAN_ADVECTION}",
    #         ha='right', va='bottom', fontsize=18
    #     )
    plt.show()

    bootstrap_results = {}
    
    for scheme_name, err_data in error_distributions.items():
        print(f"\nBootstrapping {scheme_name}...")
        # We flatten the data and remove NaNs to ensure a clean 1D array
        clean_data = err_data[~np.isnan(err_data)]
        
        # Run the bootstrap
        bootstrap_results[scheme_name] = block_bootstrap_kurtosis(
            clean_data, 
            block_size=5000,  # Larger blocks for 14M points
            n_iterations=500
        )

    # --- Visualization of Significance ---
    plt.figure(figsize=(10, 6))
    for name, boots in bootstrap_results.items():
        sns.kdeplot(boots, label=f"{name} (Mean K: {np.mean(boots):.1f})", fill=True)
        
        # Calculate 95% Confidence Interval
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        print(f"{name} 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

    plt.title("Bootstrap Distribution of Kurtosis")
    plt.xlabel("Excess Kurtosis")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    # --- Statistical Significance Test ---
    # Difference between Semi-Implicit (90) and Implicit (67)
    if "Semi-Implicit Scheme" in bootstrap_results and "Implicit Scheme" in bootstrap_results:
        diff = bootstrap_results["Semi-Implicit Scheme"] - bootstrap_results["Implicit Scheme"]
        p_val = np.mean(diff <= 0)
        print(f"\n--- Significance Result ---")
        print(f"Empirical P-value for (Semi-Imp > Imp): {p_val}")
        if p_val < 0.05:
            print("The difference in Kurtosis is Statistically Significant.")

    # plt.figure(figsize=(12, 7))

    # for name, err_data in error_distributions.items():
    #     # KDE plot
    #     sns.kdeplot(err_data, bw_adjust=5.0, label=name)

    # # Formatting
    # plt.xlim(-3, 3)        
    # plt.yscale("log")      
    # plt.ylim(1e-4, 1e2)
    # plt.title("Comparison of Temporal Error Distributions (Log-Scale)")
    # plt.xlabel("Error (K)")
    # plt.ylabel("Probability Density (Log Scale)")
    # plt.legend()
    # plt.grid(True, which="both", ls="--", alpha=0.3)
    # fig.text(
    #         0.99, 0.01,
    #         f"Gamma = {gamma_0}\n"
    #         f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
    #         f"INCLUDE_EKMAN_ANOM = {INCLUDE_EKMAN_ANOM_ADVECTION}\n"
    #         f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
    #         f"INCLUDE_GEOSTROPHIC_ANOM = {INCLUDE_GEOSTROPHIC_ANOM_ADVECTION}\n"
    #         f"INCLUDE_GEOSTROPHIC_MEAN_ADV = {INCLUDE_GEOSTROPHIC_MEAN_ADVECTION}\n"
    #         f"INCLUDE_EKMAN_MEAN_ADV = {INCLUDE_EKMAN_MEAN_ADVECTION}",
    #         ha='right', va='bottom', fontsize=18
    #     )
    # plt.show()

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
    plt.xlim(-3, 3) # Adjust this limit based on your data range
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.text(
            0.99, 0.01,
            f"Gamma = {gamma_0}\n"
            f"INCLUDE_SURFACE = {INCLUDE_SURFACE}\n"
            f"INCLUDE_EKMAN_ANOM = {INCLUDE_EKMAN_ANOM_ADVECTION}\n"
            f"INCLUDE_ENTRAINMENT = {INCLUDE_ENTRAINMENT}\n"
            f"INCLUDE_GEOSTROPHIC_ANOM = {INCLUDE_GEOSTROPHIC_ANOM_ADVECTION}\n"
            f"INCLUDE_GEOSTROPHIC_MEAN_ADV = {INCLUDE_GEOSTROPHIC_MEAN_ADVECTION}\n"
            f"INCLUDE_EKMAN_MEAN_ADV = {INCLUDE_EKMAN_MEAN_ADVECTION}",
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
    n_modes = 1
    ocean_mask = reynolds_anom.notnull().all(dim='TIME')
    # lat_mask = (reynolds_anom.LATITUDE > 15) | (reynolds_anom.LATITUDE < -15)
    # combined_mask = ocean_mask & lat_mask
    print("Calculating EOFs for Reynolds Observations...")

    # Using Chris function to get the eof of the obs
    reconstructed_obs, exp_var_obs, pc_obs, eof_obs_da = get_eof_with_nan_consideration(
        reynolds_anom, mask=ocean_mask, modes=n_modes
    )

    for scheme_name in all_models.data_vars:
        data = all_models[scheme_name]
        fig, axes = plt.subplots(n_modes, 1, figsize=(16, 8), sharex=True)
        if n_modes == 1:
            axes = [axes]

        for m in range(n_modes):
            ax = axes[m]
            
            # Psuedo pc calculation (dot product)
            model_pseudo_pc = xr.dot(
                data.fillna(0), 
                eof_obs_da.sel(MODE=m).fillna(0), 
                dims=['LATITUDE', 'LONGITUDE']
            )

            # Observed pc 
            obs_pc_raw = pc_obs[:, m]

            # Normalise (Z-score)
            # model_pc_norm = (model_pseudo_pc - model_pseudo_pc.mean()) / model_pseudo_pc.std()
            # obs_pc_norm = (obs_pc_raw - obs_pc_raw.mean()) / obs_pc_raw.std()

            # Plotting
            time_coords = data.TIME.values 
            decimal_years = 2004 + (data.TIME.values / 12)
            
            ax.plot(decimal_years, obs_pc_raw, label="Observed (Reynolds)", color='black', alpha=0.6, linewidth=2)
            ax.plot(decimal_years, model_pseudo_pc, label=f"Model ({scheme_name})", color='crimson', linestyle='--', linewidth=2)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            # Correlation Statistics
            correlation = np.corrcoef(obs_pc_raw, model_pseudo_pc)[0, 1] # Use pseudo_pc directly for corr
            ax.text(0.02, 0.15, f"Correlation: {correlation:.2f}", transform=ax.transAxes, fontweight='bold', fontsize=16)
            
            # RMSE Statistics 
            raw_rmse = np.sqrt(np.mean((obs_pc_raw - model_pseudo_pc)**2))
            obs_rms = np.sqrt(np.mean(obs_pc_raw**2))
            nrmse_pc = raw_rmse / obs_rms
            ax.text(0.02, 0.1, f"PC-NRMSE: {nrmse_pc:.3f}", transform=ax.transAxes, fontsize=16)
            # Aesthetics
            ax.set_ylabel("Projected Anomaly (K)", fontsize=16) # need to change
            ax.axhline(0, color='gray', linewidth=1, linestyle=':')
            ax.legend(loc='upper right', fontsize=16)
            ax.set_title(f"{scheme_name} | Principal Component Comparison: Mode {m+1}", fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=14)
        axes[-1].set_xlabel("Year", fontsize=16)
        plt.tight_layout()
        output_path = fr"C:\Users\jason\MSciProject\Viva\{scheme_name}_PC_Anomaly_Test.jpg"
        #plt.savefig(output_path)

        # if len(all_models.data_vars) ==3:
        #     fig, axes = plt.subplots(1, 1, figsize=(16, 8))

        plt.show()


if COMPUTE_SPIN_UP_PERIOD:
    
    for scheme_name in all_models.data_vars:
        data = all_models[scheme_name]

        print(f"Computing Autocorrelation for {scheme_name}")
        lags, avg_acf = autocorrelation_map(data)
        plt.figure(figsize=(10, 5))
        plt.plot(lags, avg_acf, marker='o', color='navy', linewidth=2)
        
        # Threshold lines
        plt.axhline(0.368, color='red', linestyle='--', label='e-folding (1/e)')
        plt.axhline(0.05, color='green', linestyle='--', label='95% Decorrelation')
        
        plt.title(f"{scheme_name} Global Average Autocorrelation of Temperature Anomalies", fontsize=14)
        plt.xlabel("Lag (Months)", fontsize=12)
        plt.ylabel("Correlation Coefficient", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()
        

if COMPUTE_PDF_ANALYSIS_REPORT:
    # 1. Exact color mapping from your Bootstrap plot
    colors = {'Explicit': 'C0', 'Semi-Implicit': 'C1', 'Implicit': 'C2'}
    
    # Update global font sizes for high-res output
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    error_distributions = {}
    print("Calculating errors for all schemes...")

    for scheme_name in all_models.data_vars:
        data = all_models[scheme_name]
        print(f"Processing {scheme_name}...")
        error_distributions[scheme_name] = get_clean_error_distribution(data, reynolds_anom, exclude_tropics=True)

    for name, err_data in error_distributions.items():
        # Match color based on substring
        plot_color = 'black'
        for key, val in colors.items():
            if key in name:
                plot_color = val
                break

        sns.kdeplot(
            err_data, bw_adjust=1.5, gridsize=1000, 
            color=plot_color, fill=False, linewidth=3, ax=ax, 
            label=f'{name})'
        )

    # 2. FINAL FIX FOR THE GREEN LINE: Force spines to black
    # We do this AFTER all plotting calls to override the loop color
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_color('black')
    # ax.spines['bottom'].set_linewidth(2.0)
    # ax.spines['left'].set_color('black')
    # ax.spines['left'].set_linewidth(2.0)

    # 3. Formatting
    ax.set_xlim(-2.5, 2.5) 
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_title("Global Error Distribution (Residuals)", fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel("Error (K)", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16, colors='black')
    
    # Legend formatting
    ax.legend(fontsize=14, frameon=True, loc='upper right', borderpad=1)
    
    sns.despine() # Removes top and right
    plt.savefig(r'C:\Users\jason\MSciProject\Report Figures\Error_Distribution_No_Fill_All_Terms.png', dpi=300, bbox_inches='tight')
    plt.show()

    bootstrap_results = {}
    
    for scheme_name, err_data in error_distributions.items():
        print(f"\nBootstrapping {scheme_name}...")
        # We flatten the data and remove NaNs to ensure a clean 1D array
        clean_data = err_data[~np.isnan(err_data)]
        
        # Run the bootstrap
        bootstrap_results[scheme_name] = block_bootstrap_kurtosis(
            clean_data, 
            block_size=5000,  # Larger blocks for 14M points
            n_iterations=500
        )

    # --- Visualization of Significance ---
    plt.figure(figsize=(10, 6))
    for name, boots in bootstrap_results.items():
        sns.kdeplot(boots, label=f"{name} (Mean Kurtosis: {np.mean(boots):.1f})", fill=True)
        
        # Calculate 95% Confidence Interval
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        print(f"{name} 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

    plt.title("Bootstrap Distribution of Kurtosis")
    plt.xlabel("Excess Kurtosis")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(r'C:\Users\jason\MSciProject\Report Figures\Error_Distribution_Bootstrap_All_Terms.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Statistical Significance Test ---
    # Difference between Semi-Implicit (90) and Implicit (67)
    if "Semi-Implicit Scheme" in bootstrap_results and "Implicit Scheme" in bootstrap_results:
        diff = bootstrap_results["Semi-Implicit Scheme"] - bootstrap_results["Implicit Scheme"]
        p_val = np.mean(diff <= 0)
        print(f"\n--- Significance Result ---")
        print(f"Empirical P-value for (Semi-Imp > Imp): {p_val}")
        if p_val < 0.05:
            print("The difference in Kurtosis is Statistically Significant.")