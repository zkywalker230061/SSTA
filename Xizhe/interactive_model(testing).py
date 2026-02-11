import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import xarray as xr
import numpy as np
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, coriolis_parameter
from scipy.stats import t


observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
observed_path_Reynold = "/Users/julia/Desktop/SSTA/datasets/Reynold_sst_anomalies-(2004-2018).nc"

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Anomaly_Full_Datasets.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"

# H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/hbar.nc"
NEW_H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_hbar.nc"

T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/t_sub.nc"
NEW_T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_T_sub_prime.nc"

GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated_2.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"

USE_DOWNLOADED_SSH = False
USE_NEW_H_BAR_NEW_T_SUB = True


observed_temp_ds_reynold = xr.open_dataset(observed_path_Reynold, decode_times=False)
observed_temperature_anomaly_reynold = observed_temp_ds_reynold['anom']

if USE_NEW_H_BAR_NEW_T_SUB:
    # New h bar
    hbar_ds = xr.open_dataset(NEW_H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    # New t sub
    t_sub_ds = xr.open_dataset(NEW_T_SUB_DATA_PATH, decode_times=False)
    t_sub_da = t_sub_ds["ANOMALY_SUB_TEMPERATURE"]

    observed_temp_ds_argo = xr.open_dataset(observed_path, decode_times=False)
    observed_temp_ds = observed_temp_ds_argo["UPDATED_MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_argo, "UPDATED_MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly_argo = observed_temperature_anomaly["UPDATED_MIXED_LAYER_TEMP_ANOMALY"]

    # Ekman Anomaly using new h
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds['UPDATED_TEMP_EKMAN_ANOM']
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)


else:
    # New "old" h bar
    hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    # New "old" t sub
    t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
    t_sub_da = t_sub_ds["SUB_TEMPERATURE"]

    t_sub_mean = get_monthly_mean(t_sub_da)
    t_sub_anom = get_anomaly(t_sub_ds, "SUB_TEMPERATURE", t_sub_mean)
    t_sub_anom = t_sub_anom["SUB_TEMPERATURE_ANOMALY"]

    t_sub_da = t_sub_anom

    observed_temp_ds_argo = xr.open_dataset(observed_path, decode_times=False)
    observed_temp_ds = observed_temp_ds_argo["MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_argo, "MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly_argo = observed_temperature_anomaly["MIXED_LAYER_TEMP_ANOMALY"]

    # Ekman Anomaly using new "old" h
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds["TEMP_EKMAN_ANOM"]
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)


temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)

heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + \
                                heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

if USE_DOWNLOADED_SSH:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"
    ssh_var_name = "sla"
else:
    geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
    SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_calculated_grad.nc"
    ssh_var_name = "ssh"
geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)


# --- 1. The Interactive Class ------------------------------------------------
class InteractiveSSTModel:
    def __init__(self, obs_anom_argo, obs_anom_reynolds, heat_flux, ekman, entrain_vel, t_sub, hbar, geo_anom, sea_surf_grad, ssh_var):
        # Store Datasets
        self.obs_argo = obs_anom_argo
        self.obs_reynolds = obs_anom_reynolds
        self.obs_anom = self.obs_argo # Default to Argo
        
        self.heat_flux = heat_flux
        self.ekman = ekman
        self.entrain_vel = entrain_vel
        self.t_sub = t_sub
        self.hbar = hbar
        self.geo_anom = geo_anom
        self.sea_surf_grad = sea_surf_grad
        self.ssh_var = ssh_var
        
        # INITIAL Parameters
        self.keys = [
            'INCLUDE_SURFACE', 
            'INCLUDE_EKMAN', 
            'INCLUDE_ENTRAINMENT', 
            'INCLUDE_GEO_ANOM', 
            'INCLUDE_GEO_MEAN', 
            'USE_OBS_REYNOLD',
            'USE_INITIAL_ANOM' # <--- NEW KEY
            ]
        self.params = {
            'INCLUDE_SURFACE': True, 
            'INCLUDE_EKMAN': True, 
            'INCLUDE_ENTRAINMENT': True,
            'INCLUDE_GEO_ANOM': False, 
            'INCLUDE_GEO_MEAN': False, 
            'USE_OBS_REYNOLD': False,
            'USE_INITIAL_ANOM': False, # <--- NEW PARAM DEFAULT
            'gamma': 15,
            }
        
        # Storage for calculated data
        self.model_anom = None
        
        # Corr Data
        self.da_corr = None
        self.da_corr_max_val = None
        self.da_corr_best_lag = None
        
        # Corr Data (Significant)
        self.da_corr_sig = None
        self.da_sig_max_val = None
        self.da_sig_best_lag = None
        
        # RMSE Data
        self.da_rmse_norm = None
        self.da_rmse_abs = None
        self.da_rmse_seasonal = None
        
        self.mesh = None
        
        # --- Figure Setup ---
        self.fig = plt.figure(figsize=(15, 9)) 
        
        self.ax_map         = self.fig.add_axes([0.1, 0.35, 0.8, 0.6]) 
        self.ax_check       = self.fig.add_axes([0.05, 0.05, 0.12, 0.25]) # Increased height slightly
        self.ax_radio_main  = self.fig.add_axes([0.18, 0.05, 0.15, 0.2]) 
        self.ax_radio_sub   = self.fig.add_axes([0.34, 0.05, 0.10, 0.2]) 
        
        self.ax_gamma       = self.fig.add_axes([0.48, 0.05, 0.35, 0.03])
        self.ax_lag         = self.fig.add_axes([0.48, 0.15, 0.35, 0.03])
        self.ax_run         = self.fig.add_axes([0.88, 0.05, 0.08, 0.05])
        
        self.create_widgets()
        self.run_simulation(None)

    def create_widgets(self):
        self.ax_check.set_title("Modelling Physics", loc='left', fontsize=11, fontweight='bold')
        labels = list(self.params.keys())[:-1] # Exclude 'gamma'
        actives = [self.params[k] for k in labels]
        self.check = CheckButtons(self.ax_check, labels, actives)
        for lbl in self.check.labels: lbl.set_fontsize(8)
        
        self.display_options_main = ('Correlation', 'Correlation (95% Sig)', 'RMSE (Norm)', 'RMSE (Abs)', 'RMSE (Seasonal)')
        self.radio_main = RadioButtons(self.ax_radio_main, self.display_options_main, active=0)
        self.radio_main.on_clicked(self.update_display_mode)

        self.display_options_sub = ('Specific Lag', 'Max Value', 'Lag of Max')
        self.radio_sub = RadioButtons(self.ax_radio_sub, self.display_options_sub, active=0)
        self.radio_sub.on_clicked(self.update_display_mode)
        self.ax_radio_sub.set_title("Corr View", fontsize=10)

        self.s_gamma = Slider(self.ax_gamma, 'Gamma', 0, 100, valinit=self.params['gamma'], valstep=1)
        self.s_lag = Slider(self.ax_lag, 'Lag (Months)', -12, 12, valinit=0, valstep=1)
        self.b_run = Button(self.ax_run, 'Recalculate', color='lightblue', hovercolor='0.975')
        
        self.b_run.on_clicked(self.run_simulation)
        self.s_lag.on_changed(self.update_map_visuals) 

    def run_simulation(self, event):
        print("Running Simulation...")
        # Update Params
        status = self.check.get_status()
        for k, s in zip(self.keys, status):
            self.params[k] = s

        self.params['gamma'] = self.s_gamma.val
        
        # --- Physics Loop ---
        rho_0 = 1025.0
        c_0 = 4100.0
        g = 9.81
        gamma_0 = self.params['gamma']
        delta_t = 30.4375 * 24 * 3600
        
        implicit_model_anomalies = []
        added_baseline = False

        for month in self.heat_flux.TIME.values:
            month_in_year = int((month + 0.5) % 12)
            if month_in_year == 0: month_in_year = 12
            
            if not added_baseline:
                # --- NEW LOGIC: USE_INITIAL_ANOM ---
                if self.params['USE_INITIAL_ANOM']:
                    # Start simulation with the first anomaly.
                    base = self.obs_argo.sel(TIME=month, method='nearest')
                    base = base.fillna(0)                     # Safety: fill NaNs with 0 (land/mask mismatches)
                else:
                    # Default: Start simulation with 0
                    base = self.heat_flux.sel(TIME=month) * 0 
                
                base = base.expand_dims(TIME=[month])
                implicit_model_anomalies.append(base)
                added_baseline = True
            else:
                prev_implicit_k_tm_anom_at_cur_loc = implicit_model_anomalies[-1].isel(TIME=-1)
                
                if self.params['INCLUDE_GEO_MEAN']:
                    f = coriolis_parameter(self.sea_surf_grad['LATITUDE'])
                    grad_long = self.sea_surf_grad[self.ssh_var + '_anomaly_grad_long'].sel(TIME=month)
                    grad_lat = self.sea_surf_grad[self.ssh_var + '_anomaly_grad_lat'].sel(TIME=month)
                    f = xr.broadcast(f, grad_long)[0]
                    alpha = (g / f) * grad_long
                    beta = (g / f) * grad_lat
                    back_x = self.sea_surf_grad['LONGITUDE'] + alpha * delta_t
                    back_y = self.sea_surf_grad['LATITUDE'] - beta * delta_t
                    
                    prev_implicit_k_tm_anom = prev_implicit_k_tm_anom_at_cur_loc.interp(
                        LONGITUDE=back_x, LATITUDE=back_y
                    ).combine_first(prev_implicit_k_tm_anom_at_cur_loc)
                else:
                    prev_implicit_k_tm_anom = implicit_model_anomalies[-1].isel(TIME=-1)

                cur_tsub_anom = self.t_sub.sel(TIME=month)
                cur_heat_flux_anom = self.heat_flux.sel(TIME=month)
                cur_ekman_anom = self.ekman.sel(TIME=month)
                cur_entrainment_vel = self.entrain_vel.sel(MONTH=month_in_year)
                cur_geo_anom = self.geo_anom.sel(TIME=month)
                cur_hbar = self.hbar.sel(MONTH=month_in_year)

                if self.params['INCLUDE_SURFACE'] and self.params['INCLUDE_EKMAN']: cur_surf_ek = cur_heat_flux_anom + cur_ekman_anom
                elif self.params['INCLUDE_SURFACE']: cur_surf_ek = cur_heat_flux_anom
                elif self.params['INCLUDE_EKMAN']: cur_surf_ek = cur_ekman_anom
                else: cur_surf_ek = cur_ekman_anom * 0

                if self.params['INCLUDE_GEO_ANOM']: cur_surf_ek = cur_surf_ek + cur_geo_anom

                if self.params['INCLUDE_ENTRAINMENT']:
                    cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar) + cur_entrainment_vel / cur_hbar * cur_tsub_anom
                    cur_a = cur_entrainment_vel / cur_hbar + gamma_0 / (rho_0 * c_0 * cur_hbar)
                else:
                    cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar)
                    cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)

                cur_implicit_k_tm_anom = (prev_implicit_k_tm_anom + delta_t * cur_b) / (1 + delta_t * cur_a)
                cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore').expand_dims(TIME=[month])
                implicit_model_anomalies.append(cur_implicit_k_tm_anom)

        self.model_anom = xr.concat(implicit_model_anomalies, 'TIME').rename("IMPLICIT_ANOMALY")
        
        # Deseasonalize result
        time_vals = self.model_anom.TIME.values
        months = ((time_vals + 0.5) % 12).astype(int); months[months == 0] = 12 
        self.model_anom = self.model_anom.assign_coords(month_idx=("TIME", months))
        monthly_mean = self.model_anom.groupby("month_idx").mean("TIME")
        self.model_anom = self.model_anom.groupby("month_idx") - monthly_mean
        self.model_anom = self.model_anom.drop_vars("month_idx")

        # --- SELECT OBSERVATION DATASET FOR COMPARISON ---
        if self.params['USE_OBS_REYNOLD']:
            print("Comparison: Reynolds Dataset")
            self.obs_anom = self.obs_reynolds.interp_like(self.model_anom, method='linear')
        else:
            print("Comparison: Argo Dataset")
            self.obs_anom = self.obs_argo

        # --- Statistics ---
        print("Calculating Statistics...")
        self.calc_correlations()     
        self.calc_significance()     
        self.calc_rmse_all()
        
        self.update_map_visuals(None)
        print("Done.")

    def calc_correlations(self):
        lags = np.arange(-12, 13)
        corrs = []
        for k in lags:
            model_shifted = self.model_anom.shift(TIME=k)
            r = xr.corr(self.obs_anom, model_shifted, dim="TIME")
            r.coords['lag'] = k
            corrs.append(r)
        self.da_corr = xr.concat(corrs, dim='lag')

        mask_obs = self.da_corr.notnull().all(dim="lag")
        abs_run = np.abs(self.da_corr)
        abs_filled = abs_run.where(np.isfinite(abs_run), -np.inf)
        best_idx = abs_filled.argmax(dim="lag") 
        
        self.da_corr_best_lag = self.da_corr["lag"].isel(lag=best_idx).where(mask_obs)
        self.da_corr_max_val = self.da_corr.isel(lag=best_idx).where(mask_obs)

    def calc_significance(self):
        lags = np.arange(-12, 13)
        sig_corr_list = []
        
        r_x = xr.corr(self.obs_anom, self.obs_anom.shift(TIME=1), dim="TIME")
        r_y = xr.corr(self.model_anom, self.model_anom.shift(TIME=1), dim="TIME")
        N_total = len(self.obs_anom.TIME)

        for k in lags:
            model_shifted = self.model_anom.shift(TIME=k)
            r = xr.corr(self.obs_anom, model_shifted, dim="TIME")
            r.coords['lag'] = k
            
            N_lagged = N_total - abs(k)
            autocorr_term = r_x * r_y
            N_effective = N_lagged * (1 - autocorr_term) / (1 + autocorr_term)
            t_stat = r * np.sqrt((N_effective - 2) / (1 - r**2))
            p_values = 2 * t.sf(np.abs(t_stat), df=(N_effective - 2))
            p_values_da = xr.DataArray(p_values, coords=r.coords, dims=r.dims)
            
            r_significant = r.where(p_values_da < 0.05)
            sig_corr_list.append(r_significant)

        self.da_corr_sig = xr.concat(sig_corr_list, dim='lag')
        
        abs_run = np.abs(self.da_corr_sig)
        abs_filled = abs_run.where(np.isfinite(abs_run), -np.inf)
        has_sig = np.isfinite(abs_run).any(dim="lag")
        
        best_idx = abs_filled.argmax(dim="lag")
        
        self.da_sig_best_lag = self.da_corr_sig["lag"].isel(lag=best_idx).where(has_sig)
        self.da_sig_max_val = self.da_corr_sig.isel(lag=best_idx).where(has_sig)

    def calc_rmse_all(self):
        error = (self.model_anom - self.obs_anom)
        rmse_val = np.sqrt((error ** 2).mean(dim="TIME"))
        rmse_obs_norm = np.sqrt((self.obs_anom ** 2).mean(dim="TIME"))
        
        self.da_rmse_norm = rmse_val / rmse_obs_norm
        self.da_rmse_abs = rmse_val

        def get_seasonal_subset(ds, lat_slice, month_offsets):
            indices = []
            for i in range(13): 
                target_times = [start + i*12 for start in month_offsets]
                indices.extend(target_times)
            ds_sub = ds.sel(TIME=indices, method="nearest")
            ds_sub = ds_sub.sel(LATITUDE=lat_slice)
            return ds_sub

        obs_n = get_seasonal_subset(self.obs_anom, slice(0, 79.5), [17.5, 18.5, 19.5])
        mod_n = get_seasonal_subset(self.model_anom, slice(0, 79.5), [17.5, 18.5, 19.5])
        obs_s = get_seasonal_subset(self.obs_anom, slice(-64.5, 0), [11.5, 12.5, 13.5])
        mod_s = get_seasonal_subset(self.model_anom, slice(-64.5, 0), [11.5, 12.5, 13.5])
        rmse_n = np.sqrt(((mod_n - obs_n)**2).mean(dim="TIME"))
        rmse_s = np.sqrt(((mod_s - obs_s)**2).mean(dim="TIME"))
        self.da_rmse_seasonal = xr.concat([rmse_s, rmse_n], dim="LATITUDE")

    def update_display_mode(self, label):
        self.update_map_visuals(None)

    def update_map_visuals(self, event):
        if self.da_corr is None: return

        main_mode = self.radio_main.value_selected
        sub_mode = self.radio_sub.value_selected
        
        data_to_plot = None
        vmin, vmax = -1, 1
        cmap = 'nipy_spectral'
        title = ""
        cbar_unit = ""

        if 'Correlation' in main_mode:
            self.ax_radio_sub.set_visible(True)
            self.radio_sub.active = self.radio_sub.active 

            use_sig = '(95% Sig)' in main_mode
            
            if sub_mode == 'Specific Lag':
                self.ax_lag.set_visible(True); self.s_lag.ax.set_visible(True)
                lag_val = int(self.s_lag.val)
                ds_source = self.da_corr_sig if use_sig else self.da_corr
                data_to_plot = ds_source.sel(lag=lag_val)
                title = f"{main_mode}\nLag: {lag_val}"
                cbar_unit = "Correlation Coefficient"

            elif sub_mode == 'Max Value':
                self.ax_lag.set_visible(False); self.s_lag.ax.set_visible(False)
                data_to_plot = self.da_sig_max_val if use_sig else self.da_corr_max_val
                title = f"{main_mode}\nMax Correlation Value"
                cbar_unit = "Correlation Coefficient"

            elif sub_mode == 'Lag of Max':
                self.ax_lag.set_visible(False); self.s_lag.ax.set_visible(False)
                data_to_plot = self.da_sig_best_lag if use_sig else self.da_corr_best_lag
                vmin, vmax = -12, 12
                title = f"{main_mode}\nLag (months) at Max Correlation"
                cbar_unit = "Lag (Months)"  
        else:
            self.ax_radio_sub.set_visible(False)
            self.ax_lag.set_visible(False); self.s_lag.ax.set_visible(False)
            
            if main_mode == 'RMSE (Norm)':
                data_to_plot = self.da_rmse_norm
                vmin, vmax = 0, 3
                title = "Normalized RMSE"
                cbar_unit = "Normalized Ratio"
            elif main_mode == 'RMSE (Abs)':
                data_to_plot = self.da_rmse_abs
                vmin, vmax = 0, 3.0
                title = "Absolute RMSE (°C)"
                cbar_unit = "RMSE [°C]"
            elif main_mode == 'RMSE (Seasonal)':
                data_to_plot = self.da_rmse_seasonal
                vmin, vmax = 0, 3.0
                title = "Seasonal RMSE (Summer)"
                cbar_unit = "RMSE [°C]"
        
        dataset_name = "Reynolds" if self.params['USE_OBS_REYNOLD'] else "Argo"
        title += f"\nGamma: {self.params['gamma']:.1f} | Obs: {dataset_name}"

        if self.mesh is None:
            self.mesh = data_to_plot.plot(
                ax=self.ax_map, cmap=cmap, vmin=vmin, vmax=vmax,
                add_colorbar=True, cbar_kwargs={'orientation': 'vertical'}
            )
        else:
            self.mesh.set_array(data_to_plot.values.ravel())
            self.mesh.set_clim(vmin, vmax)
            self.mesh.set_cmap(cmap)
        
        self.ax_map.set_title(title)
        if hasattr(self.mesh, 'colorbar') and self.mesh.colorbar:
            self.mesh.colorbar.set_label(cbar_unit)
        self.fig.canvas.draw_idle()

# --- Initialize and Run ---------------------------------------------------
dashboard = InteractiveSSTModel(
    obs_anom_argo=observed_temperature_anomaly_argo,
    obs_anom_reynolds=observed_temperature_anomaly_reynold,
    heat_flux=surface_flux_da,
    ekman=ekman_anomaly_da,
    entrain_vel=entrainment_vel_da,
    t_sub=t_sub_da,
    hbar=hbar_da,
    geo_anom=geostrophic_anomaly_da,
    sea_surf_grad=sea_surface_grad_ds,
    ssh_var=ssh_var_name
)

plt.show()