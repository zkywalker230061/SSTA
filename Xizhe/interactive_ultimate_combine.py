import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import xarray as xr
import numpy as np
import time
from scipy.stats import t
from sklearn.decomposition import PCA

# --- EXTERNAL IMPORTS ---
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, coriolis_parameter

# ==============================================================================
# 1. CONFIGURATION & DATA LOADING
# ==============================================================================

observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
observed_path_Reynold = "/Users/julia/Desktop/SSTA/datasets/Reynold_sst_anomalies-(2004-2018).nc"

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Anomaly_Full_Datasets.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"

ENTRAINMENT_VEL_DATA_PATH = "datasets/New_Entrainment/Entrainment_Vel_h.nc"
NEW_ENTRAINMENT_VEL_DATA_PATH = "datasets/New_Entrainment/Entrainment_Vel_New_h.nc"

H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/hbar.nc"
NEW_H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_hbar.nc"

T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/t_sub.nc"
NEW_T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_T_sub_prime.nc"
T_SUB_GRADMETHOD_DATA_PATH="datasets/New_Entrainment/Tsub_Max_Gradient_Method_h.nc"
NEW_T_SUB_GRADMETHOD_DATA_PATH = "datasets/New_Entrainment/Tsub_Max_Gradient_Method_New_h.nc"

GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated_2.nc"

USE_DOWNLOADED_SSH = False
USE_NEW_H_BAR_NEW_T_SUB = True

print("--- Loading Datasets ---")

if USE_NEW_H_BAR_NEW_T_SUB:
    hbar_ds = xr.open_dataset(NEW_H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    t_sub_grad_ds = xr.open_dataset(NEW_T_SUB_GRADMETHOD_DATA_PATH, decode_times=False)
    t_sub_grad_mean = get_monthly_mean(t_sub_grad_ds["SUB_TEMPERATURE"])
    t_sub_grad_anom_struct = get_anomaly(t_sub_grad_ds, "SUB_TEMPERATURE", t_sub_grad_mean)
    t_sub_grad_da = t_sub_grad_anom_struct["SUB_TEMPERATURE_ANOMALY"]
    
    t_sub_anom_ds = xr.open_dataset(NEW_T_SUB_DATA_PATH, decode_times=False)
    t_sub_anom_da = t_sub_anom_ds["ANOMALY_SUB_TEMPERATURE"]

    observed_temp_ds_argo = xr.open_dataset(observed_path, decode_times=False)
    observed_temp_ds = observed_temp_ds_argo["UPDATED_MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_argo, "UPDATED_MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly_argo = observed_temperature_anomaly["UPDATED_MIXED_LAYER_TEMP_ANOMALY"]

    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds['UPDATED_TEMP_EKMAN_ANOM']
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

    entrainment_vel_ds = xr.open_dataset(NEW_ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
    entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
    entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

else:
    hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    t_sub_grad_ds = xr.open_dataset(T_SUB_GRADMETHOD_DATA_PATH, decode_times=False)
    t_sub_grad_mean = get_monthly_mean(t_sub_grad_ds["SUB_TEMPERATURE"])
    t_sub_grad_anom_struct = get_anomaly(t_sub_grad_ds, "SUB_TEMPERATURE", t_sub_grad_mean)
    t_sub_grad_da = t_sub_grad_anom_struct["SUB_TEMPERATURE_ANOMALY"]
    
    t_sub_anom_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
    t_sub_anom_da = t_sub_anom_ds["ANOMALY_SUB_TEMPERATURE"]

    observed_temp_ds_argo = xr.open_dataset(observed_path, decode_times=False)
    observed_temp_ds = observed_temp_ds_argo["MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_argo, "MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly_argo = observed_temperature_anomaly["MIXED_LAYER_TEMP_ANOMALY"]

    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds["TEMP_EKMAN_ANOM"]
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

    entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
    entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
    entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

observed_temp_ds_reynold = xr.open_dataset(observed_path_Reynold, decode_times=False)['anom']
observed_temperature_anomaly_reynold = observed_temp_ds_reynold

heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

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

print("--- Data Loading Complete ---")


# ==============================================================================
# 2. DASHBOARD CLASS
# ==============================================================================

class ProfessionalSSTDashboard:
    def __init__(self, obs_anom_argo, obs_anom_reynolds, heat_flux, ekman, entrain_vel, t_sub_grad, t_sub_anom, hbar, geo_anom, sea_surf_grad, ssh_var):
        # --- Data Storage ---
        self.obs_argo = obs_anom_argo
        self.obs_reynolds = obs_anom_reynolds
        self.obs_current = self.obs_argo
        
        self.heat_flux = heat_flux
        self.ekman = ekman
        self.entrain_vel = entrain_vel
        self.t_sub_grad = t_sub_grad
        self.t_sub_anom = t_sub_anom
        self.hbar = hbar
        self.geo_anom = geo_anom
        self.sea_surf_grad = sea_surf_grad
        self.ssh_var = ssh_var

        # --- State ---
        self.model_anom = None
        self.last_clicked_coords = None # (lon, lat)
        self.mesh = None 
        self.ax_ts_twin = None
        self.fig_taylor = None
        self.fig_pca = None
        self.fig_eof = None
        
        self.params = {
            'INCLUDE_SURFACE': True, 
            'INCLUDE_EKMAN': True, 
            'INCLUDE_ENTRAINMENT': True,
            'INCLUDE_GEO_ANOM': False, 
            'INCLUDE_GEO_MEAN': False, 
            'USE_OBS_REYNOLD': False,
            'USE_INITIAL_ANOM': False,
            'USE_TSUB_GRAD': True,
            'gamma': 15,
        }
        
        # --- GUI Setup ---
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Mixed Layer SST Anomaly Model", fontsize=16, fontweight='bold', y=0.98)
        
        # Grid Layout
        gs = gridspec.GridSpec(3, 4, height_ratios=[0.5, 10, 6], width_ratios=[1, 4, 4, 0.2])
        
        self.ax_status = self.fig.add_subplot(gs[0, :])
        self.ax_status.axis('off')

        self.ax_controls = self.fig.add_subplot(gs[1:, 0])
        self.ax_controls.axis('off')
        
        self.ax_map = self.fig.add_subplot(gs[1, 1:3])
        self.ax_cbar = self.fig.add_subplot(gs[1, 3])
        
        self.ax_ts = self.fig.add_subplot(gs[2, 1:3])
        
        self.setup_controls()
        self.run_simulation(None)

    def setup_controls(self):
        def add_header(text, y):
            self.ax_controls.text(0.01, y, text, transform=self.ax_controls.transAxes, fontsize=11, fontweight='bold', color='#333333')
        
        # 1. Physics Toggles
        add_header("Physics Components", 1)
        self.check_labels = ['INCLUDE_SURFACE', 'INCLUDE_EKMAN', 'INCLUDE_ENTRAINMENT', 'INCLUDE_GEO_ANOM', 'INCLUDE_GEO_MEAN']
        self.ax_chk_phys = self.fig.add_axes([0.02, 0.70, 0.12, 0.18])
        self.ax_chk_phys.set_frame_on(False)
        self.check_phys = CheckButtons(self.ax_chk_phys, 
                                       [l.replace('INCLUDE_', '').replace('_', ' ').title() for l in self.check_labels], 
                                       [self.params[l] for l in self.check_labels])
        self.check_phys.on_clicked(self.update_params)
        
        # 2. Dataset/Method Toggles
        add_header("Data & Methods", 0.8)
        self.check_labels_data = ['USE_OBS_REYNOLD', 'USE_INITIAL_ANOM', 'USE_TSUB_GRAD']
        display_labels_data = ["Reynolds Obs", "Init Anom", "T_sub (Grad)"]
        self.ax_chk_data = self.fig.add_axes([0.02, 0.53, 0.12, 0.12])
        self.ax_chk_data.set_frame_on(False)
        self.check_data = CheckButtons(self.ax_chk_data, display_labels_data, 
                                       [self.params[l] for l in self.check_labels_data])
        self.check_data.on_clicked(self.update_params)

        # 3. Parameters (Sliders)
        self.ax_gamma = self.fig.add_axes([0.05, 0.44, 0.13, 0.03])
        self.s_gamma = Slider(self.ax_gamma, 'Gamma', 0, 100, valinit=self.params['gamma'], valstep=1, color='#5c92f7')
        
        self.ax_lag = self.fig.add_axes([0.05, 0.39, 0.13, 0.03])
        self.s_lag = Slider(self.ax_lag, 'Lag', -12, 12, valinit=0, valstep=1, color='#5c92f7')
        self.s_lag.on_changed(self.update_visuals_only)

        # 4. View Mode (Radio)
        add_header("Analysis View", 0.33)
        self.view_modes = ('Correlation', 'Correlation (Sig)', 'Lag of Max Corr', 'RMSE (Norm)', 'RMSE (Abs)')
        self.ax_radio = self.fig.add_axes([0.02, 0.16, 0.15, 0.14])
        self.ax_radio.set_frame_on(False)
        self.radio = RadioButtons(self.ax_radio, self.view_modes, active=0)
        self.radio.on_clicked(self.update_visuals_only)

        # 5. Advanced Analysis Buttons
        self.ax_btn_eof = self.fig.add_axes([0.04, 0.10, 0.06, 0.04])
        self.b_eof = Button(self.ax_btn_eof, 'EOF/PCA', color='#ffc107', hovercolor='#ffca2c')
        self.b_eof.label.set_fontsize(9)
        self.b_eof.on_clicked(self.run_eof_analysis)

        self.ax_btn_taylor = self.fig.add_axes([0.10, 0.10, 0.06, 0.04])
        self.b_taylor = Button(self.ax_btn_taylor, 'Taylor Diag', color='#17a2b8', hovercolor='#138496')
        self.b_taylor.label.set_fontsize(9)
        self.b_taylor.on_clicked(self.run_taylor_analysis)

        # 6. Action Button
        self.ax_run = self.fig.add_axes([0.04, 0.04, 0.15, 0.05])
        self.b_run = Button(self.ax_run, 'Run Simulation', color='#28a745', hovercolor='#218838')
        self.b_run.label.set_color('white')
        self.b_run.label.set_fontweight('bold')
        self.b_run.on_clicked(self.run_simulation)

        # Connect Map Click
        self.fig.canvas.mpl_connect('button_press_event', self.on_map_click)

    def update_params(self, label):
        for i, key in enumerate(self.check_labels):
            self.params[key] = self.check_phys.get_status()[i]
        
        for i, key in enumerate(self.check_labels_data):
            self.params[key] = self.check_data.get_status()[i]
            
        self.params['gamma'] = self.s_gamma.val
        
    def log_status(self, text, time_taken=None):
        self.ax_status.clear()
        self.ax_status.axis('off')
        msg = f"STATUS: {text}"
        if time_taken:
            msg += f" | Time: {time_taken:.3f}s"
        
        bg_color = '#e6fffa' if 'Complete' in text else '#fff3cd'
        self.ax_status.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12, 
                            bbox=dict(facecolor=bg_color, edgecolor='#dddddd', boxstyle='round,pad=0.5'))
        self.fig.canvas.draw_idle()
        plt.pause(0.001) 

    def run_simulation(self, event):
        t0 = time.time()
        self.log_status("Running Physics Model... Please wait.", 0)
        self.params['gamma'] = self.s_gamma.val

        # Constants
        rho_0, c_0, g = 1025.0, 4100.0, 9.81
        gamma_0 = self.params['gamma']
        delta_t = 30.4375 * 24 * 3600 
        
        active_t_sub = self.t_sub_grad if self.params['USE_TSUB_GRAD'] else self.t_sub_anom
        self.obs_current = self.obs_reynolds if self.params['USE_OBS_REYNOLD'] else self.obs_argo

        implicit_model_anomalies = []
        added_baseline = False
        time_coords = self.heat_flux.TIME.values
        
        # --- PHYSICS LOOP ---
        for i, month in enumerate(time_coords):
            month_in_year = int((month + 0.5) % 12)
            if month_in_year == 0: month_in_year = 12

            if not added_baseline:
                if self.params['USE_INITIAL_ANOM']:
                    base = self.obs_argo.isel(TIME=i).fillna(0)
                else:
                    base = self.heat_flux.isel(TIME=i) * 0
                
                base = base.expand_dims(TIME=[month])
                implicit_model_anomalies.append(base)
                added_baseline = True
            else:
                prev_implicit_k_tm_anom_at_cur_loc = implicit_model_anomalies[-1].isel(TIME=-1)
                
                if self.params['INCLUDE_GEO_MEAN']:
                    f = coriolis_parameter(self.sea_surf_grad['LATITUDE'])
                    grad_long = self.sea_surf_grad[self.ssh_var + '_anomaly_grad_long'].sel(TIME=month, method='nearest')
                    grad_lat = self.sea_surf_grad[self.ssh_var + '_anomaly_grad_lat'].sel(TIME=month, method='nearest')
                    f = xr.broadcast(f, grad_long)[0]
                    alpha = (g / f) * grad_long
                    beta = (g / f) * grad_lat
                    back_x = self.sea_surf_grad['LONGITUDE'] + alpha * delta_t
                    back_y = self.sea_surf_grad['LATITUDE'] - beta * delta_t
                    prev_implicit_k_tm_anom = prev_implicit_k_tm_anom_at_cur_loc.interp(
                        LONGITUDE=back_x, LATITUDE=back_y
                    ).combine_first(prev_implicit_k_tm_anom_at_cur_loc)
                else:
                    prev_implicit_k_tm_anom = prev_implicit_k_tm_anom_at_cur_loc

                cur_tsub_anom = active_t_sub.sel(TIME=month, method='nearest')
                cur_heat_flux_anom = self.heat_flux.isel(TIME=i)
                cur_ekman_anom = self.ekman.sel(TIME=month, method='nearest')
                cur_entrainment_vel = self.entrain_vel.sel(MONTH=month_in_year)
                cur_geo_anom = self.geo_anom.sel(TIME=month, method='nearest')
                cur_hbar = self.hbar.sel(MONTH=month_in_year)

                cur_surf_ek = cur_ekman_anom * 0 
                if self.params['INCLUDE_SURFACE']: cur_surf_ek = cur_surf_ek + cur_heat_flux_anom
                if self.params['INCLUDE_EKMAN']: cur_surf_ek = cur_surf_ek + cur_ekman_anom
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
        
        # Robust Deseasonalization
        time_vals = self.model_anom.TIME.values
        month_indices = ((time_vals + 0.5) % 12).astype(int)
        month_indices[month_indices == 0] = 12
        
        self.model_anom = self.model_anom.assign_coords(month_idx=("TIME", month_indices))
        monthly_mean = self.model_anom.groupby("month_idx").mean("TIME")
        self.model_anom = self.model_anom.groupby("month_idx") - monthly_mean
        self.model_anom = self.model_anom.drop_vars("month_idx")

        if self.params['USE_OBS_REYNOLD']:
            self.obs_current = self.obs_reynolds.interp_like(self.model_anom, method='linear')
        else:
            self.obs_current = self.obs_argo 

        t_sim = time.time() - t0
        self.log_status(f"Physics Done ({t_sim:.2f}s). Calculating Stats...", t_sim)
        self.calc_statistics()
        
    def calc_statistics(self):
        t0 = time.time()
        
        # 1. RMSE
        error = (self.model_anom - self.obs_current)
        self.rmse_abs = np.sqrt((error ** 2).mean(dim="TIME"))
        obs_rms = np.sqrt((self.obs_current ** 2).mean(dim="TIME"))
        self.rmse_norm = self.rmse_abs / obs_rms
        
        # 2. Correlation
        lags = np.arange(-12, 13)
        corrs = []
        for k in lags:
            model_shifted = self.model_anom.shift(TIME=k)
            r = xr.corr(self.obs_current, model_shifted, dim="TIME")
            r.coords['lag'] = k
            corrs.append(r)
        self.da_corr = xr.concat(corrs, dim='lag')
        
        # 3. Significance & Max Lag
        n = len(self.model_anom.TIME)
        t_stat = self.da_corr * np.sqrt((n-2)/(1-self.da_corr**2))
        p_val = 2 * t.sf(np.abs(t_stat), n-2)
        self.da_corr_sig = self.da_corr.where(p_val < 0.05)
        
        # Identify Lag of Maximum Correlation
        self.lag_of_max = self.da_corr.idxmax(dim="lag")

        t_stat_total = time.time() - t0
        self.log_status("Simulation & Stats Complete", t_stat_total)
        self.update_visuals_only(None)

    def update_visuals_only(self, event):
        mode = self.radio.value_selected
        data = None
        vmin, vmax = 0, 1
        cmap = 'RdBu_r'
        title = ""
        cbar_label = ""
        
        if 'Correlation' in mode:
            lag = int(self.s_lag.val)
            if '(Sig)' in mode:
                data = self.da_corr_sig.sel(lag=lag)
            else:
                data = self.da_corr.sel(lag=lag)
            vmin, vmax = -1, 1
            cmap = 'nipy_spectral'
            title = f"Correlation (Lag {lag})"
            cbar_label = "Correlation Coeff"
        
        elif 'Lag of Max Corr' in mode:
            data = self.lag_of_max
            vmin, vmax = -6, 6
            cmap = 'PuOr'
            title = "Lag (Months) of Maximum Correlation\n(Blue = Model Leads, Orange = Model Lags)"
            cbar_label = "Lag [Months]"

        elif 'RMSE (Norm)' in mode:
            data = self.rmse_norm
            vmin, vmax = 0, 3
            cmap = 'nipy_spectral'
            title = "Normalized RMSE"
            cbar_label = "RMSE Ratio"
        elif 'RMSE (Abs)' in mode:
            data = self.rmse_abs
            vmin, vmax = 0, 2.5
            cmap = 'nipy_spectral'
            title = "Absolute RMSE (°C)"
            cbar_label = "RMSE [°C]"
            
        self.ax_map.clear()
        self.mesh = data.plot(ax=self.ax_map, vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=False)
        self.ax_map.set_title(title)
        
        self.ax_cbar.clear()
        plt.colorbar(self.mesh, cax=self.ax_cbar, label=cbar_label)
        
        if self.last_clicked_coords:
            self.plot_timeseries(self.last_clicked_coords[0], self.last_clicked_coords[1])
        else:
            self.ax_ts.clear()
            self.ax_ts.text(0.5, 0.5, "Click on map to Inspect Physics Budget", ha='center', transform=self.ax_ts.transAxes)
        
        self.fig.canvas.draw_idle()

    def on_map_click(self, event):
        if event.inaxes != self.ax_map: return
        if event.xdata is None or event.ydata is None: return
        self.last_clicked_coords = (event.xdata, event.ydata)
        self.plot_timeseries(event.xdata, event.ydata)

    def plot_timeseries(self, lon, lat):
        # 1. Clear previous axes explicitly
        self.ax_ts.clear()
        if self.ax_ts_twin is not None:
            self.ax_ts_twin.remove()
        
        self.ax_ts_twin = self.ax_ts.twinx()

        try:
            # --- A. DATA RETRIEVAL ---
            model_ts = self.model_anom.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            obs_ts = self.obs_current.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            found_lat = float(model_ts.LATITUDE)
            found_lon = float(model_ts.LONGITUDE)

            loc_hf = self.heat_flux.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            loc_ek = self.ekman.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            loc_geo = self.geo_anom.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            
            t_sub_source = self.t_sub_grad if self.params['USE_TSUB_GRAD'] else self.t_sub_anom
            loc_tsub = t_sub_source.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            
            months = ((loc_hf.TIME.values + 0.5) % 12).astype(int)
            months[months==0] = 12
            month_da = xr.DataArray(months, dims="TIME", coords={"TIME": loc_hf.TIME})
            
            loc_hbar = self.hbar.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest').sel(MONTH=month_da)
            loc_we = self.entrain_vel.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest').sel(MONTH=month_da)
            
            # --- B. CALCULATE BUDGET TERMS ---
            rho_0, c_0 = 1025.0, 4100.0
            dt_month = 30.4 * 24 * 3600
            rho_c_h = rho_0 * c_0 * loc_hbar
            
            term_qnet = (loc_hf / rho_c_h) * dt_month
            term_ekman = (loc_ek / rho_c_h) * dt_month
            term_geo = (loc_geo / rho_c_h) * dt_month
            term_ent = ((loc_we / loc_hbar) * loc_tsub) * dt_month
            
            # --- C. PLOTTING ---
            l1, = self.ax_ts_twin.plot(loc_hf.TIME, term_qnet, color='orange', alpha=0.3, linewidth=1, label='Qnet')
            l2, = self.ax_ts_twin.plot(loc_hf.TIME, term_ekman, color='green', alpha=0.3, linewidth=1, label='Ekman')
            l3, = self.ax_ts_twin.plot(loc_hf.TIME, term_ent, color='purple', alpha=0.3, linewidth=1, label='Entrain')
            l4, = self.ax_ts_twin.plot(loc_hf.TIME, term_geo, color='brown', alpha=0.3, linewidth=1, linestyle=':', label='Geo')

            self.ax_ts_twin.set_ylabel("Forcing (°C/month)", color='gray', fontsize=9)
            self.ax_ts_twin.tick_params(axis='y', labelcolor='gray', labelsize=8)
            self.ax_ts_twin.grid(False) 

            l5, = self.ax_ts.plot(model_ts.TIME, model_ts, label='Model SST', color='tab:blue', linewidth=2.5)
            l6, = self.ax_ts.plot(obs_ts.TIME, obs_ts, label='Obs SST', color='black', linestyle='--', alpha=0.8, linewidth=2)
            
            # --- D. ALIGN ZERO LINES ---
            y1_min, y1_max = self.ax_ts.get_ylim()
            range1 = max(abs(y1_min), abs(y1_max))
            self.ax_ts.set_ylim(-range1, range1) 
            
            y2_min, y2_max = self.ax_ts_twin.get_ylim()
            range2 = max(abs(y2_min), abs(y2_max))
            self.ax_ts_twin.set_ylim(-range2, range2)
            
            self.ax_ts.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

            r_val = float(xr.corr(model_ts, obs_ts))
            rmse_val = float(np.sqrt(((model_ts - obs_ts)**2).mean()))
            self.ax_ts.set_title(f"Loc: {found_lat:.1f}N, {found_lon:.1f}E | R={r_val:.2f} | RMSE={rmse_val:.2f}", fontsize=11, fontweight='bold')
            
            lines = [l5, l6, l1, l2, l3, l4]
            labels = [l.get_label() for l in lines]
            self.ax_ts.legend(lines, labels, loc='upper left', fontsize=8, ncol=6, frameon=True)
            self.ax_ts.set_ylabel("SST Anomaly (°C)", fontsize=10)
            self.ax_ts.grid(True, alpha=0.3)
            
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error plotting time series: {e}")
            import traceback
            traceback.print_exc()

    # --- ADVANCED ANALYSIS METHODS ---

    def run_taylor_analysis(self, event):
        """Pop up a Taylor Diagram. If a point is selected, show Local skill. Else, Global."""
        
        # --- A. Determine Scope (Local vs Global) ---
        if self.last_clicked_coords:
            # LOCAL MODE
            lon, lat = self.last_clicked_coords
            print(f"Generating Local Taylor Diagram for {lat:.1f}N, {lon:.1f}E...")
            
            # Select single time series
            obs_data = self.obs_current.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            mod_data = self.model_anom.sel(LONGITUDE=lon, LATITUDE=lat, method='nearest')
            title_prefix = f"Local Skill\n({lat:.1f}N, {lon:.1f}E)"
        else:
            # GLOBAL MODE
            print("Generating Global Taylor Diagram...")
            obs_data = self.obs_current
            mod_data = self.model_anom
            title_prefix = "Global Skill (Domain Average)"

        # --- B. Data Prep ---
        # Flatten (works for both 1D time series and 3D global arrays)
        obs_flat = obs_data.values.flatten()
        mod_flat = mod_data.values.flatten()
        
        mask = ~np.isnan(obs_flat) & ~np.isnan(mod_flat)
        obs_valid = obs_flat[mask]
        mod_valid = mod_flat[mask]
        
        if len(obs_valid) < 2:
            print("Not enough valid data for Taylor Diagram.")
            return

        # --- C. Calculate Statistics ---
        std_obs = np.std(obs_valid)
        std_mod = np.std(mod_valid)
        
        if std_obs == 0: 
            print("Observation Std Dev is 0. Cannot normalize.")
            return
            
        norm_std_mod = std_mod / std_obs
        corr = np.corrcoef(obs_valid, mod_valid)[0, 1]
        
        # --- D. Handle Figure Refresh ---
        if self.fig_taylor is not None and plt.fignum_exists(self.fig_taylor.number):
            self.fig_taylor.clf()
        else:
            self.fig_taylor = plt.figure(figsize=(7, 7), num="Taylor Diagram")

        ax = self.fig_taylor.add_subplot(111, polar=True)

        # --- E. Plotting ---
        # 1. Dynamic Radius Limit
        r_max = 1.5 * max(1.0, norm_std_mod)
        ax.set_ylim(0, r_max)
        
        # 2. Draw Correlation Lines
        t_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        for t_val in t_vals:
            theta = np.arccos(t_val)
            ax.plot([theta, theta], [0, r_max], color='gray', linestyle=':', lw=0.6)
            ax.text(theta, r_max*1.02, str(t_val), ha='center', fontsize=8)

        # 3. Draw Reference Arc (Obs)
        x = np.linspace(0, np.pi/2, 100)
        ax.plot(x, [1.0]*100, color='black', linestyle='--', lw=1.2, label='Reference (Obs)')
        
        # 4. Plot Model Point
        model_theta = np.arccos(np.clip(corr, 0, 1))
        
        # Choose color based on Global vs Local
        color = 'red' if self.last_clicked_coords else 'blue'
        marker = 'o' if self.last_clicked_coords else 's'
        
        ax.plot(model_theta, norm_std_mod, color=color, marker=marker, markersize=12, 
                label=f'Model (R={corr:.2f})', zorder=10)
        
        # 5. Add Observation Point
        ax.plot(0, 1, 'k*', markersize=12, zorder=10)

        # Aesthetics
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_ylabel('Normalized Standard Deviation', labelpad=20)
        ax.set_title(f'{title_prefix}\n(Gamma={self.params["gamma"]})', y=1.08, fontweight='bold')
        ax.grid(False) 
        ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize='small')
        
        self.fig_taylor.canvas.draw()
        self.fig_taylor.show()

    def run_eof_analysis(self, event):
        """Compute EOFs using PCA (scikit-learn) and plot."""
        self.log_status("Computing EOF/PCA... (May take a moment)")
        
        if self.fig_eof is not None and plt.fignum_exists(self.fig_eof.number):
            self.fig_eof.clf()
        else:
            self.fig_eof = plt.figure(figsize=(12, 8), num="EOF Analysis")

        da_stacked = self.model_anom.stack(space=("LATITUDE", "LONGITUDE"))
        da_stacked = da_stacked.dropna(dim="space", how="any")
        
        if da_stacked.shape[1] == 0:
            self.log_status("EOF Error: No valid data points.")
            return

        n_modes = 2
        pca = PCA(n_components=n_modes)
        pca.fit(da_stacked.values)
        
        components = pca.components_ 
        pcs = pca.transform(da_stacked.values) 
        
        explained_var = pca.explained_variance_ratio_ * 100
        
        coords = da_stacked.coords["space"]
        
        gs_eof = gridspec.GridSpec(2, 2)
        self.fig_eof.suptitle("Empirical Orthogonal Functions (EOF) - Top 2 Modes", fontsize=14, fontweight='bold')
        
        for i in range(n_modes):
            ax_map = self.fig_eof.add_subplot(gs_eof[0, i])
            eof_map = xr.DataArray(components[i, :], coords={"space": coords}, dims="space").unstack("space")
            eof_map.plot(ax=ax_map, cmap='RdBu_r', add_colorbar=True, cbar_kwargs={'label': 'Arbitrary Units'})
            ax_map.set_title(f"EOF Mode {i+1} ({explained_var[i]:.1f}%)")
            
            ax_ts = self.fig_eof.add_subplot(gs_eof[1, i])
            ax_ts.plot(self.model_anom.TIME, pcs[:, i], color='black' if i==0 else 'blue')
            ax_ts.set_title(f"PC Time Series {i+1}")
            ax_ts.grid(True, alpha=0.3)
            
        self.log_status("EOF Analysis Complete.")
        self.fig_eof.canvas.draw()
        self.fig_eof.show()

# ==============================================================================
# 4. LAUNCH
# ==============================================================================
dashboard = ProfessionalSSTDashboard(
    obs_anom_argo=observed_temperature_anomaly_argo,
    obs_anom_reynolds=observed_temperature_anomaly_reynold,
    heat_flux=surface_flux_da,
    ekman=ekman_anomaly_da,
    entrain_vel=entrainment_vel_da,
    t_sub_grad=t_sub_grad_da,
    t_sub_anom=t_sub_anom_da,
    hbar=hbar_da,
    geo_anom=geostrophic_anomaly_da,
    sea_surf_grad=sea_surface_grad_ds,
    ssh_var=ssh_var_name
)

plt.show()