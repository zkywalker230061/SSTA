import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import xarray as xr
import numpy as np
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, coriolis_parameter
from scipy.stats import t

# --- FILE PATHS --------------------------------------------------------------
# Please adjust these paths if they differ on your local machine
observed_T_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
observed_T_path_Reynold = "/Users/julia/Desktop/SSTA/datasets/Reynold_sst_anomalies-(2004-2018).nc"

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Anomaly_Full_Datasets.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
ENTRAINMENT_VEL_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Entrainment_Velocity-(2004-2018).nc"

H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/hbar.nc"
NEW_H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_hbar.nc"

T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/t_sub.nc"
NEW_T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/New MLD & T_sub/new_T_sub_prime.nc"

GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated_2.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"

USE_DOWNLOADED_SSH = False
USE_NEW_H_BAR_NEW_T_SUB = True 

# --- DATA PREPARATION --------------------------------------------------------
print("Loading Data...")

# 1. Observed Data (Reynolds)
observed_temp_ds_reynold = xr.open_dataset(observed_T_path_Reynold, decode_times=False)
observed_temperature_anomaly_reynold = observed_temp_ds_reynold['anom']

# 2. Observed Data (Argo) + MLD + T_sub + Ekman
if USE_NEW_H_BAR_NEW_T_SUB:
    hbar_ds = xr.open_dataset(NEW_H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    t_sub_ds = xr.open_dataset(NEW_T_SUB_DATA_PATH, decode_times=False)
    t_sub_da = t_sub_ds["ANOMALY_SUB_TEMPERATURE"]

    observed_temp_ds_argo = xr.open_dataset(observed_T_path, decode_times=False)
    observed_temp_ds = observed_temp_ds_argo["UPDATED_MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_argo, "UPDATED_MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly_argo = observed_temperature_anomaly["UPDATED_MIXED_LAYER_TEMP_ANOMALY"]

    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds['UPDATED_TEMP_EKMAN_ANOM']
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)
else:
    hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
    hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
    t_sub_da = t_sub_ds["SUB_TEMPERATURE"]
    t_sub_mean = get_monthly_mean(t_sub_da)
    t_sub_anom = get_anomaly(t_sub_ds, "SUB_TEMPERATURE", t_sub_mean)
    t_sub_da = t_sub_anom["SUB_TEMPERATURE_ANOMALY"]

    observed_temp_ds_argo = xr.open_dataset(observed_T_path, decode_times=False)
    observed_temp_ds = observed_temp_ds_argo["MIXED_LAYER_TEMP"]
    obs_temp_mean = get_monthly_mean(observed_temp_ds)
    observed_temperature_anomaly = get_anomaly(observed_temp_ds_argo, "MIXED_LAYER_TEMP", obs_temp_mean)
    observed_temperature_anomaly_argo = observed_temperature_anomaly["MIXED_LAYER_TEMP_ANOMALY"]

    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    ekman_anomaly_da = ekman_anomaly_ds["TEMP_EKMAN_ANOM"]
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)

# 3. Heat Flux
heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

# 4. Entrainment Velocity
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

# 5. Geostrophic
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



print("Data Loaded.")

# --- INTERACTIVE CLASS -------------------------------------------------------
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
        
        # Pre-calculate Expanded Data
        # Forward fill to ensure no gaps in time expansion
        self.entrain_vel_expanded = xr.concat([self.entrain_vel] * 15, dim='MONTH').reset_coords(drop=True).rename({'MONTH': 'TIME'})
        self.entrain_vel_expanded['TIME'] = self.obs_argo.TIME

        self.hbar_expanded = xr.concat([self.hbar] * 15, dim='MONTH').reset_coords(drop=True).rename({'MONTH': 'TIME'})
        self.hbar_expanded['TIME'] = self.obs_argo.TIME
        
        # --- CALCULATE SEASONALLY VARYING DAMPING (Lag-1 Covariance) ---
        print("Pre-calculating Seasonally Varying Damping Map (Lag-1 Method)...")
        T_t = self.obs_argo
        Q_t = self.heat_flux
        T_t_minus_1 = self.obs_argo.shift(TIME=1)
        
        def calc_seasonal_cov(da1, da2):
            # Calculate integer month index
            t_vals = da1.TIME.values
            months = ((t_vals + 0.5) % 12).astype(int)
            months[months == 0] = 12
            
            # Assign temporary coord for grouping
            da1_c = da1.assign_coords(month_idx=("TIME", months))
            da2_c = da2.assign_coords(month_idx=("TIME", months))
            
            # Group by the new integer month index
            da1_g = da1_c.groupby('month_idx')
            da2_g = da2_c.groupby('month_idx')
            
            mean1 = da1_g.mean('TIME')
            mean2 = da2_g.mean('TIME')
            
            # Anomaly from monthly mean
            anom1 = da1_g - mean1
            anom2 = da2_g - mean2
            
            # Product & Mean (Covariance)
            prod = anom1 * anom2
            cov = prod.groupby('month_idx').mean('TIME')
            return cov

        cov_QT = calc_seasonal_cov(Q_t, T_t_minus_1)
        cov_TT = calc_seasonal_cov(T_t, T_t_minus_1)
        
        # 3. Calculate Lambda = - Cov(T, Q) / Cov(T,T)
        cov_TT_safe = cov_TT.where(np.abs(cov_TT) > 1e-5)
        self.gamma_map_seasonal = -1 * (cov_QT / cov_TT_safe)
        self.gamma_map_seasonal = self.gamma_map_seasonal.rename({'month_idx': 'MONTH'})
        # self.gamma_map_seasonal = self.gamma_map_seasonal.fillna(15.0)
        # self.gamma_map_seasonal = self.gamma_map_seasonal.where(np.isfinite(self.gamma_map_seasonal), 15.0)
        self.gamma_map_seasonal = self.gamma_map_seasonal.where(self.gamma_map_seasonal > 0, 0) #damping must be positive
        #print("Damping Map Ready (Holes filled with 15.0).")




        # Parameters Dictionary
        self.params = {
            'INCLUDE_SURFACE': True, 
            'INCLUDE_EKMAN': True, 
            'INCLUDE_ENTRAINMENT': True,
            'INCLUDE_GEO_ANOM': False, 
            'INCLUDE_GEO_MEAN': False, 
            
            'USE_OBS_REYNOLD': False,
            'USE_INITIAL_ANOM': False,
            'USE_STOCHASTIC_SCHEME': False, 
            'USE_VARYING_DAMPING': False,
            
            'gamma': 15, 
        }
        
        self.model_anom = None
        self.da_corr = None
        self.da_rmse_norm = None
        self.mesh = None
        
        # --- Figure Layout ---
        self.fig = plt.figure(figsize=(16, 10)) 
        self.ax_map = self.fig.add_axes([0.15, 0.40, 0.75, 0.55]) 
        self.ax_check = self.fig.add_axes([0.05, 0.05, 0.10, 0.25])
        self.ax_radio_model = self.fig.add_axes([0.16, 0.20, 0.12, 0.10])
        self.ax_radio_init = self.fig.add_axes([0.16, 0.05, 0.12, 0.10])
        self.ax_radio_data = self.fig.add_axes([0.29, 0.20, 0.12, 0.10])
        self.ax_radio_damp = self.fig.add_axes([0.29, 0.05, 0.12, 0.10])
        self.ax_radio_view = self.fig.add_axes([0.43, 0.05, 0.12, 0.25])
        self.ax_gamma = self.fig.add_axes([0.60, 0.22, 0.25, 0.03])
        self.ax_lag   = self.fig.add_axes([0.60, 0.15, 0.25, 0.03])
        self.ax_run   = self.fig.add_axes([0.80, 0.05, 0.10, 0.05])
        
        self.create_widgets()
        self.run_simulation(None)

    def create_widgets(self):
        self.ax_check.set_title("Physics", loc='left', fontweight='bold', fontsize=10)
        labels = ['INCLUDE_SURFACE', 'INCLUDE_EKMAN', 'INCLUDE_ENTRAINMENT', 'INCLUDE_GEO_ANOM', 'INCLUDE_GEO_MEAN']
        actives = [self.params[k] for k in labels]
        self.check = CheckButtons(self.ax_check, labels, actives)
        for lbl in self.check.labels: lbl.set_fontsize(8)

        self.ax_radio_model.set_title("Model Scheme", fontsize=9, fontweight='bold')
        self.radio_model = RadioButtons(self.ax_radio_model, ('Implicit (Slab)', 'Analytical (Stoch)'), active=0)
        self.radio_model.on_clicked(self.update_params_ui)

        self.ax_radio_init.set_title("Initial Cond.", fontsize=9, fontweight='bold')
        self.radio_init = RadioButtons(self.ax_radio_init, ('Zero', 'First Anomaly'), active=0)
        self.radio_init.on_clicked(self.update_params_ui)

        self.ax_radio_data.set_title("Comp. Data", fontsize=9, fontweight='bold')
        self.radio_data = RadioButtons(self.ax_radio_data, ('Argo', 'Reynolds'), active=0)
        self.radio_data.on_clicked(self.update_params_ui)

        self.ax_radio_damp.set_title("Damping", fontsize=9, fontweight='bold')
        self.radio_damp = RadioButtons(self.ax_radio_damp, ('Constant', 'Varying (Map)'), active=0)
        self.radio_damp.on_clicked(self.update_params_ui)

        self.ax_radio_view.set_title("View Mode", fontsize=9, fontweight='bold')
        self.radio_view = RadioButtons(self.ax_radio_view, ('Correlation', 'RMSE (Norm)', 'RMSE (Abs)', 'Raw Model (Last)'), active=0)
        self.radio_view.on_clicked(self.update_display_mode)

        self.s_gamma = Slider(self.ax_gamma, 'Const Gamma', 0, 100, valinit=self.params['gamma'], valstep=1)
        self.s_lag = Slider(self.ax_lag, 'Lag (Mo)', -12, 12, valinit=0, valstep=1)
        self.b_run = Button(self.ax_run, 'RUN', color='lightblue', hovercolor='0.975')
        
        self.b_run.on_clicked(self.run_simulation)
        self.s_lag.on_changed(self.update_map_visuals)

    def update_params_ui(self, label):
        self.params['USE_STOCHASTIC_SCHEME'] = (self.radio_model.value_selected == 'Analytical (Stoch)')
        self.params['USE_INITIAL_ANOM'] = (self.radio_init.value_selected == 'First Anomaly')
        self.params['USE_OBS_REYNOLD'] = (self.radio_data.value_selected == 'Reynolds')
        self.params['USE_VARYING_DAMPING'] = (self.radio_damp.value_selected == 'Varying (Map)')

    def run_simulation(self, event):
        print("--- Starting Simulation ---")
        
        labels = ['INCLUDE_SURFACE', 'INCLUDE_EKMAN', 'INCLUDE_ENTRAINMENT', 'INCLUDE_GEO_ANOM', 'INCLUDE_GEO_MEAN']
        status = self.check.get_status()
        for i, k in enumerate(labels):
            self.params[k] = status[i]
            
        self.update_params_ui(None)
        
        if self.params['USE_VARYING_DAMPING']:
            print("Using Seasonally Varying Damping Map")
            gamma_input = self.gamma_map_seasonal
        else:
            print(f"Using Constant Gamma: {self.s_gamma.val}")
            gamma_input = self.s_gamma.val

        if self.params['USE_STOCHASTIC_SCHEME']:
            self.run_stochastic_model(gamma_input)
        else:
            self.run_slab_model(gamma_input)

        # Post-Processing
        time_vals = self.model_anom.TIME.values
        months = ((time_vals + 0.5) % 12).astype(int); months[months == 0] = 12 
        self.model_anom = self.model_anom.assign_coords(month_idx=("TIME", months))
        monthly_mean = self.model_anom.groupby("month_idx").mean("TIME")
        self.model_anom = self.model_anom.groupby("month_idx") - monthly_mean
        self.model_anom = self.model_anom.drop_vars("month_idx")

        if self.params['USE_OBS_REYNOLD']:
            print("Comparison vs Reynolds")
            self.obs_anom = self.obs_reynolds
        else:
            print("Comparison vs Argo")
            self.obs_anom = self.obs_argo

        print("Calculating Stats...")
        self.calc_correlations()
        self.calc_rmse_all()
        
        self.update_map_visuals(None)
        print("--- Done ---")

    def run_slab_model(self, gamma_input):
        print("Running: Implicit Slab Model")
        rho_0 = 1025.0
        c_0 = 4100.0
        g = 9.81
        delta_t = 30.4375 * 24 * 3600
        
        surf_flux = self.heat_flux.fillna(0)
        ekman = self.ekman.fillna(0)
        geo = self.geo_anom.fillna(0)
        
        implicit_model_anomalies = []
        added_baseline = False

        for month in self.heat_flux.TIME.values:
            month_in_year = int((month + 0.5) % 12)
            if month_in_year == 0: month_in_year = 12
            
            # Select Gamma
            if isinstance(gamma_input, xr.DataArray) and 'MONTH' in gamma_input.dims:
                cur_gamma = gamma_input.sel(MONTH=month_in_year)
            else:
                cur_gamma = gamma_input

            if not added_baseline:
                if self.params['USE_INITIAL_ANOM']:
                    base = self.obs_argo.sel(TIME=month, method='nearest')
                else:
                    base = surf_flux.sel(TIME=month) * 0 
                
                base = base.expand_dims(TIME=[month])
                implicit_model_anomalies.append(base)
                added_baseline = True
            else:
                prev_anom = implicit_model_anomalies[-1].isel(TIME=-1)
                
                if self.params['INCLUDE_GEO_MEAN']:
                    # Note: We don't interpolate_na sea_surf_grad here to keep it simple, 
                    # but if you have gaps there, they will propagate.
                    f = coriolis_parameter(self.sea_surf_grad['LATITUDE'])
                    grad_long = self.sea_surf_grad[self.ssh_var + '_anomaly_grad_long'].sel(TIME=month)
                    grad_lat = self.sea_surf_grad[self.ssh_var + '_anomaly_grad_lat'].sel(TIME=month)
                    f = xr.broadcast(f, grad_long)[0]
                    f = f.where(np.abs(f) > 1e-10, 1e-10) 
                    
                    alpha = (g / f) * grad_long
                    beta = (g / f) * grad_lat
                    back_x = self.sea_surf_grad['LONGITUDE'] + alpha * delta_t
                    back_y = self.sea_surf_grad['LATITUDE'] - beta * delta_t
                    
                    # fill_value="extrapolate" isn't supported directly in xarray interp, 
                    # so we fillna afterwards to ensure we don't lose edges
                    prev_anom = prev_anom.interp(LONGITUDE=back_x, LATITUDE=back_y).combine_first(prev_anom)

                cur_tsub_anom = self.t_sub.sel(TIME=month)
                cur_heat_flux_anom = surf_flux.sel(TIME=month)
                cur_ekman_anom = ekman.sel(TIME=month)
                cur_entrainment_vel = self.entrain_vel.sel(MONTH=month_in_year)
                cur_geo_anom = geo.sel(TIME=month)
                cur_hbar = self.hbar.sel(MONTH=month_in_year) # Safety Depth

                if self.params['INCLUDE_SURFACE'] and self.params['INCLUDE_EKMAN']: cur_surf_ek = cur_heat_flux_anom + cur_ekman_anom
                elif self.params['INCLUDE_SURFACE']: cur_surf_ek = cur_heat_flux_anom
                elif self.params['INCLUDE_EKMAN']: cur_surf_ek = cur_ekman_anom
                else: cur_surf_ek = cur_ekman_anom * 0

                if self.params['INCLUDE_GEO_ANOM']: cur_surf_ek = cur_surf_ek + cur_geo_anom

                if self.params['INCLUDE_ENTRAINMENT']:
                    cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar) + cur_entrainment_vel / cur_hbar * cur_tsub_anom
                    cur_a = cur_entrainment_vel / cur_hbar + cur_gamma / (rho_0 * c_0 * cur_hbar)
                else:
                    cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar)
                    cur_a = cur_gamma / (rho_0 * c_0 * cur_hbar)

                cur_anom = (prev_anom + delta_t * cur_b) / (1 + delta_t * cur_a)

                
                cur_anom = cur_anom.drop_vars('MONTH', errors='ignore').expand_dims(TIME=[month])
                implicit_model_anomalies.append(cur_anom)

        self.model_anom = xr.concat(implicit_model_anomalies, 'TIME').rename("IMPLICIT_ANOMALY")

    def run_stochastic_model(self, gamma_input):
        print("Running: Analytical (Stochastic) Model")
        rho_0 = 1025.0
        c_0 = 4100.0
        seconds_month = 30.4375 * 24 * 60 * 60

        # Aggressive Fill on inputs
        surf_flux = self.heat_flux.fillna(0) if self.params['INCLUDE_SURFACE'] else (self.heat_flux * 0)
        ekman = self.ekman.fillna(0) if self.params['INCLUDE_EKMAN'] else (self.ekman * 0)
        geo = self.geo_anom.fillna(0) if self.params['INCLUDE_GEO_ANOM'] else (self.geo_anom * 0)
        
        hbar = self.hbar_expanded#.fillna(20) # Safety depth
        w_e = self.entrain_vel_expanded
        if not self.params['INCLUDE_ENTRAINMENT']:
            w_e = w_e * 0
        
        if isinstance(gamma_input, xr.DataArray) and 'MONTH' in gamma_input.dims:
            months = self.obs_argo.TIME.values
            month_indices = ((months + 0.5) % 12).astype(int); month_indices[month_indices == 0] = 12
            gamma_series = gamma_input.sel(MONTH=xr.DataArray(month_indices, coords={'TIME': self.obs_argo.TIME})).fillna(15)
        else:
            gamma_series = gamma_input 
            
        forcing_flux = (surf_flux + ekman + geo) / (rho_0 * c_0 * hbar)
        _lambda = gamma_series / (rho_0 * c_0 * hbar) + w_e / hbar
        
        simulated_list = []
        temp = None
        times = self.obs_argo.TIME.values
        
        for i, month_num in enumerate(times):
            if i == 0:
                if self.params['USE_INITIAL_ANOM']:
                    current_da = self.obs_argo.sel(TIME=month_num)
                else:
                    current_da = self.obs_argo.sel(TIME=month_num) * 0
                temp = current_da
            else:
                prev_time = times[i-1]
                
                h_curr = hbar.sel(TIME=month_num)
                h_prev = hbar.sel(TIME=prev_time)
                
                # Check for div by zero or negative logs
                h_ratio = h_curr / h_prev
                h_ratio = h_ratio.where(h_ratio > 0, 1) # Prevent log(neg)
                
                log_h_change = np.log(h_ratio) / seconds_month
                entrain_rate_masked = log_h_change.where(log_h_change > 0, 0)
                
                if self.params['INCLUDE_ENTRAINMENT']:
                    entrain_forcing = self.t_sub.sel(TIME=prev_time) * entrain_rate_masked
                else:
                    entrain_forcing = 0
                
                total_forcing = forcing_flux.sel(TIME=prev_time) + entrain_forcing
                lambda_prev = _lambda.sel(TIME=prev_time)
                lambda_prev = lambda_prev.where(lambda_prev > 1e-10, 1e-10)
                decay_factor = np.exp(-lambda_prev * seconds_month)
                
                current_da = (
                    temp * decay_factor + 
                    (total_forcing / lambda_prev) * (1 - decay_factor)
                )
                temp = current_da
            
            current_da = current_da.drop_vars('MONTH', errors='ignore')
            current_da = current_da.expand_dims(TIME=[month_num])
            simulated_list.append(current_da)
            
        self.model_anom = xr.concat(simulated_list, dim='TIME', coords='minimal').rename("STOCHASTIC_ANOMALY")

    def calc_correlations(self):
        lags = np.arange(-12, 13)
        corrs = []
        for k in lags:
            model_shifted = self.model_anom.shift(TIME=k)
            r = xr.corr(self.obs_anom, model_shifted, dim="TIME")
            r.coords['lag'] = k
            corrs.append(r)
        self.da_corr = xr.concat(corrs, dim='lag')

    def calc_rmse_all(self):
        error = (self.model_anom - self.obs_anom)
        rmse_val = np.sqrt((error ** 2).mean(dim="TIME"))
        rmse_obs_norm = np.sqrt((self.obs_anom ** 2).mean(dim="TIME"))
        
        self.da_rmse_norm = rmse_val / rmse_obs_norm
        self.da_rmse_abs = rmse_val

    def update_display_mode(self, label):
        self.update_map_visuals(None)

    def update_map_visuals(self, event):
        if self.da_corr is None: return

        mode = self.radio_view.value_selected
        
        data_to_plot = None
        vmin, vmax = -1, 1
        cmap = 'nipy_spectral'
        title = mode
        cbar_unit = ""

        if mode == 'Correlation':
            lag_val = int(self.s_lag.val)
            data_to_plot = self.da_corr.sel(lag=lag_val)
            title = f"Correlation (Lag {lag_val})"
            vmin, vmax = -1, 1
            cbar_unit = "r"
        elif mode == 'RMSE (Norm)':
            data_to_plot = self.da_rmse_norm
            vmin, vmax = 0, 2
            cbar_unit = "Ratio"
        elif mode == 'RMSE (Abs)':
            data_to_plot = self.da_rmse_abs
            vmin, vmax = 0, 2
            cbar_unit = "°C"
        elif mode == 'Raw Model (Last)':
            # Show the last time step of the model to prove it exists
            data_to_plot = self.model_anom.isel(TIME=-1)
            vmin, vmax = -2, 2
            cmap = 'RdBu_r'
            title = "Raw Model Anomaly (Final Month)"
            cbar_unit = "°C"
        
        phys_str = []
        if self.params['INCLUDE_ENTRAINMENT']: phys_str.append("Ent")
        if self.params['INCLUDE_EKMAN']: phys_str.append("Ek")
        if self.params['USE_VARYING_DAMPING']: phys_str.append("VarDamp")
        
        full_title = f"{title} | {self.radio_model.value_selected} | {','.join(phys_str)}"

        if self.mesh is None:
            self.mesh = data_to_plot.plot(
                ax=self.ax_map, cmap=cmap, vmin=vmin, vmax=vmax,
                add_colorbar=True, cbar_kwargs={'orientation': 'vertical'}
            )
        else:
            self.mesh.set_array(data_to_plot.values.ravel())
            self.mesh.set_clim(vmin, vmax)
            self.mesh.set_cmap(cmap)
        
        self.ax_map.set_title(full_title)
        if hasattr(self.mesh, 'colorbar') and self.mesh.colorbar:
            self.mesh.colorbar.set_label(cbar_unit)
        self.fig.canvas.draw_idle()

# --- RUN APPLICATION ---------------------------------------------------------
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