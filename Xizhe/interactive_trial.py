import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import xarray as xr
import numpy as np
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, coriolis_parameter


observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
# HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
ENTRAINMENT_VEL_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"
# H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/t_sub.nc"
GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"
USE_DOWNLOADED_SSH = False


temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
observed_temp_ds = xr.open_dataset(observed_path, decode_times=False)
observed_temperature_monthly_average = get_monthly_mean(observed_temp_ds['__xarray_dataarray_variable__'])
observed_temperature_anomaly = get_anomaly(observed_temp_ds, '__xarray_dataarray_variable__', observed_temperature_monthly_average)
observed_temperature_anomaly = observed_temperature_anomaly['__xarray_dataarray_variable___ANOMALY']

heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + \
                                heat_flux_ds['avg_ishf']
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

surface_flux_da_centred_mean = get_monthly_mean(heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'])
surface_flux_da_anomaly = get_anomaly(heat_flux_anomaly_ds, 'NET_HEAT_FLUX_ANOMALY', surface_flux_da_centred_mean)
surface_flux_da_final = heat_flux_anomaly_ds["NET_HEAT_FLUX_ANOMALY_ANOMALY"]

# print(surface_flux_da)
# print(surface_flux_da_final)
# print(f"Original Net Heat Flux Mean: {heat_flux_ds['NET_HEAT_FLUX'].mean().values}")
# print(f"Double-Centered Mean: {surface_flux_da_final.mean().values}")

ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
ekman_anomaly_da = ekman_anomaly_ds['Q_Ek_anom']
ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da), 0)
ekman_anomaly_da_centred_mean = get_monthly_mean(ekman_anomaly_ds["Q_Ek_anom"])
ekman_anomaly_da_final = get_anomaly(ekman_anomaly_ds, "Q_Ek_anom", ekman_anomaly_da_centred_mean)
ekman_anomaly_da_final = ekman_anomaly_da_final["Q_Ek_anom_ANOMALY"]


# Overwrite off-centred anomalies to avoid changing variables in the simulation
surface_flux_da = surface_flux_da_final
ekman_anomaly_da = ekman_anomaly_da_final
ekman_anomaly_da = ekman_anomaly_da.fillna(0)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD_PRESSURE"]

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
t_sub_da = t_sub_ds["T_sub_ANOMALY"]

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
    def __init__(self, obs_anom, heat_flux, ekman, entrain_vel, t_sub, hbar, geo_anom, sea_surf_grad, ssh_var):
        # Store Datasets
        self.obs_anom = obs_anom
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
            'INCLUDE_GEOSTROPHIC_MEAN', 
            'INCLUDE_GEOSTROPHIC_ANOM'
        ]

        self.params = {
            'INCLUDE_SURFACE': True,
            'INCLUDE_EKMAN': True,
            'INCLUDE_ENTRAINMENT': False,
            'INCLUDE_GEOSTROPHIC_MEAN': False,
            'INCLUDE_GEOSTROPHIC_ANOM': False,
            'gamma': 15
        }
        
        # Storage for calculated data
        self.model_anom = None
        self.da_corr = None
        self.da_rmse_norm = None
        self.da_rmse_abs = None
        self.da_rmse_seasonal = None
        
        self.mesh = None
        
        # --- Figure Setup ---
        self.fig = plt.figure(figsize=(14, 9))
        
        # Layout positions [left, bottom, width, height]
        self.ax_map     = self.fig.add_axes([0.1, 0.35, 0.8, 0.6]) 
        self.ax_check   = self.fig.add_axes([0.05, 0.05, 0.15, 0.2])
        self.ax_radio   = self.fig.add_axes([0.22, 0.05, 0.15, 0.2]) # New Radio Button Axis
        self.ax_gamma   = self.fig.add_axes([0.45, 0.05, 0.4, 0.03])
        self.ax_lag     = self.fig.add_axes([0.45, 0.15, 0.4, 0.03])
        self.ax_run     = self.fig.add_axes([0.88, 0.05, 0.08, 0.05])
        
        self.create_widgets()
        self.run_simulation(None)

    def create_widgets(self):
        # 1. Physics Toggles (CheckButtons)
        labels = list(self.params.keys())[:-1] 
        actives = [self.params[k] for k in labels]
        self.check = CheckButtons(self.ax_check, labels, actives)
        
        # 2. Display Mode (RadioButtons) - NEW
        # This lets us switch between Correlation and different RMSE modes
        self.display_options = ('Correlation', 'RMSE (Norm)', 'RMSE (Abs)', 'RMSE (Seasonal)')
        self.radio = RadioButtons(self.ax_radio, self.display_options, active=0)
        self.radio.on_clicked(self.update_display_mode)

        # 3. Gamma Slider
        self.s_gamma = Slider(self.ax_gamma, 'Gamma', 0, 100, valinit=self.params['gamma'], valstep=1)
        
        # 4. Lag Slider (Only for Correlation)
        self.s_lag = Slider(self.ax_lag, 'Lag (Months)', -12, 12, valinit=0, valstep=1)
        
        # 5. Run Button
        self.b_run = Button(self.ax_run, 'Recalculate', color='lightblue', hovercolor='0.975')
        
        # Connect events
        self.b_run.on_clicked(self.run_simulation)
        self.s_lag.on_changed(self.update_map_visuals) 

    def run_simulation(self, event):
        """Runs the physics model, then calculates all stats (Corr, RMSEs)."""
        print("Running Simulation...")
        
        # --- A. Update Params ---
        status = self.check.get_status()
        for k, s in zip(self.keys, status):
            self.params[k] = s
        self.params['gamma'] = self.s_gamma.val
        
        # --- B. Physics Constants ---
        rho_0 = 1025.0
        c_0 = 4100.0
        g = 9.81
        gamma_0 = self.params['gamma']
        
        def month_to_second(month):
            return month * 30.4375 * 24 * 60 * 60
        delta_t = month_to_second(1)

        # --- C. Time Loop (Implicit Model) ---
        implicit_model_anomalies = []
        added_baseline = False

        for month in self.heat_flux.TIME.values:
            month_in_year = int((month + 0.5) % 12)
            if month_in_year == 0: month_in_year = 12
            
            if not added_baseline:
                base = self.obs_anom.sel(TIME=month) * 0 
                base = base.expand_dims(TIME=[month])
                implicit_model_anomalies.append(base)
                added_baseline = True
            else:
                # 1. Fetch Previous State
                prev_implicit_k_tm_anom_at_cur_loc = implicit_model_anomalies[-1].isel(TIME=-1)
                
                # 2. Geostrophic Advection (Back-propagation)
                if self.params['INCLUDE_GEOSTROPHIC_ANOM']:
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

                # 3. Fetch Forcing Data
                cur_tsub_anom = self.t_sub.sel(TIME=month)
                cur_heat_flux_anom = self.heat_flux.sel(TIME=month)
                cur_ekman_anom = self.ekman.sel(TIME=month)
                cur_entrainment_vel = self.entrain_vel.sel(MONTH=month_in_year)
                cur_geo_anom = self.geo_anom.sel(TIME=month)
                cur_hbar = self.hbar.sel(MONTH=month_in_year)

                # 4. Construct Terms
                if self.params['INCLUDE_SURFACE'] and self.params['INCLUDE_EKMAN']:
                    cur_surf_ek = cur_heat_flux_anom + cur_ekman_anom
                elif self.params['INCLUDE_SURFACE']:
                    cur_surf_ek = cur_heat_flux_anom
                elif self.params['INCLUDE_EKMAN']:
                    cur_surf_ek = cur_ekman_anom
                else:
                    cur_surf_ek = cur_ekman_anom * 0

                if self.params['INCLUDE_GEOSTROPHIC_MEAN']:
                    cur_surf_ek = cur_surf_ek + cur_geo_anom

                if self.params['INCLUDE_ENTRAINMENT']:
                    cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar) + cur_entrainment_vel / cur_hbar * cur_tsub_anom
                    cur_a = cur_entrainment_vel / cur_hbar + gamma_0 / (rho_0 * c_0 * cur_hbar)
                else:
                    cur_b = cur_surf_ek / (rho_0 * c_0 * cur_hbar)
                    cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)

                # 5. Solve Step
                cur_implicit_k_tm_anom = (prev_implicit_k_tm_anom + delta_t * cur_b) / (1 + delta_t * cur_a)
                cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.drop_vars('MONTH', errors='ignore')
                cur_implicit_k_tm_anom = cur_implicit_k_tm_anom.expand_dims(TIME=[month])
                implicit_model_anomalies.append(cur_implicit_k_tm_anom)

        # --- D. Post-Processing ---
        self.model_anom = xr.concat(implicit_model_anomalies, 'TIME')
        self.model_anom = self.model_anom.rename("IMPLICIT_ANOMALY")
        
        # Remove Residual Seasonality
        time_vals = self.model_anom.TIME.values
        months = ((time_vals + 0.5) % 12).astype(int)
        months[months == 0] = 12 
        self.model_anom = self.model_anom.assign_coords(month_idx=("TIME", months))
        monthly_mean = self.model_anom.groupby("month_idx").mean("TIME")
        self.model_anom = self.model_anom.groupby("month_idx") - monthly_mean
        self.model_anom = self.model_anom.drop_vars("month_idx")

        # --- E. Calculate Statistics (All at once) ---
        print("Calculating Statistics...")
        self.calc_correlations()
        self.calc_rmse_all()
        
        # --- F. Update Plot ---
        self.update_map_visuals(None)
        print("Done.")

    def calc_correlations(self):
        """Calculates lag correlations."""
        lags = np.arange(-12, 13)
        corrs = []
        for k in lags:
            model_shifted = self.model_anom.shift(TIME=k)
            r = xr.corr(self.obs_anom, model_shifted, dim="TIME")
            r.coords['lag'] = k
            corrs.append(r)
        self.da_corr = xr.concat(corrs, dim='lag')

    def calc_rmse_all(self):
        """Calculates Norm RMSE, Abs RMSE, and Seasonal RMSE."""
        # 1. Common Error
        error = (self.model_anom - self.obs_anom)
        mse = (error ** 2).mean(dim="TIME")
        rmse_val = np.sqrt(mse)
        
        # 2. Observation Norm
        obs_mse = (self.obs_anom ** 2).mean(dim="TIME")
        rmse_obs_norm = np.sqrt(obs_mse)
        
        # --- A. Normalized RMSE ---
        self.da_rmse_norm = rmse_val / rmse_obs_norm
        
        # --- B. Absolute RMSE ---
        self.da_rmse_abs = rmse_val

        # --- C. Seasonal RMSE (Summer) ---
        # Helper to get subset
        def get_seasonal_subset(ds, lat_slice, month_offsets):
            # Generate indices based on your logic: 17.5 + i*12
            # Assuming TIME is monotonic and aligns with these indices roughly
            indices = []
            for i in range(13): # 13 years roughly
                # Your logic: (17.5 + i*12, 18.5 + i*12, 19.5 + i*12)
                # This suggests selecting specific time points.
                # We use 'nearest' on the existing TIME array.
                target_times = [start + i*12 for start in month_offsets]
                indices.extend(target_times)
            
            # Select Time
            ds_sub = ds.sel(TIME=indices, method="nearest")
            # Select Lat
            ds_sub = ds_sub.sel(LATITUDE=lat_slice)
            return ds_sub

        # North (Months ~17.5 -> June/July/Aug logic)
        obs_north = get_seasonal_subset(self.obs_anom, slice(0, 79.5), [17.5, 18.5, 19.5])
        mod_north = get_seasonal_subset(self.model_anom, slice(0, 79.5), [17.5, 18.5, 19.5])
        
        # South (Months ~11.5 -> Dec/Jan/Feb logic)
        obs_south = get_seasonal_subset(self.obs_anom, slice(-64.5, 0), [11.5, 12.5, 13.5])
        mod_south = get_seasonal_subset(self.model_anom, slice(-64.5, 0), [11.5, 12.5, 13.5])

        # Calculate RMSE for North
        err_n = (mod_north - obs_north)
        rmse_n = np.sqrt((err_n**2).mean(dim="TIME"))
        
        # Calculate RMSE for South
        err_s = (mod_south - obs_south)
        rmse_s = np.sqrt((err_s**2).mean(dim="TIME"))

        # Combine
        self.da_rmse_seasonal = xr.concat([rmse_s, rmse_n], dim="LATITUDE")

    def update_display_mode(self, label):
        """Called when Radio Button is clicked."""
        self.update_map_visuals(None)

    def update_map_visuals(self, event):
        """Updates the map based on the current mode and slider values."""
        if self.da_corr is None: return

        mode = self.radio.value_selected
        
        # Determine what data to plot and visual settings
        if mode == 'Correlation':
            # Enable Lag Slider
            self.ax_lag.set_visible(True)
            self.s_lag.ax.set_visible(True)
            
            lag_val = int(self.s_lag.val)
            data_to_plot = self.da_corr.sel(lag=lag_val)
            
            vmin, vmax = -1, 1
            cmap = 'nipy_spectral'
            title = f"Correlation (Lag {lag_val})\nGamma: {self.params['gamma']:.1f}"

        else:
            # Disable Lag Slider for RMSE modes
            self.ax_lag.set_visible(False)
            self.s_lag.ax.set_visible(False)
            
            if mode == 'RMSE (Norm)':
                data_to_plot = self.da_rmse_norm
                vmin, vmax = 0, 3 # Normalized usually around 1
                cmap = 'nipy_spectral'
                title = f"Normalized RMSE\nGamma: {self.params['gamma']:.1f}"
            
            elif mode == 'RMSE (Abs)':
                data_to_plot = self.da_rmse_abs
                vmin, vmax = 0, 3.0 # Absolute degrees C
                cmap = 'nipy_spectral'
                title = f"Absolute RMSE (°C)\nGamma: {self.params['gamma']:.1f}"
                
            elif mode == 'RMSE (Seasonal)':
                data_to_plot = self.da_rmse_seasonal
                vmin, vmax = 0, 3.0
                cmap = 'nipy_spectral'
                title = f"Seasonal RMSE (Summer N/S)\nGamma: {self.params['gamma']:.1f}"

        # Draw the plot
        if self.mesh is None:
            # First time initialization
            self.mesh = data_to_plot.plot(
                ax=self.ax_map,
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                add_colorbar=True,
                cbar_kwargs={'orientation': 'vertical'}
            )
        else:
            # Update data
            self.mesh.set_array(data_to_plot.values.ravel())
            self.mesh.set_clim(vmin, vmax)
            self.mesh.set_cmap(cmap)
        
        self.ax_map.set_title(title)
        self.fig.canvas.draw_idle()

# --- Initialize and Run ---------------------------------------------------
dashboard = InteractiveSSTModel(
    obs_anom=observed_temperature_anomaly,
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