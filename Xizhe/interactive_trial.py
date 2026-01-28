import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import xarray as xr
import numpy as np
from chris_utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset


observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/heat_flux_interpolated_all_contributions.nc"
# HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
ENTRAINMENT_VEL_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Entrainment_Velocity-(2004-2018).nc"
# ENTRAINMENT_VEL_DENOISED_DATA_PATH = "../datasets/entrainment_vel_denoised.nc"
H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
H_BAR_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/data_for_modelling/t_sub.nc"
GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_downloaded.nc"
GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/geostrophic_anomaly_calculated.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/sea_surface_interpolated_grad.nc"
USE_DOWNLOADED_SSH = False

rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 30
g = 9.81
f = 1 

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
# print(f"Original Mean Ekman: {ekman_anomaly_ds['Q_Ek_anom'].mean().values}")
# print(f"Centered Mean Ekman: {ekman_anomaly_da_final.mean().values}")

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
# print(entrainment_vel_da)

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
        # ... (keep all your existing dataset storage lines) ...
        self.obs_anom = obs_anom
        self.heat_flux = heat_flux
        self.ekman = ekman
        self.entrain_vel = entrain_vel
        self.t_sub = t_sub
        self.hbar = hbar
        self.geo_anom = geo_anom
        self.sea_surf_grad = sea_surf_grad
        self.ssh_var = ssh_var
        
        # ... (keep your params dictionary) ...
        self.params = {
            'INCLUDE_SURFACE': True,
            'INCLUDE_EKMAN': True,
            'INCLUDE_ENTRAINMENT': True,
            'INCLUDE_GEOSTROPHIC': False,
            'INCLUDE_GEO_DISP': False,
            'gamma': 30
        }
        
        self.model_anom = None
        self.correlations = None
        
        # --- ADD THIS LINE ---
        self.mesh = None  # Placeholder to store the plot object
        # ---------------------

        # ... (keep figure and widget setup lines) ...
        self.fig = plt.figure(figsize=(12, 8))
        self.ax_map = self.fig.add_axes([0.1, 0.35, 0.8, 0.6]) 
        
        self.ax_check = self.fig.add_axes([0.05, 0.05, 0.15, 0.2])
        self.ax_gamma = self.fig.add_axes([0.3, 0.05, 0.4, 0.03])
        self.ax_lag   = self.fig.add_axes([0.3, 0.15, 0.4, 0.03])
        self.ax_run   = self.fig.add_axes([0.8, 0.05, 0.1, 0.05])
        
        self.create_widgets()
        self.run_simulation(None)

    def create_widgets(self):
        # 1. Checkboxes
        labels = list(self.params.keys())[:-1] # All except gamma
        actives = [self.params[k] for k in labels]
        self.check = CheckButtons(self.ax_check, labels, actives)
        
        # 2. Gamma Slider
        self.s_gamma = Slider(self.ax_gamma, 'Gamma', 0, 100, valinit=self.params['gamma'])
        
        # 3. Lag Slider (For viewing the "Movie" manually)
        self.s_lag = Slider(self.ax_lag, 'Lag (Months)', -12, 12, valinit=0, valstep=1)
        
        # 4. Run Button
        self.b_run = Button(self.ax_run, 'Recalculate\nModel', color='lightblue', hovercolor='0.975')
        
        # Connect events
        self.b_run.on_clicked(self.run_simulation)
        self.s_lag.on_changed(self.update_map_only)

    def run_simulation(self, event):
        """Runs the physics loop based on current widget values."""
        print("Recalculating Physics... please wait.")
        
        # Update params from widgets
        status = self.check.get_status()
        self.params['INCLUDE_SURFACE'] = status[0]
        self.params['INCLUDE_EKMAN'] = status[1]
        self.params['INCLUDE_ENTRAINMENT'] = status[2]
        self.params['INCLUDE_GEOSTROPHIC'] = status[3]
        self.params['INCLUDE_GEO_DISP'] = status[4]
        self.params['gamma'] = self.s_gamma.val
        
        # --- THE PHYSICS LOOP ---
        implicit_anomalies = []
        added_baseline = False
        
        # Physics Constants
        rho_0 = 1025.0
        c_0 = 4100.0
        g = 9.81
        gamma_val = self.params['gamma']
        
        def month_to_second(m): return m * 30.4375 * 24 * 60 * 60
        delta_t = month_to_second(1)

        # Time Loop
        for month in self.heat_flux.TIME.values:
            prev_month = month - 1
            month_in_year = int((month + 0.5) % 12)
            if month_in_year == 0: month_in_year = 12
            
            # (Note: prev_month_in_year logic is not needed for the current loop step 
            # unless used for interpolation, simplified here for speed)

            if not added_baseline:
                # Initialize with zeros
                base = self.obs_anom.sel(TIME=month) * 0
                base = base.expand_dims(TIME=[month])
                implicit_anomalies.append(base)
                added_baseline = True
            else:
                prev_tm = implicit_anomalies[-1].isel(TIME=-1)
                
                # Load Monthly Data
                cur_heat = self.heat_flux.sel(TIME=month) if self.params['INCLUDE_SURFACE'] else 0
                
                cur_ek = self.ekman.sel(TIME=month) if self.params['INCLUDE_EKMAN'] else 0
                
                cur_geo = self.geo_anom.sel(TIME=month) if self.params['INCLUDE_GEOSTROPHIC'] else 0

                cur_surf_total = cur_heat + cur_ek + cur_geo
                
                # Entrainment logic
                cur_ent_vel = self.entrain_vel.sel(MONTH=month_in_year)
                cur_hb = self.hbar.sel(MONTH=month_in_year)
                cur_tsub = self.t_sub.sel(TIME=month)
                
                if self.params['INCLUDE_ENTRAINMENT']:
                    cur_b = cur_surf_total / (rho_0 * c_0 * cur_hb) + cur_ent_vel / cur_hb * cur_tsub
                    cur_a = cur_ent_vel / cur_hb + gamma_val / (rho_0 * c_0 * cur_hb)
                else:
                    cur_b = cur_surf_total / (rho_0 * c_0 * cur_hb)
                    cur_a = gamma_val / (rho_0 * c_0 * cur_hb)

                # Implicit Update Step
                cur_tm = (prev_tm + delta_t * cur_b) / (1 + delta_t * cur_a)
                
                cur_tm = cur_tm.drop_vars('MONTH', errors='ignore').expand_dims(TIME=[month])
                implicit_anomalies.append(cur_tm)

        # Concatenate results
        self.model_anom = xr.concat(implicit_anomalies, 'TIME')
        self.model_anom = self.model_anom.rename("IMPLICIT_ANOMALY")
        
        # --- FIXED: Remove Residual Seasonality (Manual grouping) ---
        # 1. Calculate month indices manually to match your loop logic
        time_vals = self.model_anom.TIME.values
        months = ((time_vals + 0.5) % 12).astype(int)
        months[months == 0] = 12 # Fix the 0 case to be 12
        
        # 2. Assign as a coordinate for grouping
        self.model_anom = self.model_anom.assign_coords(month_idx=("TIME", months))
        
        # 3. Group by this manual index
        monthly_mean = self.model_anom.groupby("month_idx").mean("TIME")
        self.model_anom = self.model_anom.groupby("month_idx") - monthly_mean
        
        # 4. Cleanup
        self.model_anom = self.model_anom.drop_vars("month_idx")

        print("Physics Done. Calculating Correlations...")
        self.calculate_correlations()
        
    def calculate_correlations(self):
        """Calculates lag correlations for the slider."""
        lags = np.arange(-12, 13)
        
        # Efficient correlation calculation
        corrs = []
        for k in lags:
            # Shift model
            model_shifted = self.model_anom.shift(TIME=k)
            # Correlation
            r = xr.corr(self.obs_anom, model_shifted, dim="TIME")
            r.coords['lag'] = k
            corrs.append(r)
            
        self.correlations = xr.concat(corrs, dim='lag')
        print("Done.")
        self.update_map_only(None)

    def update_map_only(self, val):
        """Updates the map pixels without redrawing the axes/colorbar."""
        if self.correlations is None: return
        
        lag_val = int(self.s_lag.val)
        data_to_plot = self.correlations.sel(lag=lag_val)
        
        # If this is the first time running, create the plot and colorbar
        if self.mesh is None:
            self.mesh = data_to_plot.plot(
                ax=self.ax_map,
                cmap='nipy_spectral',
                vmin=-1, vmax=1,
                add_colorbar=True, # Add colorbar only ONCE
                cbar_kwargs={'label': 'Correlation Coeff'}
            )
        else:
            # If plot exists, just update the pixel data!
            # .ravel() flattens the 2D array to 1D, which set_array expects
            self.mesh.set_array(data_to_plot.values.ravel())
        
        # Update the title
        self.ax_map.set_title(f"Correlation at Lag {lag_val} months\nGamma: {self.params['gamma']:.1f}")
        
        # Efficiently redraw only the canvas
        self.fig.canvas.draw_idle()

# --- 2. Initialize and Run ---------------------------------------------------
# Make sure your data variables (surface_flux_da, ekman_anomaly_da, etc) 
# are loaded from your previous cells before running this.

# Note: passing 'surface_flux_da' etc. assumes they are defined in your global scope
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