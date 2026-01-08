#%%
import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset, load_pressure_data, get_monthly_mean_from_year, tile_monthly_to_daily
from matplotlib.animation import FuncAnimation
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)

# --- File Paths (assuming these are correct) ------------------------------------------------
MLD_TEMP_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
MLD_DEPTH_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
HEAT_FLUX_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/ERA5-ARGO_Mean_Surface_Heat_Flux_Daily_2004.nc" 
TURBULENT_SURFACE_STRESS = '/Users/julia/Desktop/SSTA/datasets/datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress_Daily_2004.nc'
EK_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman Current Anomaly - Daily 2004 - Test"
CHRIS_SCHEME_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/model_anomaly_exponential_damping_implicit.nc"

# --- Model Constants -----------------------------------------------------------------------
YEAR = 2004
RHO_O = 1025.0  # kg/m^3
C_O = 4100.0  # J/(kg K)
SECONDS_DAY = 24 * 60 * 60  # s
GAMMA = 10  # bulk damping factor

# --- Load and Prepare Data (assuming helper functions are correct) -------------------------
mld_temperature_ds = xr.open_dataset(MLD_TEMP_PATH, decode_times=False)
mld_depth_ds = load_pressure_data(MLD_DEPTH_PATH, 'MONTHLY_MEAN_MLD_PRESSURE')
heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_DATA_PATH)
ekman_ds = load_and_prepare_dataset(EK_DATA_PATH)
chris_ds = load_and_prepare_dataset(CHRIS_SCHEME_DATA_PATH)

temperature = mld_temperature_ds['__xarray_dataarray_variable__']
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

# mld_depth_ds = mld_depth_ds.rename({'MONTH': 'TIME'})
print(heat_flux_ds)

#%%
heat_flux = (heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_ishf'] +
             heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_snlwrf'])
heat_flux.attrs.update(units='W m**-2', long_name='Net Surface Heat Flux')
heat_flux.name = 'NET_HEAT_FLUX'

heat_flux_monthly_mean = get_monthly_mean_from_year(heat_flux, year=YEAR)
heat_flux_anomaly = get_anomaly(heat_flux, heat_flux_monthly_mean)
# heat_flux_anomaly = heat_flux_anomaly.drop_vars(['MONTH'])
print('heat flux anomaly:\n', heat_flux_anomaly)

ekman_anomaly = ekman_ds['Q_Ek_anom']

chris_ds = chris_ds['ARGO_TEMPERATURE_ANOMALY']

times = heat_flux_anomaly.TIME.values
t0 = times[0]

# --- Helper to create efficient output arrays ---------------------------------------------
def create_output_array(name, long_name):
    """Pre-allocates an xr.DataArray with the correct coords."""
    return xr.DataArray(
        np.nan,
        coords={
            "TIME": ekman_anomaly.TIME,  # daily axis from the forcing
            "LATITUDE": temperature_anomaly["LATITUDE"],
            "LONGITUDE": temperature_anomaly["LONGITUDE"],
        },
        dims=("TIME", "LATITUDE", "LONGITUDE"),
        name=name,
        attrs={**temperature_anomaly.attrs,
               'units': 'K',
               'long_name': long_name}
    )


mld_depth_ds = tile_monthly_to_daily(mld_depth_ds, year=YEAR)
mld_depth_ds = mld_depth_ds.assign_coords(TIME=heat_flux_anomaly.TIME)


# --- Model Computation Functions ----------------------------------------------------------
def compute_implicit(start_time=1) -> xr.DataArray:
    print("Running Implicit Scheme...")
    model_anomaly_ds = create_output_array(
        "T_model_anom_implicit",
        "Mixed-layer temperature anomaly (implicit) using daily data from 2004"
    )

    # Initial condition (zero anomaly)
    T_prev = xr.zeros_like(temperature_anomaly.isel(TIME=0))
    model_anomaly_ds.loc[dict(TIME=t0)] = T_prev

    # Loop from the second time step
    for i, day in enumerate(times[start_time:], start=1):
        # 'month' is t^{n+1} (e.g., 1.5, 2.5, ...)

        # Get coefficients for time n+1
        # Your logic: for t=1.5, moy=1.5. sel(1.5, nearest) -> 2.0 (Feb MLD)
        # moy_n_plus_1 = ((month - 0.5) % 12) + 0.5


        denominator = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=day, method='nearest', tolerance=0.51)
        )
        # a^{n+1}
        damp_factor = GAMMA / denominator

        # b^{n+1}
        forcing_term = (
            heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
            + ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
        ) / denominator

        # T^{n+1} = (T^n + dt*b^{n+1}) / (1 + dt*a^{n+1})
        T_next = (T_prev + SECONDS_DAY * forcing_term) / (1 + SECONDS_DAY * damp_factor)

        # Store result and update T_prev
        model_anomaly_ds.loc[dict(TIME=day)] = T_next
        T_prev = T_next

    return model_anomaly_ds

def compute_explicit(start_time=1) -> xr.DataArray:
    print("Running Explicit Scheme...")
    model_anomaly_ds = create_output_array(
        "T_model_anom_explicit",
        "Mixed-layer temperature anomaly (explicit) using daily data from 2004"
    )

    # Initial condition (zero anomaly)
    T_prev = xr.zeros_like(temperature_anomaly.isel(TIME=0))
    model_anomaly_ds.loc[dict(TIME=t0)] = T_prev

    # Loop from the second time step
    for i, day in enumerate(times[start_time:], start=1):
        # 'month' is t^{n+1} (e.g., 1.5, 2.5, ...)
        prev_time = times[i - 1]  # 'prev_time' is t^n (e.g., 0.5, 1.5, ...)

        # Get coefficients for time n
        # Your logic: for t=0.5, moy=0.5. sel(0.5, nearest) -> 1.0 (Jan MLD)
        # moy_n = ((prev_time - 0.5) % 12) + 0.5

        denominator = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=day, method='nearest', tolerance=0.51))
        # a^n
        damp_factor = GAMMA / denominator

        # b^n
        # *** LOGIC FIX ***: Use prev_time, not moy_n, to select forcing data.
        forcing_term = (
            heat_flux_anomaly.sel(TIME=prev_time, method='nearest', tolerance=0.51)
            + ekman_anomaly.sel(TIME=prev_time, method='nearest', tolerance=0.51)) / denominator

        # T^{n+1} = (1 - dt*a^n)T^n + dt*b^n
        T_next = (1 - SECONDS_DAY * damp_factor) * T_prev + SECONDS_DAY * forcing_term

        # Store result and update T_prev
        model_anomaly_ds.loc[dict(TIME=day)] = T_next
        T_prev = T_next

    return model_anomaly_ds

def compute_semi_implicit(start_time=1) -> xr.DataArray:
    print("Running Semi-Implicit Scheme...")
    model_anomaly_ds = create_output_array(
        "T_model_anom_semi_implicit",
        "Mixed-layer temperature anomaly (semi-implicit) using daily data from 2004"
    )

    # Initial condition (zero anomaly)
    T_prev = xr.zeros_like(temperature_anomaly.isel(TIME=0))
    model_anomaly_ds.loc[dict(TIME=t0)] = T_prev

    # Loop from the second time step
    for i, day in enumerate(times[start_time:], start=1):
        # 'month' is t^{n+1} (e.g., 1.5, 2.5, ...)
        prev_time = times[i - 1]  # 'prev_time' is t^n (e.g., 0.5, 1.5, ...)
        # moy_n = ((prev_time - 0.5) % 12) + 0.5
        # moy_n_plus_1 = ((month - 0.5) % 12) + 0.5

        denominator_n_plus_1 = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=day, method='nearest', tolerance=0.51)
        )

        denominator_n = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=prev_time, method='nearest', tolerance=0.51)
        )
        # a^{n+1}
        damp_factor = GAMMA / denominator_n_plus_1

        # b^{n+1}
        forcing_term = (
            heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
            + ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
        ) / denominator_n

        # T^{n+1} = (T^n + dt*b^{n+1}) / (1 + dt*a^{n+1})
        T_next = (T_prev + SECONDS_DAY * forcing_term) / (1 + SECONDS_DAY * damp_factor)

        # Store result and update T_prev
        model_anomaly_ds.loc[dict(TIME=prev_time)] = T_next
        T_prev = T_next

    return model_anomaly_ds

def compute_crank(start_time=1) -> xr.DataArray:

    print("Running Crank Nicolson Scheme...")
    model_anomaly_ds = create_output_array(
        "T_model_anom_crank_nicolson",
        "Mixed-layer temperature anomaly (Crank Nicolson) using daily data from 2004"
    )

    # Initial condition (zero anomaly)
    T_prev = xr.zeros_like(temperature_anomaly.isel(TIME=0))
    model_anomaly_ds.loc[dict(TIME=t0)] = T_prev

    # Loop from the second time step
    for i, day in enumerate(times[start_time:], start=1):
        # 'month' is t^{n+1} (e.g., 1.5, 2.5, ...)

        prev_time = times[i - 1]  # 'prev_time' is t^n (e.g., 0.5, 1.5, ...)
        # moy_n = ((prev_time - 0.5) % 12) + 0.5
        # moy_n_plus_1 = ((month - 0.5) % 12) + 0.5
        day_plus_1 = day + 1

        denominator_n_plus_1 = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=day, method='nearest', tolerance=0.51)
        )

        denominator_n = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=prev_time, method='nearest', tolerance=0.51)
        )

        # a^{n+1}
        damp_factor_nplus1 = GAMMA / denominator_n_plus_1
        damp_factor_n = GAMMA / denominator_n

        # b^{n+1}
        forcing_term_n = (
            heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
            + ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
        )/ denominator_n

        forcing_term_nplus1 = (
        (heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
            # ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
        )) / denominator_n_plus_1

        # T^{n+1} = (T^n + dt*b^{n+1}) / (1 + dt*a^{n+1})
        T_next = ((1-0.5*damp_factor_n*SECONDS_DAY)*T_prev
                  + SECONDS_DAY * (0.5*forcing_term_n + 0.5*forcing_term_nplus1)
                )/ (1 + 0.5* SECONDS_DAY * damp_factor_nplus1)

        # Store result and update T_prev
        model_anomaly_ds.loc[dict(TIME=prev_time)] = T_next
        T_prev = T_next

    return model_anomaly_ds


# --- Run both simulations -------------------------------------------------
sim_implicit = compute_implicit(start_time=1)
sim_explicit = compute_explicit(start_time=1)
sim_semi_implicit = compute_semi_implicit(start_time=1)
sim_crank = compute_crank(start_time=1)

# sim_implicit.to_netcdf('datasets/sim_Implicit_Scheme_Test_ConstDamp(10)_daily_2004')
# sim_explicit.to_netcdf('datasets/sim_explicit_Scheme_Test_ConstDamp(10)_daily_2004')
# sim_semi_implicit.to_netcdf('datasets/sim_semi_implicit_Scheme_Test_ConstDamp(10)_daily_2004')
# sim_crank.to_netcdf('datasets/sim_crank_Scheme_Test_ConstDamp(10)_daily_2004')

# --- Side-by-side animation -------------------------------------------------
print("Starting animation...")
# Use the time grid from one of the simulations
anim_times = sim_implicit.TIME.values

# Common lon/lat
lons = sim_implicit.LONGITUDE.values
lats = sim_implicit.LATITUDE.values

VMIN, VMAX = -10,10

fig = plt.figure(figsize=(16, 6))

# Explicit panel
ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
mesh_exp = ax1.pcolormesh(lons, lats, sim_explicit.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax1.coastlines()
ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl1.top_labels = False
gl1.right_labels = False
gl1.xlines = False
gl1.ylines = False
gl1.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl1.ylocator = LatitudeLocator()
gl1.xformatter = LongitudeFormatter()
gl1.yformatter = LatitudeFormatter()
gl1.ylabel_style = {'size': 12, 'color': 'gray'}
gl1.xlabel_style = {'size': 12, 'color': 'gray'}
ax1.set_title(f'Explicit — t={anim_times[0]} in year{YEAR}')

# Implicit panel
ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
mesh_imp = ax2.pcolormesh(lons, lats, sim_implicit.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax2.coastlines()
ax2.set_xlim(-180, 180)
ax2.set_ylim(-90, 90)
gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl2.top_labels = False
gl2.right_labels = False
gl2.left_labels = True
gl2.xlines = False
gl2.ylines = False
gl2.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl2.ylocator = LatitudeLocator()
gl2.xformatter = LongitudeFormatter()
gl2.yformatter = LatitudeFormatter()
gl2.ylabel_style = {'size': 12, 'color': 'gray'}
gl2.xlabel_style = {'size': 12, 'color': 'gray'}
ax2.set_title(f'Implicit — t={anim_times[0]}')

# Semi-Implicit panel
ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
mesh_semi_imp = ax3.pcolormesh(lons, lats, sim_semi_implicit.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax3.coastlines()
ax3.set_xlim(-180, 180)
ax3.set_ylim(-90, 90)
gl3 = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl3.top_labels = False
gl3.right_labels = False
gl3.left_labels = True
gl3.xlines = False
gl3.ylines = False
gl3.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl3.ylocator = LatitudeLocator()
gl3.xformatter = LongitudeFormatter()
gl3.yformatter = LatitudeFormatter()
gl3.ylabel_style = {'size': 12, 'color': 'gray'}
gl3.xlabel_style = {'size': 12, 'color': 'gray'}
ax3.set_title(f'Semi-Implicit — t={anim_times[0]} in year{YEAR}')

# Crank Nicolson panel
ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
mesh_crank = ax4.pcolormesh(lons, lats, sim_crank.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax4.coastlines()
ax4.set_xlim(-180, 180)
ax4.set_ylim(-90, 90)
gl4 = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl4.top_labels = False
gl4.right_labels = False
gl4.xlines = False
gl4.ylines = False
gl4.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl4.ylocator = LatitudeLocator()
gl4.xformatter = LongitudeFormatter()
gl4.yformatter = LatitudeFormatter()
gl4.ylabel_style = {'size': 12, 'color': 'gray'}
gl4.xlabel_style = {'size': 12, 'color': 'gray'}
ax4.set_title(f'Crank Nicolson — t={anim_times[0]} in year{YEAR}')


#--------- Chris Plot Settings--------------------------------
anim_times_chris = chris_ds.TIME.values

# Common lon/lat
lons = chris_ds.LONGITUDE.values
lats = chris_ds.LATITUDE.values

fig_chris = plt.figure(figsize=(16, 6))

# Chris panel
ax5 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
mesh_chris = ax5.pcolormesh(lons, lats, chris_ds.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax5.coastlines()
ax5.set_xlim(-180, 180)
ax5.set_ylim(-90, 90)
gl5 = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl5.top_labels = False
gl5.right_labels = False
gl5.xlines = False
gl5.ylines = False
gl5.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl5.ylocator = LatitudeLocator()
gl5.xformatter = LongitudeFormatter()
gl5.yformatter = LatitudeFormatter()
gl5.ylabel_style = {'size': 12, 'color': 'gray'}
gl5.xlabel_style = {'size': 12, 'color': 'gray'}
ax5.set_title(f'Explicit — t={anim_times_chris[0]} in year{YEAR}')


ax6 = plt.subplot(1,2,2, projection = ccrs.PlateCarree())
mesh_observed = ax6.pcolormesh(lons, lats, temperature_anomaly.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax6.coastlines()
ax6.set_xlim(-180, 180)
ax6.set_ylim(-90, 90)
gl6 = ax6.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl6.top_labels = False
gl6.right_labels = False
gl6.xlines = False
gl6.ylines = False
gl6.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl6.ylocator = LatitudeLocator()
gl6.xformatter = LongitudeFormatter()
gl6.yformatter = LatitudeFormatter()
gl6.ylabel_style = {'size': 12, 'color': 'gray'}
gl6.xlabel_style = {'size': 12, 'color': 'gray'}
ax6.set_title(f'Observation — t={anim_times_chris[0]} in year{YEAR}')

# ----Observed Tm'-----------------------------------------------------

# Shared colorbar
cbar = fig.colorbar(mesh_imp, ax=[ax1, ax2,ax3,ax4], shrink=0.8,
                    label=sim_implicit.attrs.get('units', 'K'))

cbar_chris = fig_chris.colorbar(mesh_chris, ax=ax5, shrink=0.8,
                    label=chris_ds.attrs.get('units', 'K'))


cbar_obs = fig_chris.colorbar(mesh_observed, ax=ax6, shrink=0.8,
                    label=temperature_anomaly.attrs.get('units', 'K'))

def update(frame):
    # Update explicit
    Z_exp = sim_explicit.isel(TIME=frame).values
    mesh_exp.set_array(Z_exp.ravel())

    # Update implicit
    Z_imp = sim_implicit.isel(TIME=frame).values
    mesh_imp.set_array(Z_imp.ravel())

    # Update semi-implicit
    Z_semi_imp = sim_semi_implicit.isel(TIME=frame).values
    mesh_semi_imp.set_array(Z_semi_imp.ravel())

    Z_crank = sim_crank.isel(TIME=frame).values
    mesh_crank.set_array(Z_crank.ravel())
   
    # Update titles
    current_time = anim_times[frame]
    ax1.set_title(f'Explicit — Day: {current_time} in year{YEAR}')
    ax2.set_title(f'Implicit — Day: {current_time} in year{YEAR}')
    ax3.set_title(f'Semi-Implicit — Day: {current_time} in year{YEAR}')
    ax4.set_title(f'Crank Nicolson — Day: {current_time} in year{YEAR}')
    return [mesh_exp, mesh_imp, mesh_semi_imp, mesh_crank]


def update_chris(frame):
    Z_chris = chris_ds.isel(TIME=frame).values
    mesh_chris.set_array(Z_chris.ravel())
    # Update titles
    current_time = anim_times_chris[frame]
    ax5.set_title(f'Chris Scheme — Day: {current_time} in year{YEAR}')

    Z_obs = temperature_anomaly.isel(TIME=frame).values
    mesh_observed.set_array(Z_obs.ravel())
    # Update titles
    current_time = anim_times_chris[frame]
    ax6.set_title(f'Observation — Day: {current_time} in year{YEAR}')
    return [mesh_chris, mesh_observed]


# Create and show animation
animation = FuncAnimation(fig, update, frames=len(anim_times), interval=300, blit=False)
animation_chris = FuncAnimation(fig_chris, update_chris, frames=len(anim_times_chris), interval=300, blit=False)

# plt.tight_layout()
plt.show()

