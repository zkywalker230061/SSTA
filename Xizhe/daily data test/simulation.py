import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset, load_pressure_data
from matplotlib.animation import FuncAnimation
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)

# --- File Paths (assuming these are correct) ------------------------------------------------
MLD_TEMP_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
MLD_DEPTH_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
HEAT_FLUX_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/ERA5-ARGO_Mean_Surface_Heat_Flux_Daily.nc"
TURBULENT_SURFACE_STRESS = '/Users/julia/Desktop/SSTA/datasets/datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress_Daily_2004.nc'
EK_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman Current Anomaly - Daily 2004 - Test"
CHRIS_SCHEME_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/model_anomaly_exponential_damping_implicit.nc"

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

heat_flux = (heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_ishf'] +
             heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_snlwrf'])
heat_flux.attrs.update(units='W m**-2', long_name='Net Surface Heat Flux')
heat_flux.name = 'NET_HEAT_FLUX'
heat_flux_monthly_mean = get_monthly_mean(heat_flux)
heat_flux_anomaly = get_anomaly(heat_flux, heat_flux_monthly_mean)
heat_flux_anomaly = heat_flux_anomaly.drop_vars(['MONTH'])

ekman_anomaly = ekman_ds['Q_Ek_anom']

chris_ds = chris_ds['ARGO_TEMPERATURE_ANOMALY']

# --- Model Constants -----------------------------------------------------------------------
RHO_O = 1025.0  # kg/m^3
C_O = 4100.0  # J/(kg K)
SECONDS_DAY = 24 * 60 * 60  # s
GAMMA = 10.0  # bulk damping factor
times = ekman_anomaly.TIME.values
t0 = times[0]
month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

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

def tile_monthly_to_daily(gradient_monthly, tile_array):
    daily_list = []
    for i in range(12):
        monthly_field = gradient_monthly.isel(MONTH=i)
        repeated = monthly_field.expand_dims(TIME=tile_array[i]).copy()
        daily_list.append(repeated)
    
    daily_gradient = xr.concat(daily_list, dim="TIME")
    return daily_gradient

mld_depth_ds = tile_monthly_to_daily(mld_depth_ds, month_lengths)
mld_depth_ds = mld_depth_ds.assign_coords(TIME=ekman_anomaly.TIME)
print(mld_depth_ds)

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
            heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51) +
            ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
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
            heat_flux_anomaly.sel(TIME=prev_time, method='nearest', tolerance=0.51) +
            ekman_anomaly.sel(TIME=prev_time, method='nearest', tolerance=0.51)
        ) / denominator

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
            heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51) +
            ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
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
            heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51) +
            ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51)
        )/ denominator_n

        forcing_term_nplus1 = (
        (heat_flux_anomaly.sel(TIME=day, method='nearest', tolerance=0.51) +
            ekman_anomaly.sel(TIME=day, method='nearest', tolerance=0.51))
        ) / denominator_n_plus_1

        # T^{n+1} = (T^n + dt*b^{n+1}) / (1 + dt*a^{n+1})
        T_next = ((1-0.5*damp_factor_n*SECONDS_DAY)*T_prev
                  + SECONDS_DAY * (0.5*forcing_term_n + 0.5*forcing_term_nplus1)
                )/ (1 + 0.5* SECONDS_DAY * damp_factor_nplus1)

        # Store result and update T_prev
        model_anomaly_ds.loc[dict(TIME=prev_time)] = T_next
        T_prev = T_next

    return model_anomaly_ds


# --- Run both simulations -------------------------------------------------
sim_implicit = compute_implicit(start_time=0)
sim_explicit = compute_explicit(start_time=0)
sim_semi_implicit = compute_semi_implicit(start_time=0)
sim_crank = compute_crank(start_time=0)

print(sim_explicit)