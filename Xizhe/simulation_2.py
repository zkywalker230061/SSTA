#%%
import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
#import esmpy as ESMF
from read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
matplotlib.use('TkAgg')

#HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "../datasets/heat_flux_interpolated_all_contributions.nc"
#HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
MLD_TEMP_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature-(2004-2018).nc"
MLD_DEPTH_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
#TEMP_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc"
HEAT_FLUX_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
EK_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Current_Anomaly.nc"

mld_temperature_ds = xr.open_dataset(MLD_TEMP_PATH, decode_times=False)
mld_depth_ds = xr.open_dataset(MLD_DEPTH_PATH, decode_times=False)
heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_DATA_PATH)
ekman_ds = load_and_prepare_dataset(EK_DATA_PATH)

temperature = mld_temperature_ds['__xarray_dataarray_variable__']
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

mld_depth_ds = mld_depth_ds.rename({'MONTH': 'TIME'})

heat_flux = (heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_ishf'] + heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_snlwrf'])
heat_flux.attrs.update(units='W m**-2', long_name='Net Surface Heat Flux')
heat_flux.name = 'NET_HEAT_FLUX'
heat_flux_monthly_mean = get_monthly_mean(heat_flux)
heat_flux_anomaly = get_anomaly(heat_flux, heat_flux_monthly_mean)
heat_flux_anomaly = heat_flux_anomaly.drop_vars(['MONTH'])

ekman_anomaly = ekman_ds['Q_Ek_anom']
# =========================
# Time-stepping framework
# =========================

RHO_O = 1025.0
C_O = 4100.0
SECONDS_MONTH = 30.4375 * 24 * 60 * 60
GAMMA = 10.0  # damping [W m^-2 K^-1] equivalent in your formulation

# -- Helpers -------------------------------------------------------

def _phi1(a_dt: xr.DataArray):
    """phi1(z) = (1 - exp(-z)) / z, with safe limit phi1(0)=1."""
    alpha = xr.ufuncs.exp(-a_dt)
    return xr.where(a_dt != 0, (1.0 - alpha) / a_dt, 1.0)

def _dbar_to_meters(p_dbar: xr.DataArray, lat: xr.DataArray):
    """
    Convert pressure (dbar) to depth (m) using TEOS-10.
    Requires latitude grid to be aligned with p.
    """
    # Broadcast latitude to (LATITUDE, LONGITUDE) if needed
    lat_b = lat
    if set(lat.dims) != set(p_dbar.dims):
        lat_b = lat.broadcast_like(p_dbar)
    # z_from_p returns negative below sea level; flip sign to get positive depth
    z_m = xr.apply_ufunc(
        lambda P, LA: -gsw.z_from_p(P, LA),
        p_dbar, lat_b,
        vectorize=True, dask="parallelized", output_dtypes=[float],
    )
    return z_m

def _prepare_monthly_depth_meters(mld_depth_ds: xr.Dataset, lat_coord="LATITUDE"):
    """
    Returns monthly MLD depth (meters) on TIME in {0.5,1.5,...,11.5}.
    If your file already stores meters, set `already_meters=True`.
    """
    p = mld_depth_ds["MONTHLY_MEAN_MLD_PRESSURE"]  # likely dbar
    lat = mld_depth_ds[lat_coord]
    # Convert dbar -> meters (near-surface 1 dbar ≈ 1 m, but use TEOS-10 for rigor)
    h_m = _dbar_to_meters(p, lat)
    # Safety floor to avoid division by tiny depths
    h_m = h_m.where(np.isfinite(h_m) & (h_m > 5.0))
    return h_m

def _coefficients(month, h_monthly_m):
    """
    Build coefficients a_next, b_next, b_prev for a given model month.
    h_monthly_m: depth (m) with TIME in {0.5,1.5,...,11.5}
    """
    # Index seasonal MLD by month-in-year at n+1
    h_next = h_monthly_m.sel(TIME=(month % 12 + 0.5))
    a_next = GAMMA / (RHO_O * C_O * h_next)

    # Forcing at n and n+1 (using same h at n+1; you could also use h at n for b_prev)
    b_next = (heat_flux_anomaly.sel(TIME=month) + ekman_anomaly.sel(TIME=month)) / (RHO_O * C_O * h_next)
    b_prev = (heat_flux_anomaly.sel(TIME=month-1) + ekman_anomaly.sel(TIME=month-1)) / (RHO_O * C_O * h_next)
    return a_next, b_next, b_prev, h_next

# -- Schemes -------------------------------------------------------

def step_explicit(prev, month, h_monthly_m, dt=SECONDS_MONTH):
    """Forward Euler: T^{n+1} = (1 - dt*a^n) T^n + dt*b^n"""
    # Use n ("previous") coefficients
    h_prev = h_monthly_m.sel(TIME=((month-1) % 12 + 0.5))
    a_prev = GAMMA / (RHO_O * C_O * h_prev)
    b_prev = (heat_flux_anomaly.sel(TIME=month-1) + ekman_anomaly.sel(TIME=month-1)) / (RHO_O * C_O * h_prev)
    return (1.0 - dt * a_prev) * prev + dt * b_prev

def step_implicit(prev, month, h_monthly_m, dt=SECONDS_MONTH):
    """Backward Euler: T^{n+1} = (T^n + dt*b^{n+1}) / (1 + dt*a^{n+1})"""
    a_next, b_next, _, _ = _coefficients(month, h_monthly_m)
    return (prev + dt * b_next) / (1.0 + dt * a_next)

def step_semi_implicit_damp(prev, month, h_monthly_m, dt=SECONDS_MONTH):
    """Implicit damping, explicit forcing: (T^n + dt*b^n)/(1 + dt*a^{n+1})"""
    a_next, _, b_prev, _ = _coefficients(month, h_monthly_m)
    return (prev + dt * b_prev) / (1.0 + dt * a_next)

def step_crank_nicolson(prev, month, h_monthly_m, dt=SECONDS_MONTH):
    """
    CN (2nd order):
    T^{n+1} = [(1 - 0.5 a^{n+1} dt) T^n + 0.5 dt (b^n + b^{n+1})] / (1 + 0.5 a^{n+1} dt)
    """
    a_next, b_next, b_prev, _ = _coefficients(month, h_monthly_m)
    numer = (1.0 - 0.5 * a_next * dt) * prev + 0.5 * dt * (b_prev + b_next)
    denom = (1.0 + 0.5 * a_next * dt)
    return numer / denom

def step_integrating_factor(prev, month, h_monthly_m, dt=SECONDS_MONTH):
    """
    Exact step for linear damping with piecewise-constant a,b on the step:
    T^{n+1} = e^{-a^{n+1} dt} T^n + dt * phi1(a^{n+1} dt) * b^{n+1}
    """
    a_next, b_next, _, _ = _coefficients(month, h_monthly_m)
    a_dt = a_next * dt
    alpha = xr.ufuncs.exp(-a_dt)
    phi1 = _phi1(a_dt)
    return alpha * prev + dt * phi1 * b_next

def step_etd_cn(prev, month, h_monthly_m, dt=SECONDS_MONTH):
    """
    Exponential Time-Differencing Crank–Nicolson:
    T^{n+1} = e^{-a^{n+1} dt} T^n + dt * phi1(a^{n+1} dt) * 0.5 * (b^n + b^{n+1})
    """
    a_next, b_next, b_prev, _ = _coefficients(month, h_monthly_m)
    a_dt = a_next * dt
    alpha = xr.ufuncs.exp(-a_dt)
    phi1 = _phi1(a_dt)
    return alpha * prev + dt * phi1 * 0.5 * (b_prev + b_next)

SCHEMES = {
    "explicit": step_explicit,
    "implicit": step_implicit,
    "semi_implicit_damp": step_semi_implicit_damp,
    "crank_nicolson": step_crank_nicolson,
    "integrating_factor": step_integrating_factor,
    "etd_cn": step_etd_cn,
}

# -- Driver --------------------------------------------------------

def run_model(scheme: str = "implicit"):
    """
    Integrate T anomaly with selected scheme.
    Returns xr.DataArray with dims (TIME, LATITUDE, LONGITUDE).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from {list(SCHEMES)}")

    # Precompute monthly depth (meters) once
    h_monthly_m = _prepare_monthly_depth_meters(mld_depth_ds)

    # Model times from your anomaly field
    times = temperature_anomaly.TIME.values

    # Initial condition: zero anomaly at the first step (matches your earlier construction)
    start = float(times[0])  # e.g., 0.5
    prev = xr.zeros_like(temperature_anomaly.sel(TIME=start))

    out = prev.expand_dims(TIME=[start])

    stepper = SCHEMES[scheme]
    for month in times[1:]:
        cur = stepper(prev, float(month), h_monthly_m, dt=SECONDS_MONTH)
        cur = cur.assign_attrs(temperature_anomaly.attrs)  # keep attrs/units if desired
        out = xr.concat([out, cur.expand_dims(TIME=[float(month)])], dim="TIME")
        prev = cur

    out.name = f"T_model_anom_{scheme}"
    out.attrs.update(units="K", long_name=f"Mixed-layer temperature anomaly ({scheme})")
    return out

# =========================
# Example: run and compare
# =========================

sim_implicit = run_model("implicit")
sim_if       = run_model("integrating_factor")
sim_etdcn    = run_model("etd_cn")

# Stack into one Dataset for easy comparison
comparison = xr.Dataset({
    sim_implicit.name: sim_implicit,
    sim_if.name: sim_if,
    sim_etdcn.name: sim_etdcn,
})

print(comparison)
