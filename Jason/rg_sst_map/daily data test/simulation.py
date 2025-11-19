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

# --- File Paths (assuming these are correct) -------------------------------
MLD_TEMP_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
MLD_DEPTH_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
HEAT_FLUX_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/ERA5-ARGO_Mean_Surface_Heat_Flux_Daily.nc"
TURBULENT_SURFACE_STRESS = '/Users/julia/Desktop/SSTA/datasets/datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress_Daily.nc'
EK_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Current_Anomaly.nc"
CHRIS_SCHEME_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/model_anomaly_exponential_damping_implicit.nc"

# --- Load and Prepare Data (assuming helper functions are correct) --------
mld_temperature_ds = xr.open_dataset(MLD_TEMP_PATH, decode_times=False)
mld_depth_ds = load_pressure_data(MLD_DEPTH_PATH, 'MONTHLY_MEAN_MLD_PRESSURE')
heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_DATA_PATH)
ekman_ds = load_and_prepare_dataset(EK_DATA_PATH)
chris_ds = load_and_prepare_dataset(CHRIS_SCHEME_DATA_PATH)

temperature = mld_temperature_ds['__xarray_dataarray_variable__']
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)