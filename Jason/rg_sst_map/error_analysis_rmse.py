
import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)


# semi_implicit_path = "/Users/julia/Desktop/SSTA/datasets/Semi_Implicit_Scheme_Test_ConstDamp(10)"
# implicit_path = "/Users/julia/Desktop/SSTA/datasets/Implicit_Scheme_Test_ConstDamp(10)"
# explicit_path = "/Users/julia/Desktop/SSTA/datasets/Explicit_Scheme_Test_ConstDamp(10)"
# crank_path = "/Users/julia/Desktop/SSTA/datasets/Crack_Scheme_Test_ConstDamp(10)"
# chris_path = "/Users/julia/Desktop/SSTA/datasets/model_anomaly_exponential_damping_implicit.nc"
observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc"
EK_DATA_PATH = r"C:\Users\jason\MSciProject\Ekman_Current_Anomaly.nc"
HEAT_FLUX_DATA_PATH = r"C:\Users\jason\MSciProject\ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
all_anomalies_path = r"C:\Users\jason\MSciProject\all_anomalies.nc"
hopohopo_path = r"C:\Users\jason\MSciProject\chris_prev_cur_scheme_denoised.nc"


observed_temp_ds = xr.open_dataset(observed_path, decode_times=False) 
all_anomalies = load_and_prepare_dataset(all_anomalies_path)
# implicit_ds = load_and_prepare_dataset(implicit_path)
# explicit_ds = load_and_prepare_dataset(explicit_path)
# crank_ds = load_and_prepare_dataset(crank_path)
# semi_implicit_ds = load_and_prepare_dataset(semi_implicit_path)
# chris_ds = load_and_prepare_dataset(chris_path)
hopohopo = load_and_prepare_dataset(hopohopo_path)


observed_temperature = observed_temp_ds['__xarray_dataarray_variable__']
observed_temperature_monthly_average = get_monthly_mean(observed_temperature)
observed_temperature_anomaly = get_anomaly(observed_temperature, observed_temperature_monthly_average)

schemes = {
    "Explicit": all_anomalies["EXPLICIT"],
    "Implicit": all_anomalies["IMPLICIT"],
    "Semi-Implicit": all_anomalies["SEMI_IMPLICIT"],
    "Chris Mean K": all_anomalies["CHRIS_MEAN_K"],
    "Chris Capped": all_anomalies["CHRIS_CAPPED_EXPONENT"],
    "Hopohopo": hopohopo["__xarray_dataarray_variable__"]
}




for key in schemes:
    schemes[key] = schemes[key].isel(TIME=slice(1, None))


def calculate_RMSE (obs, model, dim = 'TIME'):
    """
    Calculates Root Mean Square Error.
    Formula: sqrt( mean( (obs - model)^2 ) )
    """
    obs = obs.isel(TIME=slice(1, None))
    print(obs.shape, model.shape)
    error = model - obs
    squared_error = error ** 2
    mean_squared_error = squared_error.mean(dim=dim)
    rmse = np.sqrt(mean_squared_error)
    return rmse




fig, axes = plt.subplots(3, 2, figsize=(12,7))

# for ax, (scheme_name, model_da) in zip(axes.flat, schemes.items()):
#     # Calculate RMSE over the 'TIME' dimension
#     model_da_mean = get_monthly_mean(model_da)
#     model_da_anomaly = get_anomaly(model_da, model_da_mean)
#     rmse_map = calculate_RMSE(observed_temperature_anomaly, model_da_anomaly, dim='TIME')
    
#     # Plotting
#     # ax = plt.subplot(3, 2, i + 1)
#     rmse_map.plot(ax=ax, cmap='nipy_spectral', cbar_kwargs={'label': 'RMSE (K)'}, vmin = 0, vmax = 3)
#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Lattitude")
#     ax.set_title(f'{scheme_name} Scheme - Overall RMSE')
#     max_rmse = rmse_map.max().item()
#     print(scheme_name, max_rmse)
# plt.tight_layout()
# plt.show()

# Test

for ax, (scheme_name, model_da) in zip(axes.flat, schemes.items()):
    # 1. Calculate the simple global mean (DC offset)
    simple_mean_val = model_da.mean().item()
    
    print(f"--- {scheme_name} ---")
    print(f"Original Mean (Drift): {simple_mean_val}")
    
    # 2. Force a Simple Correction (bypass get_monthly_mean for a moment)
    # This subtracts the average over time, recentering the data on 0.
    model_da_corrected = model_da - model_da.mean(dim='TIME')
    
    # 3. Calculate RMSE on the FORCED corrected data
    rmse_map = calculate_RMSE(observed_temperature_anomaly, model_da_corrected, dim='TIME')
    
    # ... plotting code ...
    print(f"New Max RMSE: {rmse_map.max().item()}\n")