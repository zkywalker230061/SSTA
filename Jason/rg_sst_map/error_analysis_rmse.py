
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


observed_temp_ds = xr.open_dataset(observed_path, decode_times=False) 
all_anomalies = load_and_prepare_dataset(all_anomalies_path)
# implicit_ds = load_and_prepare_dataset(implicit_path)
# explicit_ds = load_and_prepare_dataset(explicit_path)
# crank_ds = load_and_prepare_dataset(crank_path)
# semi_implicit_ds = load_and_prepare_dataset(semi_implicit_path)
# chris_ds = load_and_prepare_dataset(chris_path)


observed_temperature = observed_temp_ds['__xarray_dataarray_variable__']
observed_temperature_monthly_average = get_monthly_mean(observed_temperature)
observed_temperature_anomaly = get_anomaly(observed_temperature, observed_temperature_monthly_average)

schemes = {
    "Explicit": all_anomalies["EXPLICIT"],
    "Implicit": all_anomalies["IMPLICIT"],
    "Semi-Implicit": all_anomalies["SEMI_IMPLICIT"],
    "Chris Mean K": all_anomalies["CHRIS_MEAN_K"],
    "Chris Capped": all_anomalies["CHRIS_CAPPED_EXPONENT"]
}


# print(implicit_ds)
# print(explicit_ds)
# print(semi_implicit_ds)
# print(crank_ds)
# print(chris_ds)
# print(observed_temp_ds)

def calculate_RMSE (obs, model, dim = 'TIME'):
    """
    Calculates Root Mean Square Error.
    Formula: sqrt( mean( (obs - model)^2 ) )
    """
    error = model - obs
    squared_error = error ** 2
    mean_squared_error = squared_error.mean(dim=dim)
    rmse = np.sqrt(mean_squared_error)
    return rmse



for key in schemes:
    schemes[key] = schemes[key].isel(TIME=slice(1, None))




for i, (scheme_name, model_da) in enumerate(schemes.items()):
    # Calculate RMSE over the 'TIME' dimension

    rmse_map = calculate_RMSE(observed_temperature_anomaly, model_da, dim='TIME')
    
    # Plotting
    fig, ax = plt.subplots(3,2, figsize=(8,6), subplot_kw={'projection': ccrs.PlateCarree()})
    rmse_map.plot(ax=ax, cmap='RdBu_r', cbar_kwargs={'label': 'RMSE (°C)'}, vmin = 0, vmax = 10)
    max_rmse = rmse_map.max().item()
    print(scheme_name, max_rmse)
    ax.set_title(f'{scheme_name} Scheme - Overall RMSE')
    plt.show()