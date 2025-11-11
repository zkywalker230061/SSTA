import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset


#---1. Read Files ----------------------------------------------------------
semi_implicit_path = r"C:\Users\jason\MSciProject\Semi_Implicit_Scheme_Test_ConstDamp(10)"
implicit_path = r"C:\Users\jason\MSciProject\Implicit_Scheme_Test_ConstDamp(10)"
explicit_path = r"C:\Users\jason\MSciProject\Explicit_Scheme_Test_ConstDamp(10)"
crank_path = r"C:\Users\jason\MSciProject\Crack_Scheme_Test_ConstDamp(10)"
observed_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Temperature(T_m).nc"

# --- Load and Prepare Data (assuming helper functions are correct) --------
observed_temp = xr.open_dataset(observed_path, decode_times=False)
implicit = load_and_prepare_dataset(implicit_path)
explicit = load_and_prepare_dataset(explicit_path)
crank = load_and_prepare_dataset(crank_path)
semi_implicit = load_and_prepare_dataset(semi_implicit_path)

# --- Extracting the correct DataArray -------------------
temperature = observed_temp['__xarray_dataarray_variable__']
implicit = implicit["T_model_anom_implicit"]
explicit = explicit["T_model_anom_explicit"]
crank = crank["T_model_anom_crank_nicolson"]
semi_implicit = semi_implicit["T_model_anom_semi_implicit"]


#---- Defining the Anomaly Temperature Dataset ----------------------
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)


#-----2. Defining Error Function Calculation
times = implicit.TIME.values
# def calculate_error(test_data, observed_data):
