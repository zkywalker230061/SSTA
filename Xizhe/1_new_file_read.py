import xarray as xr

observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Datasets.nc"
T_SUB_GRAD_PATH = "datasets/New_Entrainment/Tsub_Max_Gradient_Method_h.nc" # Update if needed

observed_temp_ds_argo = xr.open_dataset(observed_path, decode_times=False)
a = xr.open_dataset(T_SUB_GRAD_PATH, decode_times=False)
print(a)

