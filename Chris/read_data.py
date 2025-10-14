import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

TEMP_DATA_PATH = "RG_ArgoClim_Temperature_2019.nc"

dataset = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
print(dataset)

anomaly = dataset.variables['ARGO_TEMPERATURE_ANOMALY'][:]
surface_anomaly_0104 = dataset['ARGO_TEMPERATURE_ANOMALY'].sel(TIME=0.5).isel(PRESSURE=0)
print(surface_anomaly_0104)

print("plotting")
plt.figure()
plt.imshow(surface_anomaly_0104, cmap='viridis')
plt.colorbar(label='Kelvin')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
