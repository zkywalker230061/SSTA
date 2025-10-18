import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"

dataset = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
print(dataset)

# lon_atrib = dataset.coords['LONGITUDE'].attrs
# dataset['LONGITUDE'] = ((dataset['LONGITUDE'] + 180) % 360) - 180
# dataset = dataset.sortby(dataset['LONGITUDE'])
# dataset['LONGITUDE'].attrs.update(lon_atrib)
# dataset['LONGITUDE'].attrs['modulo'] = 180

anomaly = dataset.sel(TIME=0.5).variables['ARGO_TEMPERATURE_ANOMALY'][:].isel(PRESSURE=0)
surface_anomaly_0104 = dataset['ARGO_TEMPERATURE_ANOMALY'].sel(TIME=0.5).isel(PRESSURE=0)
print("PLOTTING this")
print(surface_anomaly_0104)
print("end")


surface_anomaly_0104_lat = surface_anomaly_0104.coords['LATITUDE'][:]
surface_anomaly_0104_long = surface_anomaly_0104.coords['LONGITUDE'][:]


print(surface_anomaly_0104)

print(dataset.variables['LONGITUDE'])
print(dataset.variables['LATITUDE'])


print("plotting")
print(type(surface_anomaly_0104))

surface_anomaly_0104_plot = xr.DataArray(anomaly, coords=[surface_anomaly_0104_lat, surface_anomaly_0104_long], dims=['lat', 'lon'])
print(anomaly)
print(surface_anomaly_0104_lat)

plt.figure()
ax = plt.gca()
#ax.set_xlim([-180, 180])
#ax.set_ylim([-90, 90])
#surface_anomaly_0104.plot.imshow(x='LONGITUDE', y='LATITUDE', cmap='viridis')
plt.imshow(surface_anomaly_0104_plot, cmap='viridis')
plt.colorbar(label='Kelvin')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
