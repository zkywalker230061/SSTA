import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar

from SSTA.Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, compute_gradient_lon, compute_gradient_lat
import matplotlib

TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
MEAN_TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Temperature_Monthly_Mean.nc"
SEA_SURFACE_HEIGHT_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
rho_0 = 1025.0
c_0 = 4100.0
g = 9.81
omega = 2 * np.pi / (24 * 3600)


def coriolis_parameter(lat):
    phi_rad = np.deg2rad(lat)
    f = 2 * omega * np.sin(phi_rad)
    f = xr.DataArray(f, coords={'LATITUDE': lat}, dims=['LATITUDE'])
    f.attrs['units'] = 's^-1'
    return f

mean_temp = xr.open_dataset(MEAN_TEMP_DATA_PATH, decode_times=False)
ssh_anomaly = xr.open_dataset(SEA_SURFACE_HEIGHT_DATA_PATH, decode_times=False)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD_PRESSURE"]

temp_grad_lat = compute_gradient_lat(mean_temp["MONTHLY_MEAN_TEMPERATURE"])
temp_grad_long = compute_gradient_lon(mean_temp["MONTHLY_MEAN_TEMPERATURE"])
ssh_grad_anomaly_lat = ssh_anomaly['sla_anomaly_grad_lat']
ssh_grad_anomaly_long = ssh_anomaly['sla_anomaly_grad_long']

#lat = temp_grad_lat["LATITUDE"]
f = coriolis_parameter(ssh_anomaly['LATITUDE']).broadcast_like(ssh_anomaly['sla']).broadcast_like(ssh_grad_anomaly_long)

# chunk into Dask arrays for efficiency
time_chunk = -1
lat_chunk = 40
long_chunk = 40
hbar_da = hbar_da.chunk({"MONTH": -1, "LATITUDE": lat_chunk, "LONGITUDE": long_chunk})
f = f.chunk({"TIME": -1, "LATITUDE": lat_chunk, "LONGITUDE": long_chunk})
ssh_grad_anomaly_lat = ssh_grad_anomaly_lat.chunk({"TIME": -1, "LATITUDE": lat_chunk, "LONGITUDE": long_chunk})
temp_grad_long = temp_grad_long.chunk({"MONTH": -1, "LATITUDE": lat_chunk, "LONGITUDE": long_chunk})
ssh_grad_anomaly_long = ssh_grad_anomaly_long.chunk({"TIME": -1, "LATITUDE": lat_chunk, "LONGITUDE": long_chunk})
temp_grad_lat = temp_grad_lat.chunk({"MONTH": -1, "LATITUDE": lat_chunk, "LONGITUDE": long_chunk})


def get_geostrophic_term(time, hbar, f, ssh_lat, temp_lon, ssh_lon, temp_lat):
    month = int((time + 0.5) % 12)
    if month == 0:
        month = 12
    return (rho_0 * c_0 * hbar * g / f.sel(TIME=time)) * (ssh_lat.sel(TIME=time) * temp_lon.sel(MONTH=month) - ssh_lon.sel(TIME=time) * temp_lat.sel(MONTH=month))
# seems to return something with a month dimension as well, which isn't helpful
geostropic_anomalies = []
for time in ssh_grad_anomaly_lat.TIME.values:
    geostropic_anomalies.append(get_geostrophic_term(time, hbar_da, f, ssh_grad_anomaly_lat, temp_grad_long, ssh_grad_anomaly_long, temp_grad_lat))
geostrophic_anomaly = xr.concat(geostropic_anomalies, "TIME")

print(geostrophic_anomaly)
with ProgressBar():
    geostrophic_anomaly.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/geostrophic_anomaly.nc")
