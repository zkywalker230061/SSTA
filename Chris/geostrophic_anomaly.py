import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SSTA.Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, coriolis_parameter
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, compute_gradient_lon, compute_gradient_lat
import matplotlib

DOWNLOADED_SSH = False

TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
MEAN_TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Temperature_Monthly_Mean.nc"
if DOWNLOADED_SSH:
    SEA_SURFACE_HEIGHT_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc"
    var_name = "sla"
else:
    SEA_SURFACE_HEIGHT_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"
    var_name = "ssh"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
rho_0 = 1025.0
c_0 = 4100.0
g = 9.81

mean_temp = xr.open_dataset(MEAN_TEMP_DATA_PATH, decode_times=False)
ssh_anomaly = xr.open_dataset(SEA_SURFACE_HEIGHT_DATA_PATH, decode_times=False)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD_PRESSURE"]

temp_grad_lat = compute_gradient_lat(mean_temp["MONTHLY_MEAN_TEMPERATURE"]).sel(PRESSURE=2.5)   # really should integrate over mixed layer, but this should be close enough
temp_grad_long = compute_gradient_lon(mean_temp["MONTHLY_MEAN_TEMPERATURE"]).sel(PRESSURE=2.5)
ssh_grad_anomaly_lat = ssh_anomaly[var_name + '_anomaly_grad_lat']
ssh_grad_anomaly_long = ssh_anomaly[var_name + '_anomaly_grad_long']

f = coriolis_parameter(ssh_anomaly['LATITUDE']).broadcast_like(ssh_anomaly[var_name]).broadcast_like(ssh_grad_anomaly_long)    # broadcasting based on Jason/Julia's usage


def get_geostrophic_term(time, hbar, f, ssh_lat, temp_long, ssh_long, temp_lat):
    month = int((time + 0.5) % 12)
    if month == 0:
        month = 12
    return (rho_0 * c_0 * hbar.sel(MONTH=month) * g / f.sel(TIME=time)) * (ssh_lat.sel(TIME=time) * temp_long.sel(MONTH=month) - ssh_long.sel(TIME=time) * temp_lat.sel(MONTH=month))

geostropic_anomalies = []
for time in ssh_grad_anomaly_lat.TIME.values:
    geostrophic_term = get_geostrophic_term(time, hbar_da, f, ssh_grad_anomaly_lat, temp_grad_long, ssh_grad_anomaly_long, temp_grad_lat)
    geostrophic_term = geostrophic_term.reset_coords(("PRESSURE", "MONTH"), drop=True)
    geostropic_anomalies.append(geostrophic_term)
geostrophic_anomaly = xr.concat(geostropic_anomalies, "TIME")
geostrophic_anomaly = geostrophic_anomaly.rename("GEOSTROPHIC_ANOMALY")

geostrophic_anomaly = geostrophic_anomaly.where((geostrophic_anomaly['LATITUDE'] > 5) | (geostrophic_anomaly['LATITUDE'] < -5), 0)
if DOWNLOADED_SSH:
    geostrophic_anomaly.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/geostrophic_anomaly_downloaded.nc")
else:
    geostrophic_anomaly.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/geostrophic_anomaly_calculated.nc")
