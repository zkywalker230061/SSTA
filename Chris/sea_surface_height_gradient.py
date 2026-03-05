import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from SSTA.Chris.utils import make_movie, load_and_prepare_dataset, compute_gradient_lat, compute_gradient_lon, \
    get_monthly_mean, get_anomaly, coriolis_parameter

DOWNLOADED = False

if DOWNLOADED:
    SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated.nc"
    var_name = "sla"
else:
    SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated.nc"
    var_name = "ssh"
sea_surface_ds = xr.open_dataset(SEA_SURFACE_DATA_PATH, decode_times=False)
print(sea_surface_ds)
temp_ds = load_and_prepare_dataset("/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc")

g = 9.81

sea_surface_ds[var_name] = sea_surface_ds[var_name] * g     # mistaken divide by g in calculate

monthly_mean_sla = get_monthly_mean(sea_surface_ds[var_name])
sea_surface_ds[var_name + '_ANOMALY'] = get_anomaly(sea_surface_ds, var_name, monthly_mean_sla)[var_name + "_ANOMALY"]
sea_surface_ds[var_name + '_ANOMALY'].attrs['units'] = ''

sea_surface_ds[var_name + '_anomaly_grad_lat'] = compute_gradient_lat(sea_surface_ds[var_name + '_ANOMALY'])
sea_surface_ds[var_name + '_anomaly_grad_long'] = compute_gradient_lon(sea_surface_ds[var_name + '_ANOMALY'])
print(abs(sea_surface_ds[var_name + '_anomaly_grad_lat']).mean().item())
print(abs(sea_surface_ds[var_name + '_anomaly_grad_long']).mean().item())

ssh_monthlymean_grad_lat = compute_gradient_lat(monthly_mean_sla)
ssh_monthlymean_grad_long = compute_gradient_lon(monthly_mean_sla)
monthly_mean_ssh_ds = xr.Dataset({var_name + "_monthlymean": monthly_mean_sla, var_name + '_monthlymean_grad_lat': ssh_monthlymean_grad_lat, var_name + '_monthlymean_grad_long': ssh_monthlymean_grad_long})
monthly_mean_ssh_ds[var_name + "_monthlymean"].attrs["units"] = "dbar"
monthly_mean_ssh_ds[var_name + '_monthlymean_grad_lat'].attrs["units"] = ""
monthly_mean_ssh_ds[var_name + '_monthlymean_grad_long'].attrs["units"] = ""

g = 9.81
f = coriolis_parameter(sea_surface_ds['LATITUDE']).broadcast_like(sea_surface_ds[var_name]).broadcast_like(sea_surface_ds[var_name + '_anomaly_grad_long']).sel(TIME=0.5)  # broadcasting based on Jason/Julia's usage; take any time because they're always the same
alpha = g / f * monthly_mean_ssh_ds[var_name + '_monthlymean_grad_lat']
beta = g / f * monthly_mean_ssh_ds[var_name + '_monthlymean_grad_long']
alpha_grad_lon = compute_gradient_lon(alpha)
beta_grad_lat = compute_gradient_lat(beta)

tropics_mask = (monthly_mean_ssh_ds.LATITUDE >= -5) & (monthly_mean_ssh_ds.LATITUDE <= 5)
alpha = xr.where(tropics_mask, 0, alpha)
beta = xr.where(tropics_mask, 0, beta)
alpha_grad_lon = xr.where(tropics_mask, 0, alpha_grad_lon)
beta_grad_lat = xr.where(tropics_mask, 0, beta_grad_lat)

monthly_mean_ssh_ds["alpha"] = alpha
monthly_mean_ssh_ds["alpha_grad_long"] = alpha_grad_lon
monthly_mean_ssh_ds["beta"] = beta
monthly_mean_ssh_ds["beta_grad_lat"] = beta_grad_lat



# make_movie(sea_surface_ds[var_name + '_anomaly_grad_lat'], -2e-6, 2e-6)
# make_movie(sea_surface_ds[var_name + '_anomaly_grad_long'], -2e-6, 2e-6)

# print(sea_surface_ds)

if DOWNLOADED:
    sea_surface_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc")
    monthly_mean_ssh_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_monthly_mean_interpolated_grad.nc")
else:
    sea_surface_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc")
    #sea_surface_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_specific_volume_method_grad.nc")
    monthly_mean_ssh_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_monthly_mean_calculated_grad.nc")
