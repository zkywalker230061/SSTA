import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from SSTA.Chris.utils import make_movie, load_and_prepare_dataset, compute_gradient_lat, compute_gradient_lon, \
    get_monthly_mean, get_anomaly

DOWNLOADED = False

if DOWNLOADED:
    SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated.nc"
    var_name = "sla"
else:
    SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated.nc"
    var_name = "ssh"
sea_surface_ds = xr.open_dataset(SEA_SURFACE_DATA_PATH, decode_times=False)
temp_ds = load_and_prepare_dataset("/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc")

monthly_mean_sla = get_monthly_mean(sea_surface_ds[var_name])
sea_surface_ds[var_name + '_ANOMALY'] = get_anomaly(sea_surface_ds, var_name, monthly_mean_sla)[var_name + "_ANOMALY"]
sea_surface_ds[var_name + '_ANOMALY'].attrs['units'] = ''

sea_surface_ds[var_name + '_anomaly_grad_lat'] = compute_gradient_lat(sea_surface_ds[var_name + '_ANOMALY'])
sea_surface_ds[var_name + '_anomaly_grad_long'] = compute_gradient_lon(sea_surface_ds[var_name + '_ANOMALY'])
print(abs(sea_surface_ds[var_name + '_anomaly_grad_lat']).mean().item())
print(abs(sea_surface_ds[var_name + '_anomaly_grad_long']).mean().item())

make_movie(sea_surface_ds[var_name + '_anomaly_grad_lat'], -2e-6, 2e-6)
make_movie(sea_surface_ds[var_name + '_anomaly_grad_long'], -2e-6, 2e-6)

print(sea_surface_ds)

if DOWNLOADED:
    sea_surface_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc")
else:
    sea_surface_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc")
