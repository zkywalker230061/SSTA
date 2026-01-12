import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from SSTA.Chris.utils import make_movie, load_and_prepare_dataset, compute_gradient_lat, compute_gradient_lon, \
    get_monthly_mean, get_anomaly

SEA_SURFACE_DOWNLOAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated.nc"
SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated.nc"
sea_surface_download_ds = xr.open_dataset(SEA_SURFACE_DOWNLOAD_DATA_PATH, decode_times=False)
sea_surface_ds = xr.open_dataset(SEA_SURFACE_DATA_PATH, decode_times=False)

ssh_download_da = sea_surface_download_ds['sla']
ssh_da = sea_surface_ds['ssh']
ssh_da = (ssh_da - 9810) / 9.81

print(abs(ssh_download_da).mean().item())
print(abs(ssh_da).mean().item())

make_movie(ssh_download_da, -0.2, 0.2)
make_movie(ssh_da - 9810, -0.25, 0.25)


