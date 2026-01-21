import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from SSTA.Chris.utils import make_movie, load_and_prepare_dataset, compute_gradient_lat, compute_gradient_lon, \
    get_monthly_mean, get_anomaly

SEA_SURFACE_DOWNLOAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated.nc"
SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated.nc"
SEA_SURFACE_GRAD_DOWNLOAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"

sea_surface_download_ds = xr.open_dataset(SEA_SURFACE_DOWNLOAD_DATA_PATH, decode_times=False)
sea_surface_ds = xr.open_dataset(SEA_SURFACE_DATA_PATH, decode_times=False)
sea_surface_grad_download_ds = xr.open_dataset(SEA_SURFACE_GRAD_DOWNLOAD_DATA_PATH, decode_times=False)
sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)

ssh_download_da = sea_surface_download_ds['sla']
ssh_da = sea_surface_ds['ssh']
ssh_da = (ssh_da - 2000)    # this "ssh" is the height of the sea surface above the point of 2000 metres depth (the reference point)
ssh_grad_long_download_da = sea_surface_grad_download_ds['sla_anomaly_grad_long']
ssh_grad_lat_download_da = sea_surface_grad_download_ds['sla_anomaly_grad_lat']
ssh_grad_long_da = sea_surface_grad_ds['ssh_anomaly_grad_long']
ssh_grad_lat_da = sea_surface_grad_ds['ssh_anomaly_grad_lat']


print(abs(ssh_download_da).mean().item())
print(abs(ssh_da).mean().item())

# make_movie(ssh_download_da, -0.1, 0.1)
# make_movie(ssh_da, -0.1, 0.1)

xr.corr(ssh_download_da, ssh_da, dim='TIME').plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()

xr.corr(ssh_grad_long_download_da, ssh_grad_long_da, dim='TIME').plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()

xr.corr(ssh_grad_lat_download_da, ssh_grad_lat_da, dim='TIME').plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()




