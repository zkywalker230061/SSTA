import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from SSTA.Chris.utils import make_movie, load_and_prepare_dataset, compute_gradient_lat, compute_gradient_lon, \
    get_monthly_mean, get_anomaly

SEA_SURFACE_DOWNLOAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated.nc"

SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated.nc"
#SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_specific_volume_method.nc"

SEA_SURFACE_GRAD_DOWNLOAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc"
SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"
#SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_specific_volume_method_grad.nc"

sea_surface_download_ds = xr.open_dataset(SEA_SURFACE_DOWNLOAD_DATA_PATH, decode_times=False)
sea_surface_ds = xr.open_dataset(SEA_SURFACE_DATA_PATH, decode_times=False)
sea_surface_grad_download_ds = xr.open_dataset(SEA_SURFACE_GRAD_DOWNLOAD_DATA_PATH, decode_times=False)
sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)

ssh_download_da = sea_surface_download_ds['sla']
ssh_da = sea_surface_ds['ssh']
#ssh_da = (ssh_da - 2000)    # this "ssh" is the height of the sea surface above the point of 2000 metres depth (the reference point)
ssh_grad_long_download_da = sea_surface_grad_download_ds['sla_anomaly_grad_long']
ssh_grad_lat_download_da = sea_surface_grad_download_ds['sla_anomaly_grad_lat']
ssh_grad_long_da = sea_surface_grad_ds['ssh_anomaly_grad_long']
ssh_grad_lat_da = sea_surface_grad_ds['ssh_anomaly_grad_lat']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ssh_da.mean('TIME').plot(ax=axes[0], vmin=-0.5, vmax=0.5, cmap='RdBu_r')
axes[0].set_title('Calculated SSH')
ssh_download_da.mean('TIME').plot(ax=axes[1], vmin=-0.5, vmax=0.5, cmap='RdBu_r')
axes[1].set_title('Downloaded SSH')
plt.show()

# print(sea_surface_ds)
# print((ssh_download_da).mean().item())
# print((ssh_da).mean().item())
#
# make_movie(ssh_download_da, -0.5, 0.5)#, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_download.mp4")
# make_movie(ssh_da, -0.5, 0.5)#, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_calculated.mp4")
#
# make_movie(ssh_grad_long_download_da, -1e-6, 1e-6)#, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_grad_long_download.mp4")
# make_movie(ssh_grad_long_da, -1e-6, 1e-6)#, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_grad_long_calculated.mp4")
#
# make_movie(ssh_grad_lat_download_da, -1e-6, 1e-6)#, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_grad_lat_download.mp4")
# make_movie(ssh_grad_lat_da, -1e-6, 1e-6)#, savepath="/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_grad_lat_calculated.mp4")
#
#
# xr.corr(ssh_download_da, ssh_da, dim='TIME').plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# #plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_correlation.jpg")
# plt.show()
#
# xr.corr(ssh_grad_long_download_da, ssh_grad_long_da, dim='TIME').plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# #plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_grad_long_correlation.jpg")
# plt.show()
#
# xr.corr(ssh_grad_lat_download_da, ssh_grad_lat_da, dim='TIME').plot(x='LONGITUDE', y='LATITUDE', cmap='nipy_spectral', vmin=-1, vmax=1)
# #plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/ssh_videos/ssh_grad_lat_correlation.jpg")
# plt.show()




