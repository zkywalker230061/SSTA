import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

T_SUB_DATA_PATH = "../datasets/t_sub.nc"

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
# t_sub_ds['T_sub'] = t_sub_ds['SUB_TEMPERATURE']
# t_sub_monthly_mean = get_monthly_mean(t_sub_ds['T_sub'])
# t_sub_ds = get_anomaly(t_sub_ds, 'T_sub', t_sub_monthly_mean)
# t_sub_ds.to_netcdf("../datasets/t_sub_2.nc")

print(t_sub_ds["T_sub_ANOMALY"].max().item())
print(t_sub_ds["T_sub_ANOMALY"].min().item())
print(abs(t_sub_ds["T_sub_ANOMALY"]).mean().item())



vmin_anom = -1
vmax_anom = 1

t_sub_ds['T_sub'].sel(TIME=132.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=0, vmax=30)
plt.show()

t_sub_ds['T_sub'].sel(TIME=138.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=0, vmax=30)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=0.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=6.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=60.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=66.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=132.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=135.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=138.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=141.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=168.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=171.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=174.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()

t_sub_ds['T_sub_ANOMALY'].sel(TIME=177.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=vmin_anom, vmax=vmax_anom)
plt.show()
