import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import esmpy as ESMF
import cartopy.crs as ccrs

from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "../datasets/heat_flux_interpolated_all_contributions.nc"
HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = "../datasets/Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
ENTRAINMENT_VEL_DATA_PATH = "../datasets/Entrainment_Velocity-(2004-2018).nc"
H_BAR_DATA_PATH = "../datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = "../datasets/t_sub.nc"
USE_ALL_CONTRIBUTIONS = True
USE_EKMAN_TERM = True

if USE_ALL_CONTRIBUTIONS:
    heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_ishf']
else:
    heat_flux_ds = xr.open_dataset(HEAT_FLUX_DATA_PATH, decode_times=True)
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['slhf'] + heat_flux_ds['sshf']


temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)

if USE_EKMAN_TERM:
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'] = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'] + ekman_anomaly_ds['Q_Ek_anom']

mld_ds = xr.open_dataset(MLD_DATA_PATH, decode_times=False)

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])

model_anomalies = []
time = 30.4375 * 24 * 60 * 60 * 0.5
for month in heat_flux_anomaly_ds.TIME.values:
    month_in_year = int((month + 0.5) % 12)
    if month_in_year == 0:
        month_in_year = 12
    hbar = hbar_ds.sel(MONTH=month_in_year)['MONTHLY_MEAN_MLD_PRESSURE']
    entrainment_vel = entrainment_vel_ds.sel(MONTH=month_in_year)['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']
    Tm_anomaly = (t_sub_ds.sel(TIME=month)['T_sub_ANOMALY'] + heat_flux_anomaly_ds.sel(TIME=month)['NET_HEAT_FLUX_ANOMALY'] / (entrainment_vel * 1025 * 4100)) * (1-np.exp(-1 * entrainment_vel * time / hbar))
    Tm_anomaly = Tm_anomaly.expand_dims(TIME=[month])
    model_anomalies.append(Tm_anomaly)
    time += 30.4375 * 24 * 60 * 60
model_anomaly_ds = xr.concat(model_anomalies, 'TIME')
#print(model_anomaly_ds)
model_anomaly_ds.to_netcdf("../datasets/model_anomaly_exponential_damping.nc")
print(abs(model_anomaly_ds).mean().item())
#print(model_anomaly_ds.min().item())

# make a movie
times = model_anomaly_ds.TIME.values

fig, ax = plt.subplots()
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.coastlines()
pcolormesh = ax.pcolormesh(model_anomaly_ds.LONGITUDE.values, model_anomaly_ds.LATITUDE.values, model_anomaly_ds.isel(TIME=0), cmap='RdBu_r')
title = ax.set_title(f'Time = {times[0]}')

cbar = plt.colorbar(pcolormesh, ax=ax, label='Modelled anomaly from surface heat flux')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')


def update(frame):
    month = int((times[frame] + 0.5) % 12)
    if month == 0:
        month = 12
    year = 2004 + int((times[frame]) / 12)
    pcolormesh.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
    #pcolormesh.set_clim(vmin=float(model_anomaly_ds.isel(TIME=frame).min()), vmax=float(model_anomaly_ds.isel(TIME=frame).max()))
    pcolormesh.set_clim(vmin=-10, vmax=10)
    cbar.update_normal(pcolormesh)
    title.set_text(f'Year: {year}; Month: {month}')
    return [pcolormesh, title]

animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
plt.show()
