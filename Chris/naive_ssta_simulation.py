import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import esmpy as ESMF
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "../datasets/heat_flux_interpolated_all_contributions.nc"
HEAT_FLUX_DATA_PATH = "../datasets/heat_flux_interpolated.nc"
EKMAN_ANOMALY_DATA_PATH = "../datasets/Ekman_Current_Anomaly.nc"
TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
USE_ALL_CONTRIBUTIONS = True
USE_EKMAN_TERM = True

ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)

print(ekman_anomaly_ds)

if USE_ALL_CONTRIBUTIONS:
    heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_ishf']
else:
    heat_flux_ds = xr.open_dataset(HEAT_FLUX_DATA_PATH, decode_times=True)
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['slhf'] + heat_flux_ds['sshf']


temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
#print(temperature_ds)
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)

if USE_EKMAN_TERM:
    heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'] = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'] + ekman_anomaly_ds['Q_Ek_anom']

#print(heat_flux_anomaly_ds)
mld_ds = xr.open_dataset(MLD_DATA_PATH, decode_times=False)
#print(mld_ds)
# ds = xr.concat([heat_flux_anomaly_ds, mld_ds], 'TIME')
# print(ds)

model_anomalies = []
added_baseline = False
for month in heat_flux_anomaly_ds.TIME.values:
    if not added_baseline:
        base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY'] - temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
        base = base.expand_dims(TIME=[month])
        model_anomalies.append(base)
        added_baseline = True
    else:
        prev = model_anomalies[-1].isel(TIME=-1)
        cur = prev + (30.4375 * 24 * 60 * 60) * heat_flux_anomaly_ds.sel(TIME=month)['NET_HEAT_FLUX_ANOMALY'] / (mld_ds.sel(TIME=month)['MLD_PRESSURE'] * 1025 * 4100)
        cur = cur.expand_dims(TIME=[month])
        model_anomalies.append(cur)
model_anomaly_ds = xr.concat(model_anomalies, 'TIME')
model_anomaly_ds = model_anomaly_ds.drop_vars(["PRESSURE"])
print(model_anomaly_ds)

#model_anomaly_ds.sel(TIME=100.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r')
#plt.show()

# make a movie
times = model_anomaly_ds.TIME.values

fig, ax = plt.subplots()
pcolormesh = ax.pcolormesh(model_anomaly_ds.LONGITUDE.values, model_anomaly_ds.LATITUDE.values, model_anomaly_ds.isel(TIME=0), cmap='RdBu_r')
title = ax.set_title(f'Time = {times[0]}')

cbar = plt.colorbar(pcolormesh, ax=ax, label='Modelled anomaly from surface heat flux')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')


def update(frame):
    pcolormesh.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
    #pcolormesh.set_clim(vmin=float(model_anomaly_ds.isel(TIME=frame).min()), vmax=float(model_anomaly_ds.isel(TIME=frame).max()))
    pcolormesh.set_clim(vmin=-20, vmax=20)
    cbar.update_normal(pcolormesh)
    title.set_text(f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}')
    return [pcolormesh, title]

animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
plt.show()
