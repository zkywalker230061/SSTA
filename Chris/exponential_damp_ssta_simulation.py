import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
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
USE_SURFACE_FLUX = True
USE_EKMAN_TERM = True
USE_ENTRAINMENT = True     # something broken with entrainment; if False, use gamma_0 for damping
INTEGRATE_EXPLICIT = False
SEPARATE_REGIMES = False        # if True, then consider entrainment only when it is above entrainment_lower_threshold
rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 10.0

if USE_ALL_CONTRIBUTIONS:
    heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH, decode_times=False)
    print(heat_flux_ds)
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds[
        'avg_snswrf'] + heat_flux_ds['avg_ishf']
else:
    heat_flux_ds = xr.open_dataset(HEAT_FLUX_DATA_PATH, decode_times=True)
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['slhf'] + heat_flux_ds['sshf']

if not USE_SURFACE_FLUX:        # yes, the method is poor but it works!
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['NET_HEAT_FLUX'] - heat_flux_ds['NET_HEAT_FLUX']

temperature_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
surface_flux_ds = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

if USE_EKMAN_TERM:      # it's bad naming but nevertheless the case that heat_flux_anom contains surface flux and Ekman
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'] = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY'] + ekman_anomaly_ds[
        'Q_Ek_anom']
    ekman_flux_ds = ekman_anomaly_ds['Q_Ek_anom']

mld_ds = xr.open_dataset(MLD_DATA_PATH, decode_times=False)

t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['ENTRAINMENT_VELOCITY'])

entrainment_lower_threshold = 0.5 * np.nanmean(np.abs(entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']))

def month_to_second(month):
    return month * 30.4375 * 24 * 60 * 60


model_anomalies = []
entrainment_fluxes = []
if INTEGRATE_EXPLICIT:
    time = 30.4375 * 24 * 60 * 60 * 0.5
    for month in heat_flux_anomaly_ds.TIME.values:
        month_in_year = int((month + 0.5) % 12)
        if month_in_year == 0:
            month_in_year = 12
        hbar = hbar_ds.sel(MONTH=month_in_year)['MONTHLY_MEAN_MLD_PRESSURE']
        entrainment_vel = entrainment_vel_ds.sel(MONTH=month_in_year)['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']
        Tm_anomaly = (t_sub_ds.sel(TIME=month)['T_sub_ANOMALY'] + heat_flux_anomaly_ds.sel(TIME=month)[
            'NET_HEAT_FLUX_ANOMALY'] / (entrainment_vel * rho_0 * c_0)) * (
                                 1 - np.exp(-1 * entrainment_vel * time / hbar))
        Tm_anomaly = Tm_anomaly.expand_dims(TIME=[month])
        model_anomalies.append(Tm_anomaly)
        time += 30.4375 * 24 * 60 * 60
    model_anomaly_ds = xr.concat(model_anomalies, 'TIME')
    model_anomaly_ds.to_netcdf("../datasets/model_anomaly_exponential_damping_explicit.nc")
else:
    added_baseline = False
    for month in heat_flux_anomaly_ds.TIME.values:
        # find the previous and current month from 1 to 12 to access the monthly-averaged data (hbar, entrainment vel.)
        prev_month = month - 1
        month_in_year = int((month + 0.5) % 12)
        if month_in_year == 0:
            month_in_year = 12
        prev_month_in_year = month_in_year - 1
        if prev_month_in_year == 0:
            prev_month_in_year = 12

        if not added_baseline:  # just adds the baseline of a whole bunch of zero
            base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY'] - \
                   temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
            base = base.expand_dims(TIME=[month])
            model_anomalies.append(base)
            added_baseline = True
        else:
            prev_tm_anom = model_anomalies[-1].isel(TIME=-1)
            prev_tsub_anom = t_sub_ds.sel(TIME=prev_month)['T_sub_ANOMALY']
            prev_tm_anom = xr.where(np.isfinite(prev_tm_anom), prev_tm_anom, 0)     # reset NaN to 0; I'd rather this reset to the 'last useful value', but can't figure out how to do that now
            prev_heat_flux_anom = heat_flux_anomaly_ds.sel(TIME=prev_month)['NET_HEAT_FLUX_ANOMALY']
            prev_entrainment_vel = entrainment_vel_ds.sel(MONTH=prev_month_in_year)['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']
            #prev_entrainment_vel = entrainment_vel_ds.sel(TIME=prev_month)['ENTRAINMENT_VELOCITY']
            prev_hbar = hbar_ds.sel(MONTH=prev_month_in_year)['MONTHLY_MEAN_MLD_PRESSURE']

            cur_tsub_anom = t_sub_ds.sel(TIME=month)['T_sub_ANOMALY']
            cur_heat_flux_anom = heat_flux_anomaly_ds.sel(TIME=month)['NET_HEAT_FLUX_ANOMALY']
            cur_entrainment_vel = entrainment_vel_ds.sel(MONTH=month_in_year)['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']
            #cur_entrainment_vel = entrainment_vel_ds.sel(TIME=month)['ENTRAINMENT_VELOCITY']
            cur_hbar = hbar_ds.sel(MONTH=month_in_year)['MONTHLY_MEAN_MLD_PRESSURE']
            if not USE_ENTRAINMENT:         # ignore entrainment altogether
                cur_tm_anom = cur_heat_flux_anom / gamma_0 + (prev_tm_anom - prev_heat_flux_anom / gamma_0) * np.exp(gamma_0 / (rho_0 * c_0 * prev_hbar) * month_to_second(prev_month) - gamma_0 / (rho_0 * c_0 * cur_hbar) * month_to_second(month))
            else:
                if not SEPARATE_REGIMES:
                    cur_k = (gamma_0 / (rho_0 * c_0) + cur_entrainment_vel) / cur_hbar
                    prev_k = (gamma_0 / (rho_0 * c_0) + prev_entrainment_vel) / prev_hbar
                    exponent = prev_k * month_to_second(prev_month) - prev_k * month_to_second(month)
                    # exponent = -0.5 * (prev_k + cur_k) * month_to_second(1)
                    # exponent = exponent.where(exponent <= 0, 0)
                    cur_tm_anom = (cur_entrainment_vel / (cur_k * cur_hbar)) * cur_tsub_anom + cur_heat_flux_anom / (cur_k * rho_0 * c_0 * cur_hbar) + (prev_tm_anom - (prev_entrainment_vel / (prev_k * prev_hbar)) * prev_tsub_anom - prev_heat_flux_anom / (prev_k * rho_0 * c_0 * prev_hbar)) * np.exp(exponent)

                else:       # if treating entrainment, we have to be careful not to divide by zero
                    no_entrainment_mask = np.abs(cur_entrainment_vel) <= entrainment_lower_threshold     # mask for when entrainment does not apply
                    entrainment_mask = np.abs(cur_entrainment_vel) > entrainment_lower_threshold     # opposite to no_entrainment_mask

                    no_entrainment_mask = xr.DataArray(no_entrainment_mask, coords=cur_entrainment_vel.coords, dims=cur_entrainment_vel.dims)
                    entrainment_mask = xr.DataArray(entrainment_mask, coords=cur_entrainment_vel.coords, dims=cur_entrainment_vel.dims)

                    cur_tm_anom = np.full_like(cur_tsub_anom, np.nan, dtype=float)  # initialise cur_tm
                    if np.any(entrainment_mask):
                        cur_entrainment_vel_entrainment_mask = np.where(entrainment_mask, cur_entrainment_vel, np.nan)
                        prev_entrainment_vel_entrainment_mask = np.where(entrainment_mask, prev_entrainment_vel, np.nan)
                        cur_hbar_entrainment_mask = np.where(entrainment_mask, cur_hbar, np.nan)
                        prev_hbar_entrainment_mask = np.where(entrainment_mask, prev_hbar, np.nan)

                        exponent = (prev_entrainment_vel_entrainment_mask / prev_hbar_entrainment_mask * month_to_second(prev_month) - cur_entrainment_vel_entrainment_mask / cur_hbar_entrainment_mask * month_to_second(month))
                        exponent = np.clip(exponent, -600, 600)     # prevent absurd values
                        exponent = np.exp(exponent)

                        cur_tm_anom_entrain = cur_tsub_anom + cur_heat_flux_anom / (cur_entrainment_vel_entrainment_mask * rho_0 * c_0) + (prev_tm_anom - prev_tsub_anom - prev_heat_flux_anom / (prev_entrainment_vel_entrainment_mask * rho_0 * c_0)) * exponent
                    else:
                        cur_tm_anom_entrain = np.full_like(cur_tsub_anom, np.nan, dtype=float)
                    if np.any(no_entrainment_mask):     # ignore entrainment in the same way as previously
                        cur_hbar_no_entrainment_mask = np.where(no_entrainment_mask, cur_hbar, np.nan)
                        prev_hbar_no_entrainment_mask = np.where(no_entrainment_mask, prev_hbar, np.nan)

                        exponent_no_entrainment = gamma_0 / (rho_0 * c_0 * prev_hbar_no_entrainment_mask) * month_to_second(prev_month) - gamma_0 / (rho_0 * c_0 * cur_hbar_no_entrainment_mask) * month_to_second(month)
                        exponent_no_entrainment = np.exp(exponent_no_entrainment)
                        cur_tm_anom_no_entrain = cur_heat_flux_anom / gamma_0 + (prev_tm_anom - prev_heat_flux_anom / gamma_0) * exponent_no_entrainment
                    else:
                        cur_tm_anom_no_entrain = np.full_like(cur_tsub_anom, np.nan, dtype=float)
                    cur_tm_anom = xr.where(no_entrainment_mask, cur_tm_anom_no_entrain, cur_tm_anom_entrain)
            cur_tm_anom = cur_tm_anom.drop_vars('MONTH', errors='ignore')
            cur_tm_anom = cur_tm_anom.expand_dims(TIME=[month])
            model_anomalies.append(cur_tm_anom)
            entrainment_flux = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_tm_anom)
            entrainment_fluxes.append(entrainment_flux)
    model_anomaly_ds = xr.concat(model_anomalies, 'TIME')
    model_anomaly_ds.to_netcdf("../datasets/model_anomaly_exponential_damping_implicit.nc")
    # entrainment_flux_ds = xr.concat(entrainment_fluxes, 'TIME')
    # entrainment_flux_ds = entrainment_flux_ds.drop_vars(["MONTH", "PRESSURE"])
    # entrainment_flux_ds = entrainment_flux_ds.transpose("TIME", "LATITUDE", "LONGITUDE")
    # surface_flux_ds = surface_flux_ds.rename("SURFACE_FLUX_ANOMALY")
    # ekman_flux_ds = ekman_flux_ds.rename("EKMAN_FLUX_ANOMALY")
    # entrainment_flux_ds = entrainment_flux_ds.rename("ENTRAINMENT_FLUX_ANOMALY")
    # flux_components_ds = xr.merge([surface_flux_ds, ekman_flux_ds, entrainment_flux_ds])
    # flux_components_ds.to_netcdf("../datasets/flux_components.nc")


# make a movie
times = model_anomaly_ds.TIME.values

fig, ax = plt.subplots()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
pcolormesh = ax.pcolormesh(model_anomaly_ds.LONGITUDE.values, model_anomaly_ds.LATITUDE.values,
                           model_anomaly_ds.isel(TIME=0), cmap='RdBu_r')
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
