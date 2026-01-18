"""
Simulate Sea Surface Temperature Anomalies (SSTA).

Chengyun Zhu
2026-1-18
"""

import xarray as xr

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file

SURFACE = True
ENTRAINMENT = True
EKMAN = True
GEOSTROPHIC = True

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month

if SURFACE:
    q_surface_ds = load_and_prepare_dataset(
        "datasets/Simulation-Surface_Heat_Flux-(2004-2018).nc"
    )
    q_surface = (
        q_surface_ds['ANOMALY_avg_slhtf']
        + q_surface_ds['ANOMALY_avg_ishf']
        + q_surface_ds['ANOMALY_avg_snswrf']
        + q_surface_ds['ANOMALY_avg_snlwrf']
    )
    q_surface = q_surface.drop_vars('MONTH')
    q_surface.name = 'ANOMALY_SURFACE_HEAT_FLUX'
else:
    q_surface = 0
if ENTRAINMENT:
    q_entrainment = load_and_prepare_dataset(
        "datasets/Simulation-Entrainment_Heat_Flux-(2004-2018).nc"
    )['ANOMALY_ENTRAINMENT_HEAT_FLUX']
else:
    q_entrainment = 0
if EKMAN:
    q_ekman = load_and_prepare_dataset(
        "datasets/Simulation-Ekman_Heat_Flux-(2004-2018).nc"
    )['ANOMALY_EKMAN_HEAT_FLUX']
    q_ekman = q_ekman.where(
        (q_ekman['LATITUDE'] > 5) | (q_ekman['LATITUDE'] < -5), 0
    )
else:
    q_ekman = 0
if GEOSTROPHIC:
    q_geostrophic = load_and_prepare_dataset(
        "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2018).nc"
    )['ANOMALY_GEOSTROPHIC_HEAT_FLUX']
    q_geostrophic = q_geostrophic.where(
        (q_geostrophic['LATITUDE'] > 5) | (q_geostrophic['LATITUDE'] < -5), 0
    )
else:
    q_geostrophic = 0


t_m_a = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc"
)['ANOMALY_ML_TEMPERATURE']
t_m_a = t_m_a.drop_vars('MONTH')

h_monthly_mean = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-Seasonal_Mean.nc"
)['MONTHLY_MEAN_MLD']

h_monthly_mean = xr.concat([h_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
h_monthly_mean['TIME'] = t_m_a.TIME

dt_m_a_dt = (
    q_surface
    + q_entrainment
    + q_ekman
    + q_geostrophic
    # - 20 * t_m_a
) / (RHO_O * C_O * h_monthly_mean)

t_m_a_simulated_list = []

for month_num in t_m_a['TIME'].values:
    if month_num == 0.5:
        t_m_a_simulated_da = t_m_a.sel(TIME=month_num)
    else:
        t_m_a_simulated_da = (
            t_m_a.sel(TIME=month_num-1)
            + dt_m_a_dt.sel(TIME=month_num-1) * SECONDS_MONTH
        )
    t_m_a_simulated_da = t_m_a_simulated_da.expand_dims(TIME=[month_num])
    t_m_a_simulated_list.append(t_m_a_simulated_da)

t_m_a_simulated = xr.concat(
    t_m_a_simulated_list,
    dim="TIME",
    coords="minimal"
)

t_m_a_simulated_monthly_mean = get_monthly_mean(t_m_a_simulated)
t_m_a_simulated = get_anomaly(t_m_a_simulated, t_m_a_simulated_monthly_mean)

difference = t_m_a_simulated - t_m_a
print(difference)
difference.sel(TIME=6.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=-3, vmax=3)
