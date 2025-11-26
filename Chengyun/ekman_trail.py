"""
Q_Ekman term.

Chengyun Zhu
2025-11-12
"""

from IPython.display import display

import xarray as xr
import numpy as np
# import matplotlib.pyplot as plt

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from h_analysis import get_anomaly

WIND_STRESS_DATA_PATH = "../datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress.nc"
MIXED_LAYER_TEMP_DATA_PATH = "../datasets/Mixed_Layer_Temperature-(2004-2018).nc"

C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # average seconds in a month


tao_ds = load_and_prepare_dataset(WIND_STRESS_DATA_PATH)
# display(tao)
tao_x = tao_ds['avg_iews']
tao_y = tao_ds['avg_inss']
tao_x_monthly_mean = get_monthly_mean(tao_x)
tao_y_monthly_mean = get_monthly_mean(tao_y)
tao_x_anomaly = get_anomaly(tao_x, tao_x_monthly_mean)
tao_y_anomaly = get_anomaly(tao_y, tao_y_monthly_mean)
# display(tao_x_anomaly)
# display(tao_y_anomaly)

tm_ds = load_and_prepare_dataset(MIXED_LAYER_TEMP_DATA_PATH)
# display(tm_ds)
tm = tm_ds['MLD_TEMPERATURE']
tm_monthly_mean = get_monthly_mean(tm)
# display(tm_monthly_mean)

dtm_monthly_mean_dx = (
    np.gradient(tm_monthly_mean, axis=tm_monthly_mean.get_axis_num('LONGITUDE'))
)
dtm_monthly_mean_dx_da = xr.DataArray(
    dtm_monthly_mean_dx,
    coords=tm_monthly_mean.coords,
    dims=tm_monthly_mean.dims
)
# display(dtm_monthly_mean_dx_da)
dtm_monthly_mean_dy = (
    np.gradient(tm_monthly_mean, axis=tm_monthly_mean.get_axis_num('LATITUDE'))
)
dtm_monthly_mean_dy_da = xr.DataArray(
    dtm_monthly_mean_dy,
    coords=tm_monthly_mean.coords,
    dims=tm_monthly_mean.dims
)
# display(dtm_monthly_mean_dy_da)

f = 2 * (2*np.pi/86400) * np.sin(
    np.deg2rad(tm_monthly_mean['LATITUDE'])
)
f = f.expand_dims(
    LONGITUDE=tao_ds['LONGITUDE'],
)
# display(f)


MONTH = 0.5
for month in tao_x_anomaly.TIME.values:
    if month == MONTH:
        ekman_anomaly_ds = (
            C_O / f / SECONDS_MONTH * (
                tao_x_anomaly.sel(TIME=month) * dtm_monthly_mean_dy_da.sel(MONTH=(month % 12 + 0.5))
                - tao_y_anomaly.sel(TIME=month) * dtm_monthly_mean_dx_da.sel(MONTH=(month % 12 + 0.5))
            )
        )
        ekman_anomaly_ds = ekman_anomaly_ds.expand_dims(TIME=[MONTH])
    else:
        ekman_anomaly = (
            C_O / f / SECONDS_MONTH * (
                tao_x_anomaly.sel(TIME=month) * dtm_monthly_mean_dy_da.sel(MONTH=(month % 12 + 0.5))
                - tao_y_anomaly.sel(TIME=month) * dtm_monthly_mean_dx_da.sel(MONTH=(month % 12 + 0.5))
            )
        )
        ekman_anomaly_ds = xr.concat([ekman_anomaly_ds, ekman_anomaly.expand_dims(TIME=[month])], dim='TIME')

ekman_anomaly_ds = ekman_anomaly_ds.where(
    (np.abs(ekman_anomaly_ds['LATITUDE']) >= 5) | (np.isnan(ekman_anomaly_ds)),
    0
)

ekman_anomaly_ds.name = 'Q_Ek_anom'
display(ekman_anomaly_ds)
print(f.mean().item())
print(abs(tao_x_anomaly).mean().item(), abs(tao_y_anomaly).mean().item())
print(abs(dtm_monthly_mean_dx_da).mean().item(), abs(dtm_monthly_mean_dy_da).mean().item())
ekman_anomaly_ds.sel(TIME=0.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r')
ekman_anomaly_ds.to_netcdf("../datasets/Ekman_Current_Anomaly-trail.nc")
