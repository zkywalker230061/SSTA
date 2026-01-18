from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly

import xarray as xr


t_sub = load_and_prepare_dataset(
    'datasets/Sub_Layer_Temperature-(2004-2018).nc'
)['SUB_TEMPERATURE']
t_sub_monthly_mean = get_monthly_mean(t_sub)
t_sub_anomaly = get_anomaly(t_sub, t_sub_monthly_mean)

t_m = load_and_prepare_dataset(
    'datasets/Mixed_Layer_Temperature-(2004-2018).nc'
)['ML_TEMPERATURE']
t_m_monthly_mean = get_monthly_mean(t_m)
t_m_anomaly = get_anomaly(t_m, t_m_monthly_mean)

print(t_sub_anomaly.mean(), t_m_anomaly.mean())
correlation = xr.corr(t_sub_anomaly, t_m_anomaly, dim='TIME')
correlation_da = xr.DataArray(
    correlation,
    dims=['LATITUDE', 'LONGITUDE'],
    coords={
        'LATITUDE': correlation['LATITUDE'],
        'LONGITUDE': correlation['LONGITUDE']
    },
)

correlation_da.plot()
