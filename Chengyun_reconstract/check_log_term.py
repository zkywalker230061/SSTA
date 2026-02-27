"""
File to check
if the log term is the same distribution with w_e / h.
"""

import xarray as xr
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from utilities import load_and_prepare_dataset


SECONDS_MONTH = 30 * 24 * 3600


t = load_and_prepare_dataset(
    "datasets/Temperature-(2004-2018).nc"
)['TEMPERATURE']

w_e_monthly_mean = load_and_prepare_dataset(
    'datasets/Mixed_Layer_Entrainment_Velocity-Seasonal_Mean.nc'
)['MONTHLY_MEAN_w_e']
w_e_monthly_mean = xr.concat([w_e_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
w_e_monthly_mean = w_e_monthly_mean.rename({'MONTH': 'TIME'})
w_e_monthly_mean['TIME'] = t.TIME
h_monthly_mean = load_and_prepare_dataset(
    'datasets/Mixed_Layer_Depth-Seasonal_Mean.nc'
)['MONTHLY_MEAN_MLD']
h_monthly_mean = xr.concat([h_monthly_mean] * 15, dim='MONTH').reset_coords(drop=True)
h_monthly_mean = h_monthly_mean.rename({'MONTH': 'TIME'})
h_monthly_mean['TIME'] = t.TIME

sample1 = w_e_monthly_mean / h_monthly_mean

sample2_list = []
for month_num in t['TIME'].values:
    if month_num == 179.5:
        sample2_da = h_monthly_mean.sel(TIME=month_num) - h_monthly_mean.sel(TIME=month_num)
    else:
        sample2_da = (
            np.log(h_monthly_mean.sel(TIME=month_num+1)/h_monthly_mean.sel(TIME=month_num))
            / SECONDS_MONTH
        )
    sample2_da = sample2_da.expand_dims(TIME=[month_num])
    sample2_list.append(sample2_da)

sample2 = xr.concat(
    sample2_list,
    dim="TIME",
    coords="minimal"
)


def _find_stats(s1, s2):
    result = stats.kstest(s1, s2)
    return result


statistics, pvalue = xr.apply_ufunc(
    _find_stats,
    sample1, sample2,
    input_core_dims=[['TIME'], ['TIME']],
    output_core_dims=[[], []],
    vectorize=True
)

pvalue.plot()
plt.show()

normalised_rmse = (
    np.sqrt(((sample1 - sample2) ** 2).mean(dim='TIME'))
    / np.sqrt((sample2 ** 2).mean(dim='TIME'))
)
normalised_rmse.plot(cmap='nipy_spectral')
plt.show()

corr = xr.corr(sample1, sample2, dim='TIME')
corr.plot(cmap='nipy_spectral', vmin=-1, vmax=1)
plt.show()
