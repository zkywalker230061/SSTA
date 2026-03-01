"""
Machine Learning.
(Nothing to do with machine learning)
SOLID

Chengyun Zhu
2025-12-12
"""


from IPython.display import display

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
)
from scipy.stats import kstest

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from h_analysis import get_anomaly


rgargo = load_and_prepare_dataset('../datasets/RG_ArgoClim_Temperature_2019.nc')
display(rgargo)

temp_ds = rgargo['ARGO_TEMPERATURE_MEAN'] + rgargo['ARGO_TEMPERATURE_ANOMALY']
monthly_mean_temp_da = get_monthly_mean(temp_ds)
temp_anomaly_da = get_anomaly(temp_ds, monthly_mean_temp_da)
display(temp_anomaly_da)

y_test = temp_anomaly_da


def simulation(gamma, ratio_of_explicit) -> xr.DataArray:
    """
    Simulate SSTA.

    Parameters
    ----------
    gamma : float
        Hyperparameter for the model.
    ratio_of_explicit : float
        Ratio of explicit data used in the model.

    Returns
    -------
    xr.DataArray
        Simulated SSTA.
    """

    # TODO
    return None


results = pd.DataFrame(columns=['gamma', 'ratio_of_explicit',
                                'accuracy',
                                'MAE', 'MSE', 'RMSE', 'R2_score',
                                'K-S_test_statistic', 'K-S_test_p_value'])

gammas = [5, 10, 20, 100, 1000]
ratios_of_explicit = [0, 0.1, 0.4, 0.5, 0.75, 1]


for gamma in gammas:
    for ratio in ratios_of_explicit:
        # for kernel in kernels:
            y_pred = simulation(gamma, ratio)

            accuracy = accuracy_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            ks_result = kstest(y_pred, y_test)

            result = {
                'gamma': gamma,
                'ratio_of_explicit': ratio,
                'accuracy': accuracy,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2_score': r2,
                'K-S_test_statistic': ks_result.statistic,
                'K-S_test_p_value': ks_result.pvalue
            }

            results = pd.concat([results, pd.DataFrame([result])])

display(results)
