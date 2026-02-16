"""
File to calcuate feedback parameter.

Chengyun Zhu
2026-2-11
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file

TAO = 1

t_m_a = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Temperature_Anomalies-(2004-2018).nc"
)['ANOMALY_ML_TEMPERATURE']

t_m_a_tao = t_m_a.shift(TIME=-TAO)

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
q_ekman = load_and_prepare_dataset(
    "datasets/.test-Simulation-Ekman_Heat_Flux-(2004-2018).nc"
)['ANOMALY_EKMAN_HEAT_FLUX']
q_ekman = q_ekman.where(
    (q_ekman['LATITUDE'] > 5) | (q_ekman['LATITUDE'] < -5), 0
)
q_geostrophic = load_and_prepare_dataset(
    "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2018).nc"
)['ANOMALY_GEOSTROPHIC_HEAT_FLUX']
q_geostrophic = q_geostrophic.where(
    (q_geostrophic['LATITUDE'] > 5) | (q_geostrophic['LATITUDE'] < -5), 0
)
q_entrainment = load_and_prepare_dataset(
    "datasets/Simulation-Entrainment_Heat_Flux-(2004-2018).nc"
)['ANOMALY_ENTRAINMENT_HEAT_FLUX']

q_net = q_surface + q_ekman + q_geostrophic + q_entrainment


def calculate_feedback_parameter():
    """Calculate feedback parameter."""

    for longitude in t_m_a['LONGITUDE'].values:
        for latitude in t_m_a['LATITUDE'].values:
            if np.isnan(t_m_a_tao.sel(LONGITUDE=longitude, LATITUDE=latitude)).all():
                continue
            if np.isnan(q_net.sel(LONGITUDE=longitude, LATITUDE=latitude)).all():
                continue
            _lambda_a = -(
                np.nanmean(np.cov(
                    t_m_a_tao.sel(LONGITUDE=longitude, LATITUDE=latitude),
                    q_net.sel(LONGITUDE=longitude, LATITUDE=latitude)
                )) / np.nanmean(np.cov(
                    t_m_a_tao.sel(LONGITUDE=longitude, LATITUDE=latitude),
                    t_m_a.sel(LONGITUDE=longitude, LATITUDE=latitude)
                ))
            )
            print(_lambda_a)
    return _lambda_a


def main():
    """Main function to calculate feedback parameter."""

    lambda_a = calculate_feedback_parameter()
    print(lambda_a)
    print("Feedback parameter:", lambda_a.mean().item())



if __name__ == "__main__":
    main()