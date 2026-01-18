"""
Calculate Surface term.

Chengyun Zhu
2026-1-18
"""

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file


def save_monthly_mean_surface_heat_flux():
    """Calculate and save monthly mean surface heat flux dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Surface_Heat_Flux-Seasonal_Mean.nc" in logs_datasets.read():
            return

    q_surface = load_and_prepare_dataset(
        "datasets/Surface_Heat_Flux-(2004-2018).nc"
    )

    lh = q_surface['avg_slhtf']
    sh = q_surface['avg_ishf']
    sw = q_surface['avg_snswrf']
    lw = q_surface['avg_snlwrf']

    lh_monthly_mean = get_monthly_mean(lh)
    sh_monthly_mean = get_monthly_mean(sh)
    sw_monthly_mean = get_monthly_mean(sw)
    lw_monthly_mean = get_monthly_mean(lw)

    q_surface_monthly_mean = lh_monthly_mean.to_dataset(name='MONTHLY_MEAN_avg_slhtf')
    q_surface_monthly_mean['MONTHLY_MEAN_avg_ishf'] = sh_monthly_mean
    q_surface_monthly_mean['MONTHLY_MEAN_avg_snswrf'] = sw_monthly_mean
    q_surface_monthly_mean['MONTHLY_MEAN_avg_snlwrf'] = lw_monthly_mean

    save_file(
        q_surface_monthly_mean,
        "datasets/Surface_Heat_Flux-Seasonal_Mean.nc"
    )


def save_monthly_mean_surface_water_rate():
    """Calculate and save monthly mean surface water rate dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Surface_Water_Rate-Seasonal_Mean.nc" in logs_datasets.read():
            return

    q_surface = load_and_prepare_dataset(
        "datasets/Surface_Water_Rate-(2004-2018).nc"
    )

    eva = q_surface['avg_ie']
    pre = q_surface['avg_tprate']

    eva_monthly_mean = get_monthly_mean(eva)
    pre_monthly_mean = get_monthly_mean(pre)

    q_surface_monthly_mean = eva_monthly_mean.to_dataset(name='MONTHLY_MEAN_avg_ie')
    q_surface_monthly_mean['MONTHLY_MEAN_avg_tprate'] = pre_monthly_mean

    save_file(
        q_surface_monthly_mean,
        "datasets/Surface_Water_Rate-Seasonal_Mean.nc"
    )


def save_surface_heat_flux_anomalies():
    """Calculate and save surface heat flux anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Surface_Heat_Flux_Anomalies-(2004-2018).nc" in logs_datasets.read():
            return

    q_surface = load_and_prepare_dataset(
        "datasets/Surface_Heat_Flux-(2004-2018).nc"
    )
    q_surface_monthly_mean = load_and_prepare_dataset(
        "datasets/Surface_Heat_Flux-Seasonal_Mean.nc"
    )

    lh_anomalies = get_anomaly(q_surface['avg_slhtf'],
                               q_surface_monthly_mean['MONTHLY_MEAN_avg_slhtf'])
    sh_anomalies = get_anomaly(q_surface['avg_ishf'],
                               q_surface_monthly_mean['MONTHLY_MEAN_avg_ishf'])
    sw_anomalies = get_anomaly(q_surface['avg_snswrf'],
                               q_surface_monthly_mean['MONTHLY_MEAN_avg_snswrf'])
    lw_anomalies = get_anomaly(q_surface['avg_snlwrf'],
                               q_surface_monthly_mean['MONTHLY_MEAN_avg_snlwrf'])

    q_surface_anomalies = lh_anomalies.to_dataset(name='ANOMALY_avg_slhtf')
    q_surface_anomalies['ANOMALY_avg_ishf'] = sh_anomalies
    q_surface_anomalies['ANOMALY_avg_snswrf'] = sw_anomalies
    q_surface_anomalies['ANOMALY_avg_snlwrf'] = lw_anomalies

    save_file(
        q_surface_anomalies,
        "datasets/Surface_Heat_Flux_Anomalies-(2004-2018).nc"
    )


def save_surface_water_rate_anomalies():
    """Calculate and save surface water rate anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Surface_Water_Rate_Anomalies-(2004-2018).nc" in logs_datasets.read():
            return

    q_surface = load_and_prepare_dataset(
        "datasets/Surface_Water_Rate-(2004-2018).nc"
    )
    q_surface_monthly_mean = load_and_prepare_dataset(
        "datasets/Surface_Water_Rate-Seasonal_Mean.nc"
    )

    eva_anomalies = get_anomaly(q_surface['avg_ie'],
                                q_surface_monthly_mean['MONTHLY_MEAN_avg_ie'])
    pre_anomalies = get_anomaly(q_surface['avg_tprate'],
                                q_surface_monthly_mean['MONTHLY_MEAN_avg_tprate'])

    q_surface_anomalies = eva_anomalies.to_dataset(name='ANOMALY_avg_ie')
    q_surface_anomalies['ANOMALY_avg_tprate'] = pre_anomalies

    save_file(
        q_surface_anomalies,
        "datasets/Surface_Water_Rate_Anomalies-(2004-2018).nc"
    )


def save_surface_anomaly_temperature():
    """Save the Q_Surface dataset for tmperature."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Surface_Heat_Flux-(2004-2018).nc" in logs_datasets.read():
            return

    q_surface_a = load_and_prepare_dataset(
        "datasets/Surface_Heat_Flux_Anomalies-(2004-2018).nc"
    )

    save_file(
        q_surface_a,
        "datasets/Simulation-Surface_Heat_Flux-(2004-2018).nc"
    )


def save_surface_anomaly_salinity():
    """Save the Q_Surface dataset for salinity."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Simulation-Surface_Water_Rate-(2004-2018).nc" in logs_datasets.read():
            return

    q_surface_a = load_and_prepare_dataset(
        "datasets/Surface_Water_Rate_Anomalies-(2004-2018).nc"
    )

    save_file(
        q_surface_a,
        "datasets/Simulation-Surface_Water_Rate-(2004-2018).nc"
    )


def main():
    """Main function to calculate Surface term."""

    save_monthly_mean_surface_heat_flux()
    save_monthly_mean_surface_water_rate()
    save_surface_heat_flux_anomalies()
    save_surface_water_rate_anomalies()

    save_surface_anomaly_temperature()
    save_surface_anomaly_salinity()


if __name__ == "__main__":
    main()
