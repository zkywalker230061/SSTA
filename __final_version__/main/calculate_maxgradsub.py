"""
Module to calculate max gradient quantity for the sub layer.

Chengyun
2026-3-5
"""

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file


RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30 * 24 * 60 * 60  # seconds in a month


def main():
    """Main function to save sub-layer temperature and salinity for max-gradient method."""

    t = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2025).nc"
    )['TEMPERATURE']
    t_sub = load_and_prepare_dataset(
        "datasets/Sub_Layer_Temperature_Max_Gradient_Method-(2004-2025).nc"
    )['SUB_TEMPERATURE']
    t_sub['LATITUDE'].attrs = t['LATITUDE'].attrs
    t_sub['LONGITUDE'].attrs = t['LONGITUDE'].attrs
    t_sub.attrs['units'] = t.attrs['units']
    t_sub.attrs['long_name'] = (
        'Monthly Sub Layer Temperature Max Gradient Method Jan 2004 - Dec 2025 (22.0 year)'
    )
    print(t_sub)
    t_sub_monthly_mean = get_monthly_mean(t_sub)
    t_sub_anomaly = get_anomaly(t_sub, t_sub_monthly_mean)
    t_sub_monthly_mean.to_netcdf("datasets/Sub_Layer_Temperature_Max_Gradient_Method-Clim_Mean.nc")
    t_sub_anomaly.to_netcdf("datasets/Sub_Layer_Temperature_Max_Gradient_Method_Anomalies-(2004-2025).nc")


if __name__ == "__main__":
    main()
