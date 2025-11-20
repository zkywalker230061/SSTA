"""
Error analysis for datasets.

Chengyun Zhu
2025-11-13
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from h_analysis import get_anomaly
# from rgargo_plot import visualise_dataset

MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3,
    'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9,
    'Oct': 10, 'Nov': 11, 'Dec': 12
}


def get_error(
    mean_da: xr.DataArray,
    anomaly_da: xr.DataArray
) -> xr.DataArray:
    """
    Get the error (standard deviation) of the DataArray based on the provided anomaly.
    Parameters
    ----------
    mean_da: xarray.DataArray
        Mean DataArray with 'MONTH' coordinate.
    anomaly_da: xarray.DataArray
        Anomaly DataArray with 'TIME' coordinate.

    Returns
    -------
    xarray.DataArray
        Error DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """

    if 'TIME' not in anomaly_da.dims:
        raise ValueError("The Anomaly DataArray must have a TIME dimension.")
    error_list = []
    for month_num in mean_da.coords['MONTH']:
        month_num = month_num.values
        month_anomalies = anomaly_da.sel(TIME=((anomaly_da['TIME'] + 0.5) % 12) == month_num % 12)
        error = np.sqrt(np.mean(month_anomalies**2, axis=0))
        error_list.append(error)
    error_da = xr.concat(error_list, "MONTH")
    error_da.attrs['units'] = anomaly_da.attrs.get('units')
    error_da.attrs['long_name'] = f"Error of {anomaly_da.attrs.get('long_name')}"
    error_da.name = f"ERROR_{anomaly_da.name}"
    return error_da


def get_mean_of_error(
    error_da: xr.DataArray
) -> xr.DataArray:
    """
    Get the mean of the error DataArray over lattitude and longitude.

    Parameters
    ----------
    error_da: xarray.DataArray
        Error DataArray with 'MONTH' coordinate.

    Returns
    -------
    xarray.DataArray
        Mean of error DataArray.
    """

    mean_error_da = error_da.mean(dim=['LATITUDE', 'LONGITUDE'])
    mean_error_da.attrs['long_name'] = f"Mean of {error_da.attrs.get('long_name')}"
    mean_error_da.name = f"MEAN_{error_da.name}"
    return mean_error_da


def error_analysis(da: xr.DataArray):
    """
    Perform error analysis on the provided DataArray.

    Parameters
    ----------
    da: xarray.DataArray
        The DataArray to analyze.
    """

    da_monthly_mean = get_monthly_mean(da)
    da_anomaly = get_anomaly(da, da_monthly_mean)

    da_error = get_error(da_monthly_mean, da_anomaly)
    # display(da_error)
    # visualise_dataset(
    #     da_error.sel(MONTH=1),
    #     cmap='Reds',
    # )

    da_mean_error = get_mean_of_error(da_error)
    display(da_mean_error)
    x = np.array(list(MONTHS.values()))
    plt.plot(
        x, da_mean_error,
        marker='o',
        linestyle='-',
        color='#66CCFF'
    )
    plt.xticks(x)
    plt.xlabel('Month')
    plt.ylabel(da_mean_error.attrs.get('units', ''))
    plt.title(da_mean_error.attrs.get('long_name', ''))
    plt.show()


def simulation_error_analysis(
    simulation_da: xr.DataArray
):
    """
    Perform error analysis on the simulation DataArray.

    Parameters
    ----------
    simulation_da: xarray.DataArray
        The simulation DataArray to analyze.

    Returns
    -------
    xarray.DataArray
        Error DataArray.

    Raises
    ------
    ValueError
        If the DataArray does not have a TIME dimension.
    """

    t = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Temperature-(2004-2018).nc",
    )
    # display(t)
    t_monthly_mean = get_monthly_mean(t['MLD_TEMPERATURE'])
    # display(t_monthly_mean)
    t_anomaly = get_anomaly(t['MLD_TEMPERATURE'], t_monthly_mean)
    # display(t_anomaly)

    if 'TIME' not in simulation_da.dims:
        raise ValueError("The Anomaly DataArray must have a TIME dimension.")
    error_list = []
    for month_num in simulation_da.coords['TIME']:
        month_num = month_num.values
        month_error = simulation_da.sel(TIME=month_num) - t_anomaly.sel(TIME=month_num)
        # get error of this month
        # error = np.sqrt(np.mean(month_error**2))
        error = np.mean(abs(month_error))
        error_list.append(error)
    error_da = xr.concat(error_list, "TIME")
    error_da.attrs['units'] = t_anomaly.attrs.get('units')
    error_da.attrs['long_name'] = f"Error of {t_anomaly.attrs.get('long_name')}"
    error_da.name = f"ERROR_{t_anomaly.name}"
    return error_da


def main():
    """Main function for error analysis."""

    # mld_ds = load_and_prepare_dataset(
    #     "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"
    # )
    # # display(mld_ds)
    # mld = mld_ds['MLD_PRESSURE']

    # error_analysis(mld)

    # heat_flux_ds = load_and_prepare_dataset(
    #     "../datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
    # )
    # # display(heat_flux_ds)
    # heat_flux = (
    #     heat_flux_ds['avg_slhtf']
    #     + heat_flux_ds['avg_ishf']
    #     + heat_flux_ds['avg_snswrf']
    #     + heat_flux_ds['avg_snlwrf']
    # )
    # heat_flux.attrs.update(
    #     units='W m**-2', long_name='Net Surface Heat Flux'
    # )
    # heat_flux.name = 'NET_HEAT_FLUX'

    # # error_analysis(heat_flux)

    # ekman_ds = load_and_prepare_dataset(
    #     "../datasets/Ekman_Current_Anomaly.nc"
    # )
    # # display(ekman_ds)
    # ekman = ekman_ds['Q_Ek_anom']

    # # error_analysis(ekman)

    # entrainment_ds = load_and_prepare_dataset(
    #     "../datasets/Entrainment_Heat_Flux-(2004-2018).nc"
    # )
    # # display(entrainment_ds)
    # entrainment = entrainment_ds['ENTRAINMENT_HEAT_FLUX']

    # # error_analysis(entrainment)

    # heat_flux_error = get_mean_of_error(
    #     get_error(
    #         get_monthly_mean(heat_flux),
    #         get_anomaly(heat_flux, get_monthly_mean(heat_flux))
    #     )
    # )

    # entrainment_error = get_mean_of_error(
    #     get_error(
    #         get_monthly_mean(entrainment),
    #         get_anomaly(entrainment, get_monthly_mean(entrainment))
    #     )
    # )

    # ekman_error = get_mean_of_error(
    #     get_error(
    #         get_monthly_mean(ekman),
    #         get_anomaly(ekman, get_monthly_mean(ekman))
    #     )
    # )

    # plt.figure(figsize=(12, 6))
    # x = np.array(list(MONTHS.values()))
    # plt.plot(
    #     x, heat_flux_error,
    #     marker='o',
    #     linestyle='-',
    #     label='Heat Flux Standard Deviation',
    #     color='#FF9999'
    # )
    # plt.plot(
    #     x, entrainment_error,
    #     marker='o',
    #     linestyle='-',
    #     label='Entrainment Standard Deviation',
    #     color='#66CCFF'
    # )
    # plt.plot(
    #     x, ekman_error,
    #     marker='o',
    #     linestyle='-',
    #     label='Ekman Current Standard Deviation',
    #     color='#99FF99'
    # )
    # plt.xticks(x)
    # plt.xlabel('Month')
    # plt.ylabel('Standard Deviation')
    # plt.title('Mean Standard Deviation Over Months')
    # plt.legend()
    # plt.show()

    semi_implicit_ds = load_and_prepare_dataset(
        "../datasets/_Semi_Implicit_Scheme_Test_ConstDamp(10)"
    )
    # display(semi_implicit_ds)
    semi_implicit = semi_implicit_ds["T_model_anom_semi_implicit"]
    semi_implicit_error = simulation_error_analysis(
        semi_implicit
    )
    # display(semi_implicit_error)

    explicit_ds = load_and_prepare_dataset(
        "../datasets/Simulated_SSTA-Explicit.nc"
    )
    # display(explicit_ds)
    explicit = explicit_ds["ANOMALY_MLD_TEMPERATURE"]
    explicit = explicit.drop_vars(['MONTH'])
    explicit_error = simulation_error_analysis(
        explicit
    )
    # display(explicit_error)

    crack_ds = load_and_prepare_dataset(
        "../datasets/_Crack_Scheme_Test_ConstDamp(10)"
    )
    # display(crack_ds)
    crack = crack_ds["T_model_anom_crank_nicolson"]
    crack_error = simulation_error_analysis(
        crack
    )
    # display(crack_error)

    implicit_ds = load_and_prepare_dataset(
        "../datasets/Simulated_SSTA-Implicit.nc"
    )
    # display(implicit_ds)
    implicit = implicit_ds["ANOMALY_MLD_TEMPERATURE"]
    implicit = implicit.drop_vars(['MONTH'])
    implicit_error = simulation_error_analysis(
        implicit
    )
    # display(implicit_error)

    chris_ds = load_and_prepare_dataset(
        "../datasets/model_anomaly_exponential_damping_implicit.nc",
    )
    # display(chris_ds)
    chris = chris_ds["ARGO_TEMPERATURE_ANOMALY"]
    chris_error = simulation_error_analysis(
        chris
    )
    # display(chris_error)

    chris_no_entrain_ds = load_and_prepare_dataset(
        "../datasets/model_anomaly_exponential_damping_implicit_no_entrain.nc",
    )
    # display(chris_no_entrain_ds)
    chris_no_entrain = chris_no_entrain_ds["ARGO_TEMPERATURE_ANOMALY"]
    chris_no_entrain_error = simulation_error_analysis(
        chris_no_entrain
    )

    plt.figure(figsize=(12, 6))
    # plt.plot(
    #     explicit_error['TIME'], explicit_error,
    #     marker='o',
    #     linestyle='-',
    #     label='Explicit Scheme Error',
    #     color='#FF9999'
    # )
    plt.plot(
        semi_implicit_error['TIME'], semi_implicit_error,
        marker='o',
        linestyle='-',
        label='Semi-Implicit Scheme Error',
        color='#66CCFF'
    )
    plt.plot(
        crack_error['TIME'], crack_error,
        marker='o',
        linestyle='-',
        label='Crank-Nicolson Scheme Error',
        color='#99FF99'
    )
    plt.plot(
        implicit_error['TIME'], implicit_error,
        marker='o',
        linestyle='-',
        label='Implicit Scheme Error',
        color='#FFCC66'
    )
    plt.plot(
        chris_error['TIME'], chris_error,
        marker='o',
        linestyle='-',
        label="Chris' Scheme Error",
        color='#CC99FF'
    )
    # plt.plot(
    #     chris_no_entrain_error['TIME'], chris_no_entrain_error,
    #     marker='o',
    #     linestyle='-',
    #     label="Chris' Scheme without Entrainment Error",
    #     color='#FF66CC'
    # )
    plt.ylim(-0.5, 2)
    # plt.xticks(explicit_error['TIME'])
    plt.xlabel('Months')
    plt.ylabel('RMSE')
    plt.title('Simulation Scheme Error Analysis')
    plt.legend()
    plt.show()

    print("explicit", explicit_error.mean().item())
    print("semi", semi_implicit_error.mean().item())
    print("crank", crack_error.mean().item())
    print("implicit", implicit_error.mean().item())
    print("chris", chris_error.mean().item())
    print("chris no entrain", chris_no_entrain_error.mean().item())


if __name__ == "__main__":
    main()
