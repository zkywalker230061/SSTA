"""
Q_Entrainment term.

Chris O.S. & Chengyun Zhu
2025-11-05
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from h_analysis import get_anomaly
from rgargo_plot import visualise_dataset


TEMP_DATA_PATH = "../datasets/Temperature-(2004-2018).nc"
MLD_DATA_PATH = "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"

RHO_O = 1025  # kg / m^3
C_O = 4100  # J / (kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # average seconds in a month


def find_half_temperature(depth, temperature, mld):
    """
    Find the temperature at the mixed layer depth (hbar) from a temperature profile.

    Parameters
    ----------
    depth : np.ndarray
        1D array of depth values.
    temperature : np.ndarray
        1D array of temperature values corresponding to the depth profile.
    mld : float
        The mixed layer depth.

    Returns
    -------
    float
        The temperature at the mixed layer depth (hbar).
    """

    above_mld = np.where(depth <= mld)[0]
    below_mld = np.where(depth > mld)[0]

    # catch case where no MLD (should be only over land)
    if len(above_mld) == 0 or len(below_mld) == 0:
        return np.nan

    above_mld_index = above_mld[-1]
    below_mld_index = below_mld[0]
    mld_temp = np.interp(
        mld,
        [depth[above_mld_index], depth[below_mld_index]],
        [temperature[above_mld_index], temperature[below_mld_index]]
    )
    return mld_temp


def get_monthly_sub_temperature(temp_ds, mld_ds, month, make_plots=True):
    """
    Get sub-layer temperature for a given month.

    Parameters
    ----------
    temp_ds : xarray.Dataset
        Dataset containing temperature profiles.
    mld_ds : xarray.Dataset
        Dataset containing mixed layer depth profiles.
    month : np.datetime64
        The month to process.
    make_plots : bool, optional
        Whether to generate plots. Default is True.

    Returns
    -------
    xarray.Dataset
        Dataset with sub-layer temperature added.
    """

    temp_ds = temp_ds.sel(TIME=month)
    mld_ds = mld_ds.sel(TIME=month)

    slt = xr.apply_ufunc(
        find_half_temperature,
        temp_ds['PRESSURE'], temp_ds['TEMPERATURE'], mld_ds['MLD_PRESSURE'],
        input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True
    )
    mld_ds['SUB_TEMPERATURE'] = slt
    if make_plots:
        mld_ds['SUB_TEMPERATURE'].plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r')
        plt.show()
    return mld_ds


def save_sub_temperature_dataset():
    """Save the sub-layer temperature dataset to a NetCDF file."""

    temp_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
    mld_ds = load_and_prepare_dataset(MLD_DATA_PATH)

    monthly_datasets = []
    for month in temp_ds.TIME.values:
        monthly_datasets.append(
            get_monthly_sub_temperature(temp_ds, mld_ds, month, make_plots=False)
        )
    t_sub_ds = xr.concat(monthly_datasets, "TIME")
    t_sub = t_sub_ds['SUB_TEMPERATURE']

    # restore attributes
    t_sub['LATITUDE'].attrs = temp_ds['LATITUDE'].attrs
    t_sub['LONGITUDE'].attrs = temp_ds['LONGITUDE'].attrs
    t_sub.attrs['units'] = temp_ds['TEMPERATURE'].attrs['units']
    t_sub.attrs['long_name'] = (
        'Monthly Sub Layer Temperature Jan 2004 - Dec 2018 (15.0 year)'
    )
    t_sub.name = 'SUB_TEMPERATURE'
    display(t_sub)
    # visualise_dataset(
    #     t_sub.sel(TIME=121, method='nearest'),
    #     cmaps='RdBu_r',
    # )
    t_sub.to_netcdf("../datasets/Sub_Layer_Temperature-(2004-2018).nc")

    t_sub_monthly_mean = get_monthly_mean(t_sub)
    t_sub_anomaly = get_anomaly(t_sub, t_sub_monthly_mean)

    print(t_sub_anomaly.max().item(), t_sub_anomaly.min().item())
    # display(t_sub_anomaly)
    visualise_dataset(
        t_sub_anomaly.sel(TIME=121, method='nearest'),
        cmaps='RdBu_r',
        vmin=-30, vmax=30
    )


def save_entrainment_velocity():
    """Save the entrainment velocity (w_e) dataset."""

    mld_ds = load_and_prepare_dataset(MLD_DATA_PATH)
    # display(mld_ds)
    mld = mld_ds['MLD_PRESSURE']

    w_e = (
        np.gradient(mld, axis=mld.get_axis_num('TIME'))
        / SECONDS_MONTH  # convert to dbar/s
    )
    # set negative entrainment velocity to zero
    w_e = np.where(w_e < 0, 0, w_e)
    w_e_da = xr.DataArray(
        w_e,
        coords=mld.coords,
        dims=mld.dims,
        name='ENTRAINMENT_VELOCITY'
    )
    w_e_da.attrs['units'] = 'dbar/s'
    w_e_da.attrs['long_name'] = (
        'Monthly Entrainment Velocity Jan 2004 - Dec 2018 (15.0 year)'
    )
    display(w_e_da)
    # visualise_dataset(
    #     w_e_da.sel(TIME=0.5, method='nearest'),
    #     cmap='Blues',
    #     # vmin=-0.5, vmax=0.5
    # )
    w_e_da.to_netcdf("../datasets/Entrainment_Velocity-(2004-2018).nc")


def save_q_entrainment():
    """Save the Q_Entrainment dataset."""

    t_sub_ds = load_and_prepare_dataset(
        "../datasets/Sub_Layer_Temperature-(2004-2018).nc"
    )
    t_m_ds = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Temperature-(2004-2018).nc"
    )
    w_e_ds = load_and_prepare_dataset(
        "../datasets/Entrainment_Velocity-(2004-2018).nc"
    )

    t_sub = t_sub_ds['SUB_TEMPERATURE']
    t_m = t_m_ds['MLD_TEMPERATURE']
    w_e = w_e_ds['ENTRAINMENT_VELOCITY']

    q_entrainment = RHO_O * C_O * w_e * (t_sub - t_m)
    q_entrainment.attrs['units'] = 'W/m^2'
    q_entrainment.attrs['long_name'] = (
        'Monthly Q_Entrainment Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_entrainment.name = 'ENTRAINMENT_HEAT_FLUX'
    display(q_entrainment)
    q_entrainment.to_netcdf(
        "../datasets/Entrainment_Heat_Flux-(2004-2018).nc"
    )


def save_q_entrainment_anomaly():
    """Save the Q_Entrainment anomaly dataset."""

    t_sub_ds = load_and_prepare_dataset(
        "../datasets/Sub_Layer_Temperature-(2004-2018).nc"
    )
    t_m_ds = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Temperature-(2004-2018).nc"
    )
    w_e_ds = load_and_prepare_dataset(
        "../datasets/Entrainment_Velocity-(2004-2018).nc"
    )

    t_sub = t_sub_ds['SUB_TEMPERATURE']
    t_m = t_m_ds['MLD_TEMPERATURE']
    w_e = w_e_ds['ENTRAINMENT_VELOCITY']

    t_sub_monthly_mean = get_monthly_mean(t_sub)
    t_sub_anomaly = get_anomaly(t_sub, t_sub_monthly_mean)
    t_m_monthly_mean = get_monthly_mean(t_m)
    t_m_anomaly = get_anomaly(t_m, t_m_monthly_mean)

    q_entrainment_anomaly = RHO_O * C_O * w_e * (t_sub_anomaly - t_m_anomaly)
    q_entrainment_anomaly.attrs['units'] = 'W/m^2'
    q_entrainment_anomaly.attrs['long_name'] = (
        'Monthly Q_Entrainment Anomaly Jan 2004 - Dec 2018 (15.0 year)'
    )
    q_entrainment_anomaly.name = 'ENTRAINMENT_ANOMALY'
    display(q_entrainment_anomaly)
    q_entrainment_anomaly.to_netcdf(
        "../datasets/Entrainment_Heat_Flux_Anomaly-(2004-2018).nc"
    )


def main():
    """Main function for entrainment."""

    # save_sub_temperature_dataset()

    MONTH = 7

    t_sub = load_and_prepare_dataset(
        "../datasets/Sub_Layer_Temperature-(2004-2018).nc"
    )['SUB_TEMPERATURE']
    # display(t_sub)
    visualise_dataset(
        t_sub.sel(TIME=MONTH, method='nearest'),
        cmaps='RdBu_r',
    )

    t_sub_monthly_mean = get_monthly_mean(t_sub)
    t_sub_anomaly = get_anomaly(t_sub, t_sub_monthly_mean)
    print(t_sub_anomaly.max().item(), t_sub_anomaly.min().item())
    print(abs(t_sub_anomaly).mean().item())
    # display(t_sub_anomaly)
    visualise_dataset(
        t_sub_anomaly.sel(TIME=MONTH, method='nearest'),
        cmaps='RdBu_r',
        vmin=-2, vmax=2
    )

    # save_entrainment_velocity()

    # save_q_entrainment()

    # save_q_entrainment_anomaly()


if __name__ == "__main__":
    main()
