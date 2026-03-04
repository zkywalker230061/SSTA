"""
Prepare Reynolds dataset.

NOTE: This argolise must be run under conda environment with xesmf installed.

Chengyun Zhu
2026-3-4
"""

import xarray as xr
import pandas as pd
import xesmf as xe  # NOTE: This argolise must be run under conda environment with xesmf installed.

from utilities import load_and_prepare_dataset
from utilities import get_monthly_mean, get_anomaly
from utilities import save_file

YEARS = [
    2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022, 2023,
    2024, 2025
]


def _reynolds_regrid(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Process Reynolds datasets consistent with RG-ARGO datasets.

    Parameters
    ----------
    ds: xarray.Dataset
        Input Reynolds dataset.

    Returns
    -------
    xarray.Dataset
        Processed Reynolds dataset.

    """

    argo_ds = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2025).nc"
    )

    regridder = xe.Regridder(ds, argo_ds, "conservative")
    ds_regrided = regridder(ds)

    ds_regrided = ds_regrided.rename({'time': 'TIME'})

    ds_regrided['TIME'] = (
        (ds_regrided['TIME'].dt.year - 2004) * 12
        + (ds_regrided['TIME'].dt.month - 0.5)
    ).astype('float32')

    ds_regrided['TIME'].attrs.update({
        'units': 'months since 2004-01-01 00:00:00',
        'time_origin': '01-JAN-2004 00:00:00',
        'axis': 'T'
    })

    ds_regrided.attrs.update(ds.attrs)

    return ds_regrided


def save_regridded_reynolds_anomalies():
    """Save the regridded reynolds anomalies dataset."""

    reynolds_sst_anomalies = []

    for year in YEARS:
        ds_year = xr.open_dataset(
            f"datasets/Reynolds/sst.day.anom.{year}.nc"
        )
        ds_year_monthly = ds_year.resample(time="1M").mean()
        ds_year_monthly['time'] = pd.to_datetime(
            ds_year_monthly['time'].values
        ).to_period('M').to_timestamp()
        reynolds_sst_anomalies.append(ds_year_monthly)
    reynolds_sst_anomalies = xr.concat(reynolds_sst_anomalies, dim="time")

    reynolds_sst_anomalies_regrided = _reynolds_regrid(reynolds_sst_anomalies)

    reynolds_sst_anomalies_regrided.to_netcdf(
        "datasets/Reynolds/sst.mon.anom.2004-2025.nc"
    )


def save_regridded_reynolds_ltm():
    """Save the regridded reynolds long-term mean dataset."""

    reynolds_sst_ltm_day = xr.open_dataset(
        "datasets/Reynolds/sst.day.mean.ltm.1971-2000.nc", decode_times=False
    )

    reynolds_sst_ltm_day['time'] = pd.to_datetime(
        reynolds_sst_ltm_day['time'].values+657073, unit='D', origin=pd.Timestamp('1970-01-01')
    )

    reynolds_sst_ltm = reynolds_sst_ltm_day.resample(time="1M").mean()
    reynolds_sst_ltm['time'] = pd.to_datetime(
        reynolds_sst_ltm['time'].values
    ).to_period('M').to_timestamp()

    argo_ds = load_and_prepare_dataset(
        "datasets/Temperature-(2004-2025).nc"
    )

    regridder = xe.Regridder(reynolds_sst_ltm, argo_ds, "conservative")
    reynolds_sst_ltm_regrided = regridder(reynolds_sst_ltm)

    reynolds_sst_ltm_regrided = reynolds_sst_ltm_regrided.rename({'time': 'MONTH'})

    reynolds_sst_ltm_regrided.attrs.update(reynolds_sst_ltm.attrs)

    reynolds_sst_ltm_regrided.to_netcdf(
        "datasets/Reynolds/sst.mon.ltm.1971-2000.nc"
    )


def save_reynolds_sst():
    """Save the regridded reynolds sst dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/reynolds_sst-(2004-2025).nc" in logs_datasets.read():
            return

    sst_30years_mean = load_and_prepare_dataset(
        "datasets/Reynolds/sst.mon.ltm.1971-2000.nc",
    )['sst']
    ssta = load_and_prepare_dataset(
        "datasets/Reynolds/sst.mon.anom.2004-2025.nc",
    )['anom']

    sst_30years_mean = xr.concat([sst_30years_mean] * 22, dim='MONTH').reset_coords(drop=True)
    sst_30years_mean = sst_30years_mean.rename({'MONTH': 'TIME'})
    sst_30years_mean['TIME'] = ssta.TIME

    sst = sst_30years_mean + ssta

    sst.attrs['units'] = "degree celcius (ITS-90)"
    sst.attrs['long_name'] = (
        "Reynolds SST Jan 2004 - Dec 2025 (22.0 year)"
    )
    sst.name = "SST"

    save_file(
        sst,
        "datasets/reynolds_sst-(2004-2025).nc"
    )


def save_monthly_mean_reynolds_sst():
    """Save the monthly mean reynolds sst dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/reynolds_sst_Clim_Mean.nc" in logs_datasets.read():
            return

    sst = load_and_prepare_dataset(
        "datasets/reynolds_sst-(2004-2025).nc",
    )['SST']
    sst_monthly_mean = get_monthly_mean(sst)
    save_file(
        sst_monthly_mean,
        "datasets/reynolds_sst_Clim_Mean.nc"
    )


def save_reynolds_sst_anomalies():
    """Save the reynolds sst anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/reynolds_sst_Anomalies-(2004-2025).nc" in logs_datasets.read():
            return

    sst = load_and_prepare_dataset(
        "datasets/reynolds_sst-(2004-2025).nc",
    )['SST']
    sst_monthly_mean = load_and_prepare_dataset(
        "datasets/reynolds_sst_Clim_Mean.nc",
    )['MONTHLY_MEAN_SST']
    sst_anomalies = get_anomaly(sst, sst_monthly_mean)
    save_file(
        sst_anomalies,
        "datasets/reynolds_sst_Anomalies-(2004-2025).nc"
    )


def main():
    """Main function to prepare datasets from RGARGO."""

    save_regridded_reynolds_anomalies()
    save_regridded_reynolds_ltm()

    save_reynolds_sst()
    save_monthly_mean_reynolds_sst()
    save_reynolds_sst_anomalies()


if __name__ == "__main__":
    main()
