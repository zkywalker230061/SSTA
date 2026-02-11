"""
Prepare Reynolds dataset.

NOTE: This argolise must be run under conda environment with xesmf installed.

Chengyun Zhu
2026-2-5
"""

import xarray as xr
import pandas as pd
import xesmf as xe  # NOTE: This argolise must be run under conda environment with xesmf installed.

from utilities import load_and_prepare_dataset
# from utilities import get_monthly_mean, get_anomaly
# from utilities import save_file

YEARS = [
    2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018
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
        "datasets/Temperature-(2004-2018).nc"
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


def save_reynolds_anomalies():
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

    reynolds_sst_anomalies.to_netcdf("datasets/Reynolds/sst.mon.anom-(2004-2018).nc")


def save_regridded_reynolds_anomalies():
    """Save the regridded reynolds anomalies dataset."""

    with open("logs/datasets.txt", "r", encoding="utf-8") as logs_datasets:
        if "datasets/Reynolds/sst_anomalies-(2004-2018).nc" in logs_datasets.read():
            return

    reynolds_sst_anomalies = xr.open_dataset(
        "datasets/Reynolds/sst.mon.anom-(2004-2018).nc"
    )

    reynolds_sst_anomalies_regrided = _reynolds_regrid(reynolds_sst_anomalies)

    # save_file(
    #     reynolds_sst_anomalies_regrided,
    #     "datasets/Reynolds/sst_anomalies-(2004-2018).nc"
    # )
    reynolds_sst_anomalies_regrided.to_netcdf(
        "datasets/Reynolds/sst_anomalies-(2004-2018).nc"
    )


def main():
    """Main function to prepare datasets from RGARGO."""

    # save_reynolds_anomalies()
    # save_regridded_reynolds_anomalies()
    test = load_and_prepare_dataset('datasets/Reynolds/sst_anomalies-(2004-2018).nc')
    print(test)
    anom = test['anom']
    anom.sel(TIME=0.5).plot()


if __name__ == "__main__":
    main()
