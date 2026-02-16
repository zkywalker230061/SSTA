import xarray as xr
import pandas as pd
import numpy as np
import scipy


def monthly_to_daily(monthly_ds, MONTH_NAME, start_year=2004):
    """
    monthly_ds: xarray dataset of data which must have a time coordinate in months
    MONTH_NAME: the name of the coordinate corresponding to the month number. Expects 0.5, 1.5, 2.5 ... 179.5 ...
    start_year: the first year of the dataset, normally 2004
    WARNING! may destroy your RAM.
    """
    # get new time coordinate axis corresponding to daily data
    months = monthly_ds[MONTH_NAME]
    daily_coordinates = []
    days_in_month_list = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for month in months.values:
        if month % 1 == 0.5:
            month_as_int = int(month - 0.5)        # convert to integer
        else:
            month_as_int = int(month)
        year = start_year + month_as_int // 12
        month_in_year = month_as_int % 12 + 1
        days = days_in_month_list[month_in_year - 1]
        if year % 4 == 0 and month_in_year == 2:        # leap year case
            days = 29
        daily_coordinate = month + (np.arange(days) / days)
        daily_coordinates.append(daily_coordinate)
    daily_coordinates = np.concatenate(daily_coordinates)
    daily_ds = monthly_ds.interp({MONTH_NAME: daily_coordinates})
    return daily_ds


"""example usage"""
EXAMPLE_DATA_PATH = "../datasets/Temperature_Monthly_Mean.nc"
temp_ds = xr.open_dataset(EXAMPLE_DATA_PATH, decode_times=False)
print(temp_ds)
temp_ds_daily = monthly_to_daily(temp_ds, "MONTH", 2004)
print(temp_ds_daily)
