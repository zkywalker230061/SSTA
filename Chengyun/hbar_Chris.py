"""
hbar calculation by Chris.

Chris O.S.
2024-10-15
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

TEMP_DATA_PATH = "../datasets/Temperature_Monthly_Mean.nc"

ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)


def get_monthly_hbar(ds, month, make_plots=True):
    ds = ds.sel(MONTH=month)

    def find_half_depth(temp_profile, pressure):
        sst = temp_profile[0]  # temperature at surface == first reading
        mld_t = sst - 1  # temperature at mixed layer depth

        # find where temperature falls below mld_t
        temperatures_below_mld = np.where(temp_profile <= mld_t)[0]
        if len(temperatures_below_mld) == 0:  # indicates a continent
            return np.nan
        below_mld_index = temperatures_below_mld[0]
        above_mld_index = below_mld_index - 1

        # linear interpolation
        return np.interp(mld_t, [temp_profile[above_mld_index], temp_profile[below_mld_index]],
                         [pressure[above_mld_index], pressure[below_mld_index]])

    # Apply this function along the depth dimension
    mld_pressure = xr.apply_ufunc(find_half_depth, ds['MONTHLY_MEAN_TEMPERATURE'], ds['PRESSURE'], input_core_dims=[['PRESSURE'], ['PRESSURE']], vectorize=True)

    ds.drop_vars(["PRESSURE"])  # don't need pressure anymore
    ds['MLD_PRESSURE'] = mld_pressure   # save to dataset
    ds['MLD_PRESSURE'] = ds['MLD_PRESSURE'].where(ds['MLD_PRESSURE'] <= 500, 500)  # for a better scale
    if make_plots:
        ds['MLD_PRESSURE'].plot(x='LONGITUDE', y='LATITUDE', cmap='Blues')
    plt.show()

    return ds


monthly_datasets = []
for month in range(1, 13):
    monthly_datasets.append(get_monthly_hbar(ds, month, make_plots=False))
hbar_all_months_dataset = xr.concat(monthly_datasets, "MONTH")
# print(hbar_all_months_dataset)
# display(hbar_all_months_dataset)

hbar_ds = hbar_all_months_dataset['MLD_PRESSURE']
# restore attributes
hbar_ds['LATITUDE'].attrs = ds['LATITUDE'].attrs
hbar_ds['LONGITUDE'].attrs = ds['LONGITUDE'].attrs
hbar_ds.attrs['units'] = 'dbar'
hbar_ds.attrs['long_name'] = (
    'Monthly Mean Mixed Layer Depth Pressure Jan 2004 - Dec 2018 (15.0 year)'
)
hbar_ds.name = 'MONTHLY_MEAN_MLD_PRESSURE'
display(hbar_ds)
hbar_ds.to_netcdf("../datasets/Mixed_Layer_Depth_Pressure_Monthly_Mean.nc")
