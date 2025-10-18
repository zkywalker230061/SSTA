import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

TEMP_DATA_PATH = "Temperature_Monthly_Mean.nc"

ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)


def get_monthly_hbar(ds, month, make_plots=True):
    ds = ds.sel(MONTH=month)

    def find_mld_by_temperature(temp_profile, pressure):
        # sort pressure and temperature to be in order of increasing pressure (they should be already, but just in case)
        indices_increasing_pressure = np.argsort(pressure)
        pressure = pressure[indices_increasing_pressure]
        temp_profile = temp_profile[indices_increasing_pressure]

        sst = temp_profile[0]  # temperature at surface == first in list after sorting
        mld_t = sst - 0.2  # temperature at mixed layer depth

        # find where temperature falls below mld_t
        temperatures_below_mld = np.where(temp_profile <= mld_t)[0]
        if len(temperatures_below_mld) == 0:  # indicates a continent
            return np.nan
        below_mld_index = temperatures_below_mld[0]
        above_mld_index = below_mld_index - 1

        # linear interpolation between the indices above/below the actual MLD
        return np.interp(mld_t, [temp_profile[above_mld_index], temp_profile[below_mld_index]],
                         [pressure[above_mld_index], pressure[below_mld_index]])

    # Apply this function along the depth dimension
    mld_pressure = xr.apply_ufunc(find_mld_by_temperature, ds['MONTHLY_MEAN_TEMPERATURE'], ds['PRESSURE'], input_core_dims=[['PRESSURE'], ['PRESSURE']], vectorize=True)

    ds.drop_vars(["PRESSURE"])  # don't need pressure anymore
    ds['MLD_PRESSURE'] = mld_pressure   # save to dataset
    ds['MLD_PRESSURE'] = ds['MLD_PRESSURE'].where(ds['MLD_PRESSURE'] <= 500, 500)  # for a better scale
    if make_plots:
        ds['MLD_PRESSURE'].plot(x='LONGITUDE', y='LATITUDE', cmap='Blues')
    plt.show()

    return ds

monthly_datasets = []
for month in range(1, 13):
    monthly_datasets.append(get_monthly_hbar(ds, month, make_plots=True))
hbar_all_months_dataset = xr.concat(monthly_datasets, "MONTH")
print(hbar_all_months_dataset)
