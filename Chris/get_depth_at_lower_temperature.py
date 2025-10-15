import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

TEMP_DATA_PATH = "RG_ArgoClim_Temperature_2019.nc"

ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
ds = ds.sel(TIME=0.5)
print(ds)
print(type(ds))


def find_half_depth(temp_profile, pressure):
    sst = temp_profile[0]
    mld_t = sst - 1

    # find where temperature falls below mld_t
    temperatures_below_mld = np.where(temp_profile <= mld_t)[0]
    if len(temperatures_below_mld) == 0:        # edgecase: never reach MLD. but this should be impossible
        #print("never reached MLD. impossible?")
        return np.nan
    below_mld_index = temperatures_below_mld[0]
    if below_mld_index == 0:
        return pressure[0]
    above_mld_index = below_mld_index - 1

    # linear interpolation
    return np.interp(mld_t, [temp_profile[above_mld_index], temp_profile[below_mld_index]], [pressure[above_mld_index], pressure[below_mld_index]])


# Apply this function along the depth dimension
mld_pressure = xr.apply_ufunc(
    find_half_depth,
    ds['ARGO_TEMPERATURE_ANOMALY'],
    ds['PRESSURE'],
    input_core_dims=[['PRESSURE'], ['PRESSURE']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float],
)

ds['MLD_PRESSURE'] = mld_pressure

ds['MLD_PRESSURE'].plot(x='LONGITUDE', y='LATITUDE')
plt.show()
