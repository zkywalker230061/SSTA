import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

TEMP_DATA_PATH = "../datasets/Temperature_Monthly_Mean.nc"
SAL_DATA_PATH = "../datasets/Salinity_Monthly_Mean.nc"
THRESHOLD = 1.015       # relative to the near-surface potential density

temp_ds = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)
sal_ds = xr.open_dataset(SAL_DATA_PATH, decode_times=False)
ds = xr.merge([temp_ds, sal_ds])    # combine temperature and salinity


def get_monthly_hbar(ds, month, make_plots=True):
    ds = ds.sel(MONTH=month)
    # get potential density
    SA = gsw.SA_from_SP(ds.MONTHLY_MEAN_SALINITY, ds.PRESSURE, ds.LONGITUDE, ds.LATITUDE)
    CT = gsw.CT_from_t(SA, ds.MONTHLY_MEAN_TEMPERATURE, ds.PRESSURE)
    sigma0 = gsw.sigma0(SA, CT)
    ds['POTENTIAL_DENSITY'] = sigma0
    ds['POTENTIAL_DENSITY'].attrs = {"units": "kg/m^3"}

    # get mean potential density at the top 10 m
    ds_surface = ds.sel(PRESSURE=slice(0, 10))
    sigma0_near_surface_mean = ds_surface.POTENTIAL_DENSITY.mean(dim='PRESSURE')
    ds['NEAR_SURFACE_POTENTIAL_DENSITY_MEAN'] = sigma0_near_surface_mean

    def find_mld_by_density(potential_density_profile, pressure, potential_density_near_surface_mean):
        #threshold = potential_density_near_surface_mean * THRESHOLD
        threshold = potential_density_near_surface_mean + 0.03
        above_threshold_depths = np.where(potential_density_profile >= threshold)[0]

        # catch case where threshold is not crossed (should be only over land)
        if len(above_threshold_depths) == 0:
            return np.nan

        below_mld_index = above_threshold_depths[0]
        above_mld_index = below_mld_index - 1
        return np.interp(threshold, [potential_density_profile[above_mld_index], potential_density_profile[below_mld_index]], [pressure[above_mld_index], pressure[below_mld_index]])

    mld_pressure = xr.apply_ufunc(find_mld_by_density, ds['POTENTIAL_DENSITY'], ds['PRESSURE'], ds['NEAR_SURFACE_POTENTIAL_DENSITY_MEAN'], input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True)
    ds['MLD'] = mld_pressure
    ds['MLD'] = ds['MLD'].where(ds['MLD'] <= 500, 500)  # for a better scale
    if make_plots:
        ds['MLD'].plot(x='LONGITUDE', y='LATITUDE', cmap='Blues')
        plt.show()
    return ds

monthly_datasets = []
for month in range(1, 13):
    monthly_datasets.append(get_monthly_hbar(ds, month, make_plots=True))
hbar_all_months_dataset = xr.concat(monthly_datasets, "MONTH")
hbar_all_months_dataset.to_netcdf("../datasets/mld_by_potential_density.nc")
print(hbar_all_months_dataset)
