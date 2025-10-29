import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import esmpy as ESMF
from utils import load_and_prepare_dataset
import xesmf as xe

TEMP_DATA_PATH = "../datasets/Temperature_Monthly_Mean.nc"
WIND_STRESS_DATA_PATH = "../datasets/era5_mean_surface_wind_stress.nc"
HEAT_FLUX_DATA_PATH = "../datasets/era5_net_heat_flux.nc"

def reformat_era5(era5_ds):
    era5_ds = era5_ds.assign_coords(TIME=temp_ds.TIME)  # match time coordinates (due to different labelling format)
    era5_ds = era5_ds.drop_vars(["valid_time", "number", "expver"])
    era5_ds = era5_ds.rename({"valid_time": "TIME"})
    return era5_ds


def interpolate_era5(era5_ds, temp_ds_for_interpolation, new_filename):
    # interpolate ERA5 onto Argo
    regridder = xe.Regridder(era5_ds, temp_ds_for_interpolation, "conservative")
    era5_ds_interpolated = regridder(era5_ds)
    #argo_era5_ds = xr.merge([temp_ds, sal_ds, era5_ds_interpolated])    # combine all datasets
    era5_ds_interpolated.to_netcdf(new_filename)
    #argo_era5_ds.to_netcdf("../datasets/argo_era5.nc")
    print(era5_ds_interpolated)
    return era5_ds_interpolated

temp_ds_for_interpolation = xr.open_dataset(TEMP_DATA_PATH, decode_times=False) # smaller dataset for interpolation only
temp_ds = load_and_prepare_dataset("../datasets/RG_ArgoClim_Temperature_2019.nc")   # for time coordinates
sal_ds = load_and_prepare_dataset("../datasets/RG_ArgoClim_Salinity_2019.nc")

# wind_stress_ds = xr.open_dataset(WIND_STRESS_DATA_PATH)
# wind_stress_ds = reformat_era5(wind_stress_ds)
# wind_stress_ds_interpolated = interpolate_era5(wind_stress_ds, temp_ds_for_interpolation, "../datasets/wind_stress_interpolated.nc")

heat_flux_ds = xr.open_dataset(HEAT_FLUX_DATA_PATH)
heat_flux_ds = reformat_era5(heat_flux_ds)
heat_flux_ds_interpolated = interpolate_era5(heat_flux_ds, temp_ds_for_interpolation, "../datasets/heat_flux_interpolated.nc")


# plot to check interpolated data looks similar
# wind_stress_ds_interpolated['avg_iews'].sel(TIME=0.5).plot(x='LONGITUDE', y='LATITUDE', cmap='viridis')
# plt.show()
# wind_stress_ds['avg_iews'].sel(TIME=0.5).plot(x='longitude', y='latitude', cmap='viridis')
# plt.show()
