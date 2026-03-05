import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import esmpy as ESMF
from utils import load_and_prepare_dataset, make_movie
import xesmf as xe
#import copernicusmarine

# copernicusmarine.subset(
#     dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m",
#     variables=["sla"],
#     minimum_longitude=-179.9375,
#     maximum_longitude=179.9375,
#     minimum_latitude=-65,
#     maximum_latitude=80,
#     start_datetime="2004-01-01T00:00:00",
#     end_datetime="2019-01-01T00:00:00",
#     output_filename="sea_surface_heights.nc",
#     output_directory="../Volumes/G-DRIVE ArmorATD/Extension/datasets"
# )

TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Temperature_Monthly_Mean.nc"
SEA_SURFACE_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_height.nc"

temp_ds_for_interpolation = xr.open_dataset(TEMP_DATA_PATH, decode_times=False)  # smaller dataset for interpolation only
temp_ds = load_and_prepare_dataset("/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc")  # for time coordinates

def reformat_era5(era5_ds):
    era5_ds = era5_ds.isel(time=slice(0, -1))
    era5_ds = era5_ds.assign_coords(TIME=("time", temp_ds.TIME.values))  # match time coordinates (due to different labelling format)
    era5_ds = era5_ds.swap_dims({"time": "TIME"})
    era5_ds = era5_ds.drop_vars("time")
    era5_ds = era5_ds.rename({"latitude": "LATITUDE"})
    era5_ds = era5_ds.rename({"longitude": "LONGITUDE"})
    return era5_ds


def interpolate_era5(era5_ds, temp_ds_for_interpolation, new_filename):
    # interpolate ERA5 onto Argo
    regridder = xe.Regridder(era5_ds, temp_ds_for_interpolation, "conservative")
    era5_ds_interpolated = regridder(era5_ds)
    era5_ds_interpolated.to_netcdf(new_filename)
    print(era5_ds_interpolated)
    return era5_ds_interpolated


sea_surface_ds = xr.open_dataset(SEA_SURFACE_DATA_PATH)
sea_surface_ds = reformat_era5(sea_surface_ds)
sea_surface_ds_interpolated = interpolate_era5(sea_surface_ds, temp_ds_for_interpolation, "../datasets/sea_surface_interpolated.nc")

print(sea_surface_ds_interpolated)
print(sea_surface_ds_interpolated['sla'])

#plot to check interpolated data looks similar
sea_surface_ds_interpolated['sla'].sel(TIME=0.5).plot(x='LONGITUDE', y='LATITUDE', cmap='viridis')
plt.show()
sea_surface_ds['sla'].sel(TIME=0.5).plot(x='LONGITUDE', y='LATITUDE', cmap='viridis')
plt.show()

#make_movie(sea_surface_ds, -20, 20)
