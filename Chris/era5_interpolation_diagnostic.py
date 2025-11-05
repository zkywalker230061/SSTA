import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import esmpy as ESMF
from utils import load_and_prepare_dataset
import xesmf as xe

ERA5_DATA_PATH = "../datasets/era5_interpolated.nc"

era5_ds_interpolated = xr.open_dataset(ERA5_DATA_PATH, decode_times=False)

print(abs(era5_ds_interpolated['avg_iews']).mean().item())
print(era5_ds_interpolated['avg_iews'].max().item())

print(abs(era5_ds_interpolated['avg_inss']).mean().item())
print(era5_ds_interpolated['avg_inss'].max().item())


