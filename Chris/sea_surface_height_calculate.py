import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask
from dask.diagnostics import ProgressBar

from utils import load_and_prepare_dataset, make_movie

ALREADY_COMBINED = True
g = 9.81
ref_pressure = 2000     # dbar (==m more or less); choose to be where horizontal gradient is 0
ref_dym_pressure = g * ref_pressure

if ALREADY_COMBINED:
    ds = xr.open_dataset("/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Actual.nc", decode_times=False).sel(PRESSURE=slice(0, ref_pressure))
else:
    temp_ds = load_and_prepare_dataset("/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc")
    sal_ds = load_and_prepare_dataset("/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Salinity_2019.nc")
    print("starting")
    # combine mean with anomaly to reconstruct actual temperature and salinity
    tm = temp_ds["ARGO_TEMPERATURE_MEAN"]
    ta = temp_ds["ARGO_TEMPERATURE_ANOMALY"]
    tm = tm.expand_dims(TIME=180)
    actual_temp = tm + ta
    print("done temperature")

    sm = sal_ds["ARGO_SALINITY_MEAN"]
    sa = sal_ds["ARGO_SALINITY_ANOMALY"]
    sm = sm.expand_dims(TIME=180)
    actual_sal = sm + sa
    print("done salinity")

    ds = xr.Dataset({'MEASURED_TEMPERATURE': actual_temp, 'MEASURED_SALINITY': actual_sal})
    ds.to_netcdf("../datasets/RG_ArgoClim_Actual.nc")


def get_alpha(salinity, temperature, pressure, longitude, latitude):
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    CT = gsw.CT_from_t(SA, temperature, pressure)
    return gsw.alpha(SA, CT, pressure)

ds = ds.chunk({"TIME": -1, "PRESSURE": -1, "LATITUDE": 60, "LONGITUDE": 90})    # use dask arrays for considerably improved efficiency. hopeless without it (out of RAM on my machine)

alpha = xr.apply_ufunc(get_alpha, ds.MEASURED_SALINITY, ds.MEASURED_TEMPERATURE, ds.PRESSURE, ds.LONGITUDE, ds.LATITUDE, input_core_dims=[["PRESSURE"], ["PRESSURE"], ["PRESSURE"], [], []], output_core_dims=[["PRESSURE"]], vectorize=True, dask="parallelized")
alpha_integrate = alpha.sel(PRESSURE=slice(0, ref_pressure)).integrate("PRESSURE")

ssh = (ref_dym_pressure + alpha_integrate) / g
print("ssh obtained")
print(ssh)
ssh = ssh.rename("ssh")

with ProgressBar():     # lengthy. adjust chunk sizes as needed; bigger chunk gives shorter loading time.
    ssh.to_netcdf("../datasets/sea_surface_calculated.nc")
