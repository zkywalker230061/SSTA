import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import dask
from dask.diagnostics import ProgressBar
import gc

from utils import load_and_prepare_dataset, make_movie

ALREADY_COMBINED = True
g = 9.81
ref_pressure = 950     # dbar (==m more or less); choose to be where horizontal gradient is 0
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
    ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Actual.nc")


def get_ssh_by_integrating_alpha(ds):
    ds = ds.chunk({"TIME": 30, "PRESSURE": -1, "LATITUDE": 60, "LONGITUDE": 60})    # use dask arrays for considerably improved efficiency. hopeless without it (out of RAM on my machine)

    SA = xr.apply_ufunc(gsw.SA_from_SP, ds.MEASURED_SALINITY, ds.PRESSURE, ds.LONGITUDE, ds.LATITUDE, dask="parallelized", output_dtypes=[float])
    CT = xr.apply_ufunc(gsw.CT_from_t, SA, ds.MEASURED_TEMPERATURE, ds.PRESSURE, dask="parallelized", output_dtypes=[float])

    print("computing SA/CT")
    with ProgressBar():
        SA = SA.compute()
        CT = CT.compute()

    print("computing reference SA/CT")
    with ProgressBar():
        ref_SA = SA.sel(PRESSURE=ref_pressure, method='nearest').mean(['TIME', 'LATITUDE', 'LONGITUDE']).compute()
        ref_CT = CT.sel(PRESSURE=ref_pressure, method='nearest').mean(['TIME', 'LATITUDE', 'LONGITUDE']).compute()

    SA = SA.chunk({"TIME": 30, "PRESSURE": -1, "LATITUDE": 60, "LONGITUDE": 60})
    CT = CT.chunk({"TIME": 30, "PRESSURE": -1, "LATITUDE": 60, "LONGITUDE": 60})

    beta = xr.apply_ufunc(gsw.beta, SA, CT, ds.PRESSURE, dask="parallelized", output_dtypes=[float])
    alpha = xr.apply_ufunc(gsw.alpha, SA, CT, ds.PRESSURE, dask="parallelized", output_dtypes=[float])

    print("computing alpha/beta")
    with ProgressBar():
        beta = beta.compute()
        alpha = alpha.compute()

    delta_S = SA - ref_SA
    delta_T = CT - ref_CT

    T_ssh_contribution = (alpha * delta_T).sel(PRESSURE=slice(0, ref_pressure)).integrate('PRESSURE') / g
    S_ssh_contribution = -(beta * delta_S).sel(PRESSURE=slice(0, ref_pressure)).integrate('PRESSURE') / g
    ssh = T_ssh_contribution + S_ssh_contribution

    T_ssh_contribution = T_ssh_contribution.rename("ssh_temperature_contribution")
    S_ssh_contribution = S_ssh_contribution.rename("ssh_salinity_contribution")
    ssh = ssh.rename("ssh")

    ssh_ds = xr.Dataset({"ssh_temperature_contribution": T_ssh_contribution, "ssh_salinity_contribution": S_ssh_contribution, "ssh": ssh})

    # alpha = xr.apply_ufunc(gsw.alpha, SA, CT, ds.PRESSURE, dask="parallelized")
    # alpha_integrate = alpha.sel(PRESSURE=slice(0, ref_pressure)).integrate("PRESSURE")
    #
    # ssh = (ref_dym_pressure + alpha_integrate) / g - ref_pressure
    # print("ssh obtained")
    # print(ssh)
    # ssh = ssh.rename("ssh")

    del SA, CT, alpha, beta, delta_T, delta_S       # delete older stuff before saving
    gc.collect()

    encoding = {var: {'zlib': True, 'complevel': 5} for var in ssh_ds.data_vars}    # split encoding from saving should prevent RAM overload... possibly. I'm not familiar with the nitty-gritty of RAM usage.

    print("saving")
    with ProgressBar():
        ssh_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated.nc", encoding=encoding, compute=True)

    # with ProgressBar():     # lengthy. adjust chunk sizes as needed; bigger chunk gives shorter loading time.
    #     ssh_ds.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated.nc")


def get_ssh_by_specific_volume_anomaly(ds):
    # note: why not use geo_strf_dyn_height? Tried it. It doesn't work because it requires uniform pressure levels over the entire space.
    max_measured_pressure = ds.PRESSURE.where(~np.isnan(ds.MEASURED_TEMPERATURE)).max("PRESSURE")
    pressure_mask = max_measured_pressure >= ref_pressure
    ds = ds.where(pressure_mask)

    ds = ds.chunk({"TIME": 30, "PRESSURE": -1, "LATITUDE": 60, "LONGITUDE": 90})    # use dask arrays for considerably improved efficiency. hopeless without it (out of RAM on my machine)

    SA = xr.apply_ufunc(gsw.SA_from_SP, ds.MEASURED_SALINITY, ds.PRESSURE, ds.LONGITUDE, ds.LATITUDE, dask="parallelized", output_dtypes=[float])
    CT = xr.apply_ufunc(gsw.CT_from_t, SA, ds.MEASURED_TEMPERATURE, ds.PRESSURE, dask="parallelized", output_dtypes=[float])

    valid_mask = (~np.isnan(SA)) & (~np.isnan(CT)) & (~np.isnan(ds.PRESSURE))
    SA = SA.where(valid_mask)
    CT = CT.where(valid_mask)

    geo_strf_dyn_height = xr.apply_ufunc(gsw.geo_strf_dyn_height, SA, CT, ds.PRESSURE, ref_pressure, dask="parallelized", output_dtypes=[float])

    print(geo_strf_dyn_height)
    ssh = geo_strf_dyn_height.isel(PRESSURE=0)
    ssh = ssh / g# - ref_pressure
    ssh = ssh.rename("ssh")

    with ProgressBar():
        ssh.compute()
    ssh.to_netcdf("../datasets/sea_surface_calculated_specific_volume_method.nc")

get_ssh_by_integrating_alpha(ds)
#get_ssh_by_specific_volume_anomaly(ds)
