import gsw
#import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import load_and_prepare_dataset, make_movie

ALREADY_COMBINED = True
g = 9.81
ref_pressure = 1000     # dbar (==m more or less); choose to be where horizontal gradient is 0
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


def get_alpha(ds, time):
    ds = ds.sel(TIME=time)
    SA = gsw.SA_from_SP(ds.MEASURED_SALINITY, ds.PRESSURE, ds.LONGITUDE, ds.LATITUDE)
    CT = gsw.CT_from_t(SA, ds.MEASURED_TEMPERATURE, ds.PRESSURE)
    alpha = gsw.alpha(SA, CT, ds.PRESSURE)
    return alpha

alpha_list = []
for time in ds.TIME.values:
    print(time)
    alpha = get_alpha(ds, time)
    alpha_list.append(alpha)
ds = None
alpha_ds = xr.concat(alpha_list, "TIME")
print("concatenate done")
#print(alpha_ds)

alpha_integrate = alpha_ds.sel(PRESSURE=slice(0, ref_pressure)).integrate("PRESSURE")
print("integrate done")
#print(alpha_integrate)

ssh = ref_dym_pressure + alpha_integrate
print("ssh obtained")
#print(ssh)

ssh.to_netcdf("../datasets/sea_surface_calculated.nc")
