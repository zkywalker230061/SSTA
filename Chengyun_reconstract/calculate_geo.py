"""
Calculate Geostrophic term.

Chengyun Zhu
2026-1-12
"""

import xarray as xr
import numpy as np
import gsw

from utilities import load_and_prepare_dataset
# from utilities import get_monthly_mean, get_anomaly
from utilities import save_file

g = 9.81
ref_pressure = 1000     # dbar (==m more or less); choose to be where horizontal gradient is 0
ref_dym_pressure = g * ref_pressure


def get_alpha(
    salinity, temperature, pressure, longitude, latitude
):
    sa = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    ct = gsw.CT_from_t(sa, temperature, pressure)
    return gsw.alpha(sa, ct, pressure)


t = load_and_prepare_dataset("datasets/Temperature-(2004-2018).nc")
s = load_and_prepare_dataset("datasets/Salinity-(2004-2018).nc")

alpha = xr.apply_ufunc(
    get_alpha,
    s.SALINITY, t.TEMPERATURE, s.PRESSURE, s.LONGITUDE, s.LATITUDE,
    input_core_dims=[["PRESSURE"], ["PRESSURE"], ["PRESSURE"], [], []],
    output_core_dims=[["PRESSURE"]], vectorize=True
)
alpha_integrate = alpha.sel(PRESSURE=slice(0, ref_pressure)).integrate("PRESSURE")

ssh = (ref_dym_pressure + alpha_integrate) / g
print("ssh obtained")
print(ssh)
ssh = ssh.rename("ssh")

# ssh.to_netcdf("../datasets/sea_surface_calculated.nc")
