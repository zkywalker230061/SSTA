import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from SSTA.Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, coriolis_parameter
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, compute_gradient_lon, compute_gradient_lat
import matplotlib

DOWNLOADED_SSH = False

WIND_STRESS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/wind_stress_interpolated.nc"
H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/hbar.nc"
CORIOLIS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/coriolis_parameter.nc"

rho_0 = 1025.0
c_0 = 4100.0
g = 9.81

wind_stress_ds = xr.open_dataset(WIND_STRESS_DATA_PATH, decode_times=False)

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

coriolis_parameter_ds = xr.open_dataset(CORIOLIS_DATA_PATH, decode_times=False)
coriolis_parameter_da = coriolis_parameter_ds["__xarray_dataarray_variable__"]

mean_ew_wind_stress = get_monthly_mean(wind_stress_ds["avg_iews"]).rename("mean_ew_wind_stress")
mean_ns_wind_stress = get_monthly_mean(wind_stress_ds["avg_inss"]).rename("mean_ns_wind_stress")
wind_stress_monthly_mean_ds = xr.Dataset({"mean_ew_wind_stress": mean_ew_wind_stress, "mean_ns_wind_stress": mean_ns_wind_stress})


def get_ekman_mean_advection(time, mean_ns_wind_stress, mean_ew_wind_stress, coriolis_parameter_da, hbar_da):
    alpha_contribution = -1 * mean_ns_wind_stress.sel(MONTH=month) / (coriolis_parameter_da * rho_0 * hbar_da.sel(MONTH=month))
    beta_contribution = -1 * mean_ew_wind_stress.sel(MONTH=month) / (coriolis_parameter_da * rho_0 * hbar_da.sel(MONTH=month))
    return [alpha_contribution, beta_contribution]

alphas = []
betas = []
for month in range(1, 13):
    alphabeta = get_ekman_mean_advection(month, mean_ns_wind_stress, mean_ew_wind_stress, coriolis_parameter_da.sel(TIME=0.5), hbar_da)
    alpha = alphabeta[0].reset_coords("TIME", drop=True)
    beta = alphabeta[1].reset_coords("TIME", drop=True)
    alphas.append(alpha)
    betas.append(beta)
alpha_contribution_da = xr.concat(alphas, "MONTH")
beta_contribution_da = xr.concat(betas, "MONTH")

alpha_contribution_da = alpha_contribution_da.where((alpha_contribution_da['LATITUDE'] > 5) | (alpha_contribution_da['LATITUDE'] < -5), 0)
beta_contribution_da = beta_contribution_da.where((beta_contribution_da['LATITUDE'] > 5) | (beta_contribution_da['LATITUDE'] < -5), 0)

alpha_grad_long_contribution_ds = compute_gradient_lon(alpha_contribution_da)
beta_grad_lat_contribution_ds = compute_gradient_lat(beta_contribution_da)

alpha_contribution_da.attrs["units"] = ""
beta_contribution_da.attrs["units"] = ""
alpha_grad_long_contribution_ds.attrs["units"] = ""
beta_grad_lat_contribution_ds.attrs["units"] = ""

ekman_mean_advection = xr.Dataset({"ekman_alpha": alpha_contribution_da, "ekman_beta": beta_contribution_da, "ekman_alpha_grad_long": alpha_grad_long_contribution_ds, "ekman_beta_grad_lat": beta_grad_lat_contribution_ds})
print(ekman_mean_advection)

ekman_mean_advection.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/ekman_mean_advection.nc")
