import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset, get_eof

TEMP_DATA_PATH = "../datasets/RG_ArgoClim_Temperature_2019.nc"
MLD_DATA_PATH = "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc"

temp_ds = load_and_prepare_dataset(TEMP_DATA_PATH)
mld_ds = xr.open_dataset(MLD_DATA_PATH, decode_times=False)

# get actual temperature by combining mean with anomaly from Argo
tm = temp_ds["ARGO_TEMPERATURE_MEAN"]
ta = temp_ds["ARGO_TEMPERATURE_ANOMALY"]
tm = tm.expand_dims(TIME=180)
temp_ds["TEMPERATURE"] = tm + ta


def get_T_sub(temp_ds, mld_ds, month, make_plots=True):
    temp_ds = temp_ds.sel(TIME=month)
    mld_ds = mld_ds.sel(TIME=month)

    def interpolation(depth, temperature, mld):
        above_mld = np.where(depth <= mld)[0]
        below_mld = np.where(depth > mld)[0]

        # catch case where no MLD (should be only over land)
        if len(above_mld) == 0 or len(below_mld) == 0:
            return np.nan

        above_mld_index = above_mld[-1]
        below_mld_index = below_mld[0]

        return np.interp(mld, [depth[below_mld_index], depth[above_mld_index]],
                         [temperature[below_mld_index], temperature[above_mld_index]])

    T_sub = xr.apply_ufunc(interpolation, temp_ds['PRESSURE'], temp_ds['TEMPERATURE'], mld_ds['MLD_PRESSURE'],
                           input_core_dims=[['PRESSURE'], ['PRESSURE'], []], vectorize=True)
    mld_ds['T_sub'] = T_sub
    if make_plots:
        mld_ds['T_sub'].plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r')
        plt.show()
    return mld_ds


# combine dataset from each of the months
datasets_over_time = []
for month in temp_ds.TIME.values:
    datasets_over_time.append(get_T_sub(temp_ds, mld_ds, month, make_plots=False))
t_sub_ds = xr.concat(datasets_over_time, "TIME")
t_sub_ds = t_sub_ds.drop_vars("MLD_PRESSURE")

# get mean and anomaly
t_sub_monthly_mean = get_monthly_mean(t_sub_ds['T_sub'])
t_sub_ds = get_anomaly(t_sub_ds, 'T_sub', t_sub_monthly_mean)

t_sub_ds.to_netcdf("../datasets/t_sub.nc")
print(t_sub_ds)
print(t_sub_ds["T_sub_ANOMALY"].max().item())
print(t_sub_ds["T_sub_ANOMALY"].min().item())
print(abs(t_sub_ds["T_sub_ANOMALY"]).mean().item())

# get eof
t_sub_anomaly_eof = get_eof(t_sub_ds["T_sub_ANOMALY"], 10, "TIME")
print(t_sub_anomaly_eof)
print(t_sub_anomaly_eof.components())
