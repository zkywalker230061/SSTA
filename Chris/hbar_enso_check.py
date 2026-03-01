import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie

H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Mixed_Layer_Depth_Pressure_uncapped-Seasonal_Cycle_Mean.nc"
T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"

hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
hbar_da = hbar_ds['MONTHLY_MEAN_MLD_PRESSURE']

tsub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
tsub_da = tsub_ds["T_sub"]
tsub_anomaly_da = tsub_ds["T_sub_ANOMALY"]

make_movie(tsub_da, 0, 40)
make_movie(tsub_anomaly_da, -3, 3)

"""conclude: ENSO effects are shown in tsub anomaly. BUT it will only be of importance when entrainment matters"""

# for month in range(1, 13):
#     pcolormesh = plt.pcolormesh(hbar_da.LONGITUDE.values, hbar_da.LATITUDE.values, hbar_da.sel(MONTH=month), cmap='Blues')
#     cbar = plt.colorbar(pcolormesh, label="Temperature per PC from Regression (K)")
#     pcolormesh.set_clim(vmin=0, vmax=100)
#     plt.show()

# print(hbar_ds.max().items)
# print(hbar_ds.min().items)
#
# make_movie(hbar_ds, -10, 10)

