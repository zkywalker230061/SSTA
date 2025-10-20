"""
Julia Xie, JY
"""

from read_nc import fix_rg_time
from calculate_Tm_Sm import depth_from_pressure, _full_field, z_to_xarray, vertical_integral
import xarray as xr
import matplotlib.pyplot as plt


#---1. -------------------------------------------
# ds_temp = xr.open_dataset(
#     "/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc",
#     engine="netcdf4",
#     decode_times=False,
#     mask_and_scale=True,
# )

ds_sal = xr.open_dataset(
    "C:\Msci Project\RG_ArgoClim_Salinity_2019.nc",
    engine="netcdf4",
    decode_times=False,
    mask_and_scale=True,
)

# ds_temp = fix_rg_time(ds_temp)
ds_sal = fix_rg_time(ds_sal)

S_mean = ds_sal["ARGO_SALINITY_MEAN"]          # (P, Y, X)
S_anom = ds_sal["ARGO_SALINITY_ANOMALY"]
S_full = _full_field(S_mean, S_anom)

#----2. gsw--------------------------------------------
p = ds_sal['PRESSURE']
lat = ds_sal['LATITUDE']
depth = depth_from_pressure(p,lat)

#-----3. z to x array-----------------------------------------
z_new = z_to_xarray(depth, S_full)
#print(z_new)

#----4. Vertical Integration -------------------------------
vertical = vertical_integral(S_full,-z_new)          #??????i changed here to -z_new





if __name__ == "__main__":

    print(vertical)
    print(vertical.shape)


    #----Plot Map----------------------------------------------------
    t0 = vertical.sel(TIME="2006-01-01")

    plt.figure(figsize=(10,5))
    pc = plt.pcolormesh(
        t0["LONGITUDE"], t0["LATITUDE"], t0,
        cmap="RdYlBu_r", shading="auto", vmin=30, vmax=38
    )
    plt.colorbar(pc, label="Mean Salinity (Scale 78)")
    plt.title("Upper 100 m Mean Salinity - Jan 2006")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    #vertical.to_netcdf("Mean Salinity Dataset (2004-2018)")