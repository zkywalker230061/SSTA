import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from utils_read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)


#---1. Read Files ----------------------------------------------------------
semi_implicit_path = "/Users/julia/Desktop/SSTA/datasets/Semi_Implicit_Scheme_Test_ConstDamp(10)"
implicit_path = "/Users/julia/Desktop/SSTA/datasets/Implicit_Scheme_Test_ConstDamp(10)"
explicit_path = "/Users/julia/Desktop/SSTA/datasets/Explicit_Scheme_Test_ConstDamp(10)"
crank_path = "/Users/julia/Desktop/SSTA/datasets/Crack_Scheme_Test_ConstDamp(10)"
observed_path = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature(T_m).nc"
EK_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Current_Anomaly.nc"
HEAT_FLUX_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc"

# --- Load and Prepare Data (assuming helper functions are correct) --------
observed_temp = xr.open_dataset(observed_path, decode_times=False)
# implicit = load_and_prepare_dataset(implicit_path)
# explicit = load_and_prepare_dataset(explicit_path)
# crank = load_and_prepare_dataset(crank_path)
# semi_implicit = load_and_prepare_dataset(semi_implicit_path)
heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_DATA_PATH)
ekman_anom_ds = load_and_prepare_dataset(EK_DATA_PATH)


# --- Extracting the correct DataArray -------------------
temperature = observed_temp['__xarray_dataarray_variable__']
# implicit = implicit["T_model_anom_implicit"]
# explicit = explicit["T_model_anom_explicit"]
# crank = crank["T_model_anom_crank_nicolson"]
# semi_implicit = semi_implicit["T_model_anom_semi_implicit"]




#---- Defining the Anomaly Dataset ----------------------
heat_flux = (heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_ishf'] +
             heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_snlwrf'])
heat_flux.attrs.update(units='W m**-2', long_name='Net Surface Heat Flux')
heat_flux.name = 'NET_HEAT_FLUX'
heat_flux_monthly_mean = get_monthly_mean(heat_flux)
heat_flux_anomaly = get_anomaly(heat_flux, heat_flux_monthly_mean)
heat_flux_anomaly = heat_flux_anomaly.drop_vars(['MONTH'])
heat_flux_anomaly = xr.where(heat_flux_anomaly == 0, np.nan, heat_flux_anomaly)

temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

ekman_anomaly = ekman_anom_ds['Q_Ek_anom']


times = temperature.TIME.values
t0 = times[0]



heat_flux_anom_std = heat_flux_anomaly.std(dim='TIME', skipna=True, keep_attrs=True)
heat_flux_anom_std.name = "heat_flux_anom_std"
heat_flux_anom_std.attrs.update({"long_name": "heat flux anomaly standard deviation"})
ekman_anom_std = ekman_anomaly.std(dim='TIME', skipna=True, keep_attrs=True)
ekman_anom_std.name = "ekman_anom_std"
ekman_anom_std.attrs.update({"long_name": "ekman current anomaly standard deviation"})



def month_idx (time_da: xr.DataArray) -> xr.DataArray:
    n = time_da.sizes['TIME']
    # Repeat 1..12 along TIME; align coords to TIME
    month_idx = (xr.DataArray(np.arange(n) % 12 + 1, dims=['TIME'])
                 .assign_coords(TIME=time_da))
    month_idx.name = 'MONTH'
    return month_idx

def monthly_std(ds):
    if 'TIME' not in ds.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    m = month_idx(ds['TIME'])
    monthly_anom_std = ds.groupby(m).std(dim='TIME', skipna=True, keep_attrs=True)
    return monthly_anom_std

monthly_heat_flux_anom_std = monthly_std(heat_flux_anomaly)
monthly_heat_flux_anom_std.name = "monthly_heat_flux_anom_std"
monthly_heat_flux_anom_std.attrs.update({"long_name": "monthly averaged heat flux anomaly standard deviation"})
monthly_ekman_anom_std = monthly_std(ekman_anomaly)
monthly_ekman_anom_std.name = "ekman_amonthly_ekman_anom_stdnom_std"
monthly_ekman_anom_std.attrs.update({"long_name": "monthly averaged ekman current anomaly standard deviation"})

# print('ekman_anom_std', ekman_anom_std)
#print('heat_flux_anomly', ekman_anomaly.isnull().sum().item())

print('monthly_ekman_anom_std',monthly_ekman_anom_std)
print('monthly_heat_flux_anom_std',monthly_heat_flux_anom_std)

VMIN, VMAX = 0,45
#---------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=[16,6])

ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
pc = plt.pcolormesh(
        ekman_anom_std["LONGITUDE"], ekman_anom_std["LATITUDE"], ekman_anom_std,
        cmap='RdBu_r', shading="auto", vmin=VMIN, vmax=VMAX
    )
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.set_title("Ekman Current Anomaly Standard Deviation")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.coastlines()


ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
pc2 = plt.pcolormesh(
        heat_flux_anom_std["LONGITUDE"], heat_flux_anom_std["LATITUDE"], heat_flux_anom_std,
        cmap='RdBu_r', shading="auto", vmin=VMIN, vmax=VMAX
    )
cbar2 = fig.colorbar(pc2, ax=[ax,ax2], orientation="vertical", label="Standard Deviation", shrink = 0.7)
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax2.set_title("Air Sea Heat Flux Anomaly Standard Deviation")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.coastlines()
plt.savefig('Anomaly Standard Deviation')
plt.show()


#--------------------------------------------------------------------------------------------------------------------------------------------
anim_times = monthly_heat_flux_anom_std.MONTH.values

# Common lon/lat
lons = monthly_heat_flux_anom_std.LONGITUDE.values
lats = monthly_heat_flux_anom_std.LATITUDE.values

fig2 = plt.figure(figsize=(16, 6))

# Chris panel
ax3 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
mesh_ekman = ax3.pcolormesh(lons, lats, monthly_ekman_anom_std.isel(MONTH=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax3.coastlines()
ax3.set_xlim(-180, 180)
ax3.set_ylim(-90, 90)
gl3 = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl3.top_labels = False
gl3.right_labels = False
gl3.xlines = False
gl3.ylines = False
gl3.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl3.ylocator = LatitudeLocator()
gl3.xformatter = LongitudeFormatter()
gl3.yformatter = LatitudeFormatter()
gl3.ylabel_style = {'size': 12, 'color': 'gray'}
gl3.xlabel_style = {'size': 12, 'color': 'gray'}
ax3.set_title(f'monthly_ekman_current_anom_std — t={anim_times[0]}')


ax4 = plt.subplot(1,2,2, projection = ccrs.PlateCarree())
mesh_flux = ax4.pcolormesh(lons, lats, monthly_heat_flux_anom_std.isel(MONTH=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax4.coastlines()
ax4.set_xlim(-180, 180)
ax4.set_ylim(-90, 90)
gl4 = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl4.top_labels = False
gl4.right_labels = False
gl4.xlines = False
gl4.ylines = False
gl4.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl4.ylocator = LatitudeLocator()
gl4.xformatter = LongitudeFormatter()
gl4.yformatter = LatitudeFormatter()
gl4.ylabel_style = {'size': 12, 'color': 'gray'}
gl4.xlabel_style = {'size': 12, 'color': 'gray'}
ax4.set_title(f'monthly_heat_flux_std — t={anim_times[0]}')

cbar = fig2.colorbar(mesh_flux, ax=[ax3,ax4], shrink=0.8, label=('standard deviation'))


def update(frame):
    Z_ekman = monthly_ekman_anom_std.isel(MONTH=frame).values
    mesh_ekman.set_array(Z_ekman.ravel())

    Z_flux = monthly_heat_flux_anom_std.isel(MONTH=frame).values
    mesh_flux.set_array(Z_flux.ravel())
   
    # Update titles
    current_time = anim_times[frame]
    # month_in_year = (current_time % 12) + 0.5  # 0.5 (Jan) to 11.5 (Dec)
    ax3.set_title(f'Monthly Averaged Ekman Current Anomaly std — MONTH: {current_time}')
    ax4.set_title(f'Monthly Averaged Heat Flux std — MONTH: {current_time}')
    return [mesh_ekman, mesh_flux]

animation = FuncAnimation(fig2, update, frames=len(anim_times), interval=600, blit=False)
# animation.save('Climatology Averaged Standard Deviation.mp4', writer='ffmpeg', fps=2)
plt.show()

mesh_ekman.get_array().size

