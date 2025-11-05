import gsw
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from read_nc import get_monthly_mean, get_anomaly, load_and_prepare_dataset, load_pressure_data
from matplotlib.animation import FuncAnimation
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

# Set backend
matplotlib.use('TkAgg')

# --- File Paths (assuming these are correct) -------------------------------
MLD_TEMP_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature-(2004-2018).nc"
MLD_DEPTH_PATH = "/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
HEAT_FLUX_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/ERA5-ARGO_Mean_Surface_Heat_Flux.nc"
EK_DATA_PATH = "/Users/julia/Desktop/SSTA/datasets/Ekman_Current_Anomaly.nc"

# --- Load and Prepare Data (assuming helper functions are correct) --------
mld_temperature_ds = xr.open_dataset(MLD_TEMP_PATH, decode_times=False)
mld_depth_ds = load_pressure_data(MLD_DEPTH_PATH, 'MONTHLY_MEAN_MLD_PRESSURE')
heat_flux_ds = load_and_prepare_dataset(HEAT_FLUX_DATA_PATH)
ekman_ds = load_and_prepare_dataset(EK_DATA_PATH)

temperature = mld_temperature_ds['__xarray_dataarray_variable__']
temperature_monthly_mean = get_monthly_mean(temperature)
temperature_anomaly = get_anomaly(temperature, temperature_monthly_mean)

mld_depth_ds = mld_depth_ds.rename({'MONTH': 'TIME'})  # TIME: 1, 2, ..., 12

heat_flux = (heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_ishf'] +
             heat_flux_ds['avg_snswrf'] + heat_flux_ds['avg_snlwrf'])
heat_flux.attrs.update(units='W m**-2', long_name='Net Surface Heat Flux')
heat_flux.name = 'NET_HEAT_FLUX'
heat_flux_monthly_mean = get_monthly_mean(heat_flux)
heat_flux_anomaly = get_anomaly(heat_flux, heat_flux_monthly_mean)
heat_flux_anomaly = heat_flux_anomaly.drop_vars(['MONTH'])

ekman_anomaly = ekman_ds['Q_Ek_anom']

# --- Model Constants ------------------------------------------------------
RHO_O = 1025.0  # kg/m^3
C_O = 4100.0  # J/(kg K)
SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # s
GAMMA = 10.0  # bulk damping factor

# Get time coordinates
times = temperature_anomaly.TIME.values
t0 = times[0]

# --- Helper to create efficient output arrays -----------------------------
def create_output_array(name, long_name):
    """Pre-allocates an xr.DataArray with the correct coords."""
    return xr.DataArray(
        np.nan,
        coords=temperature_anomaly.coords,
        dims=temperature_anomaly.dims,
        name=name,
        attrs={**temperature_anomaly.attrs,
               'units': 'K',
               'long_name': long_name}
    )

# --- Model Computation Functions ------------------------------------------

def compute_implicit() -> xr.DataArray:
    """Runs the implicit scheme using your logic."""
    print("Running Implicit Scheme...")
    model_anomaly_ds = create_output_array(
        "T_model_anom_implicit",
        "Mixed-layer temperature anomaly (implicit)"
    )

    # Initial condition (zero anomaly)
    T_prev = xr.zeros_like(temperature_anomaly.isel(TIME=0))
    model_anomaly_ds.loc[dict(TIME=t0)] = T_prev

    # Loop from the second time step
    for i, month in enumerate(times[1:], start=1):
        # 'month' is t^{n+1} (e.g., 1.5, 2.5, ...)

        # Get coefficients for time n+1
        # Your logic: for t=1.5, moy=1.5. sel(1.5, nearest) -> 2.0 (Feb MLD)
        moy_n_plus_1 = ((month - 0.5) % 12) + 0.5

        denominator = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=moy_n_plus_1, method='nearest', tolerance=0.51)
        )
        # a^{n+1}
        damp_factor = GAMMA / denominator

        # b^{n+1}
        forcing_term = (
            heat_flux_anomaly.sel(TIME=month, method='nearest', tolerance=0.51) +
            ekman_anomaly.sel(TIME=month, method='nearest', tolerance=0.51)
        ) / denominator

        # T^{n+1} = (T^n + dt*b^{n+1}) / (1 + dt*a^{n+1})
        T_next = (T_prev + SECONDS_MONTH * forcing_term) / (1 + SECONDS_MONTH * damp_factor)

        # Store result and update T_prev
        model_anomaly_ds.loc[dict(TIME=month)] = T_next
        T_prev = T_next

    return model_anomaly_ds


def compute_explicit() -> xr.DataArray:
    """Runs the explicit scheme using your logic (with bug fix)."""
    print("Running Explicit Scheme...")
    model_anomaly_ds = create_output_array(
        "T_model_anom_explicit",
        "Mixed-layer temperature anomaly (explicit)"
    )

    # Initial condition (zero anomaly)
    T_prev = xr.zeros_like(temperature_anomaly.isel(TIME=0))
    model_anomaly_ds.loc[dict(TIME=t0)] = T_prev

    # Loop from the second time step
    for i, month in enumerate(times[1:], start=1):
        # 'month' is t^{n+1} (e.g., 1.5, 2.5, ...)
        prev_time = times[i - 1]  # 'prev_time' is t^n (e.g., 0.5, 1.5, ...)

        # Get coefficients for time n
        # Your logic: for t=0.5, moy=0.5. sel(0.5, nearest) -> 1.0 (Jan MLD)
        moy_n = ((prev_time - 0.5) % 12) + 0.5

        denominator = (
            RHO_O * C_O *
            mld_depth_ds
            .sel(TIME=moy_n, method='nearest', tolerance=0.51))
        # a^n
        damp_factor = GAMMA / denominator

        # b^n
        # *** LOGIC FIX ***: Use prev_time, not moy_n, to select forcing data.
        forcing_term = (
            heat_flux_anomaly.sel(TIME=prev_time, method='nearest', tolerance=0.51) +
            ekman_anomaly.sel(TIME=prev_time, method='nearest', tolerance=0.51)
        ) / denominator

        # T^{n+1} = (1 - dt*a^n)T^n + dt*b^n
        T_next = (1 - SECONDS_MONTH * damp_factor) * T_prev + SECONDS_MONTH * forcing_term

        # Store result and update T_prev
        model_anomaly_ds.loc[dict(TIME=month)] = T_next
        T_prev = T_next

    return model_anomaly_ds

# --- Run both simulations -------------------------------------------------
sim_implicit = compute_implicit()
sim_explicit = compute_explicit()

# --- Side-by-side animation -------------------------------------------------
print("Starting animation...")
# Use the time grid from one of the simulations
anim_times = sim_implicit.TIME.values

# Common lon/lat
lons = sim_implicit.LONGITUDE.values
lats = sim_implicit.LATITUDE.values

VMIN, VMAX = -10,10

fig = plt.figure(figsize=(16, 6))

# Explicit panel
ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
mesh_exp = ax1.pcolormesh(lons, lats, sim_explicit.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax1.coastlines()
ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl1.top_labels = False
gl1.right_labels = False
gl1.xlines = False
gl1.ylines = False
gl1.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl1.ylocator = LatitudeLocator()
gl1.xformatter = LongitudeFormatter()
gl1.yformatter = LatitudeFormatter()
gl1.ylabel_style = {'size': 12, 'color': 'gray'}
gl1.xlabel_style = {'size': 12, 'color': 'gray'}
ax1.set_title(f'Explicit — t={anim_times[0]}')

# Implicit panel
ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
mesh_imp = ax2.pcolormesh(lons, lats, sim_implicit.isel(TIME=0),
                          cmap='RdBu_r', vmin=VMIN, vmax=VMAX)
ax2.coastlines()
ax2.set_xlim(-180, 180)
ax2.set_ylim(-90, 90)
gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl2.top_labels = False
gl2.right_labels = True
gl2.left_labels = False
gl2.xlines = False
gl2.ylines = False
gl2.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl2.ylocator = LatitudeLocator()
gl2.xformatter = LongitudeFormatter()
gl2.yformatter = LatitudeFormatter()
gl2.ylabel_style = {'size': 12, 'color': 'gray'}
gl2.xlabel_style = {'size': 12, 'color': 'gray'}
ax2.set_title(f'Implicit — t={anim_times[0]}')

# Shared colorbar
cbar = fig.colorbar(mesh_imp, ax=[ax1, ax2], shrink=0.8,
                    label=sim_implicit.attrs.get('units', 'K'))

def update(frame):
    # Update explicit
    Z_exp = sim_explicit.isel(TIME=frame).values
    mesh_exp.set_array(Z_exp.ravel())

    # Update implicit
    Z_imp = sim_implicit.isel(TIME=frame).values
    mesh_imp.set_array(Z_imp.ravel())

    # Update titles
    current_time = anim_times[frame]
    month_in_year = (current_time % 12) + 0.5  # 0.5 (Jan) to 11.5 (Dec)
    ax1.set_title(f'Explicit — Time: {current_time} (Month: {month_in_year:.1f})')
    ax2.set_title(f'Implicit — Time: {current_time} (Month: {month_in_year:.1f})')
    return [mesh_exp, mesh_imp]

# Create and show animation
animation = FuncAnimation(fig, update, frames=len(anim_times), interval=300, blit=False)
# plt.tight_layout()
plt.show()


#%%
#--CHENGYUN---------------------------------------------------------------------------------------------------------
# RHO_O = 1025  # kg / m^3
# C_O = 4100  # J / (kg K)
# SECONDS_MONTH = 30.4375 * 24 * 60 * 60  # average seconds in a month
# GAMMA = 10


# MONTH = 0.5
# for month in temperature.TIME.values:
#     if month == MONTH:
#         model_anomaly_ds = (
#             temperature_anomaly.sel(TIME=MONTH)
#             - temperature_anomaly.sel(TIME=MONTH)
#         )
#         model_anomaly_ds = model_anomaly_ds.expand_dims(TIME=[MONTH])
#     else:
#         prev = model_anomaly_ds.sel(TIME=month-1)
#         cur = (
#             (1 - (GAMMA / (RHO_O * C_O * mld_depth_ds.sel(TIME=(month % 12 + 0.5))['MONTHLY_MEAN_MLD_PRESSURE']))) * prev
#             + SECONDS_MONTH * (
#                 heat_flux_anomaly.sel(TIME=month)
#                 # + ENT
#                 # + GEO
#                 + ekman_anomaly.sel(TIME=month)
#             ) / (RHO_O * C_O * mld_depth_ds.sel(TIME=(month % 12 + 0.5))['MONTHLY_MEAN_MLD_PRESSURE'])
#         )
#         cur = cur.expand_dims(TIME=[month])
#         model_anomaly_ds = xr.concat([model_anomaly_ds, cur], dim='TIME')

# # model_anomaly_ds.sel(TIME=100.5).plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r')
# # plt.show()

# # make a movie
# times = model_anomaly_ds.TIME.values

# fig, ax = plt.subplots(figsize=(12, 6))

# ax = plt.axes(projection=ccrs.PlateCarree())

# pcolormesh = ax.pcolormesh(
#     model_anomaly_ds.LONGITUDE.values,
#     model_anomaly_ds.LATITUDE.values,
#     model_anomaly_ds.isel(TIME=0),
#     cmap='RdBu_r',
#     vmin=-20, vmax=20
# )
# # contourf = ax.contourf(
# #     model_anomaly_ds.LONGITUDE.values,
# #     model_anomaly_ds.LATITUDE.values,
# #     model_anomaly_ds.isel(TIME=0),
# #     cmap='RdBu_r',
# #     levels=200,
# #     vmin=-20, vmax=20
# # )
# ax.coastlines()
# ax.set_xlim(-180, 180)
# ax.set_ylim(-90, 90)

# gl = ax.gridlines(
#     crs=ccrs.PlateCarree(), draw_labels=True,
#     linewidth=2, color='gray', alpha=0.5, linestyle='--'
#     )
# gl.top_labels = False
# gl.left_labels = True
# gl.right_labels = False
# gl.xlines = False
# gl.ylines = False
# gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
# gl.ylocator = LatitudeLocator()
# gl.xformatter = LongitudeFormatter()
# gl.yformatter = LatitudeFormatter()
# gl.ylabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'size': 15, 'color': 'gray'}

# cbar = plt.colorbar(pcolormesh, ax=ax, label=model_anomaly_ds.attrs.get('units'))
# # cbar = plt.colorbar(contourf, ax=ax, label=model_anomaly_ds.attrs.get('units'))

# title = ax.set_title(f'Time = {times[0]}')


# def update(frame):
#     pcolormesh.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
#     cbar.update_normal(pcolormesh)
#     title.set_text(
#         f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}'
#     )
#     return [pcolormesh, title]
#     # contourf.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
#     # cbar.update_normal(contourf)
#     # contourf = ax.contourf(
#     #     model_anomaly_ds.LONGITUDE.values,
#     #     model_anomaly_ds.LATITUDE.values,
#     #     model_anomaly_ds.isel(TIME=frame),
#     #     cmap='RdBu_r',
#     #     levels=200,
#     #     vmin=-20, vmax=20
#     # )
#     # title.set_text(
#     #     f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}'
#     # )
#     # return [contourf, title]


# animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
# plt.show()



# #%%
# #--Chris Code----------------------------------------------
# model_anomalies = []
# added_baseline = False
# for month in heat_flux_anomaly_ds.TIME.values:
#     if not added_baseline:
#         base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
#         base = base.expand_dims(TIME=[month])  # <-- keep its time
#         model_anomalies.append(base)
#         added_baseline = True
#     else:
#         prev = model_anomalies[-1].isel(TIME=-1)
#         cur = prev + (30.4375 * 24 * 60 * 60) * heat_flux_anomaly_ds.sel(TIME=month)['NET_HEAT_FLUX_ANOMALY'] / (mld_ds.sel(TIME=month)['MLD_PRESSURE'] * 1025 * 4100)
#         cur = cur.expand_dims(TIME=[month])
#         model_anomalies.append(cur)
# model_anomaly_ds = xr.concat(model_anomalies, 'TIME')
# model_anomaly_ds = model_anomaly_ds.drop_vars(["PRESSURE"])
# print(model_anomaly_ds)

# times = model_anomaly_ds.TIME.values

# fig, ax = plt.subplots()
# pcolormesh = ax.pcolormesh(model_anomaly_ds.LONGITUDE.values, model_anomaly_ds.LATITUDE.values, model_anomaly_ds.isel(TIME=0), cmap='RdBu_r')
# title = ax.set_title(f'Time = {times[0]}')

# cbar = plt.colorbar(pcolormesh, ax=ax, label='Modelled anomaly from surface heat flux')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')


# def update(frame):
#     pcolormesh.set_array(model_anomaly_ds.isel(TIME=frame).values.ravel())
#     #pcolormesh.set_clim(vmin=float(model_anomaly_ds.isel(TIME=frame).min()), vmax=float(model_anomaly_ds.isel(TIME=frame).max()))
#     pcolormesh.set_clim(vmin=-20, vmax=20)
#     cbar.update_normal(pcolormesh)
#     title.set_text(f'Months since January 2004: {times[frame]}; month in year: {(times[frame] + 0.5) % 12}')
#     return [pcolormesh, title]

# animation = FuncAnimation(fig, update, frames=len(times), interval=300, blit=False)
# plt.show()

