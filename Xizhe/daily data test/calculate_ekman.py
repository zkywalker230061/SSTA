#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)

c_o = 4100                         #specific heat capacity of seawater = 4100 Jkg^-1K^-1
omega = 2*np.pi/(24*3600)         #Earth's angular velocity
#f = 2*omega*np.sin(phi)            #Coriolis Parameter


def ekman_current_anomaly(tau_x_anom, tau_y_anom, dTm_dx_monthly, dTm_dy_monthly, f_2d, fmin=1e-5):
    """
    Compute Q'_Ek = c_o * (τx'/xf * dTmbar/dy - τy'/f * dTmbar/dx)
    Output dims: (TIME, LATITUDE, LONGITUDE)
    """
    # Create month index per TIME
    m = month_idx(tau_x_anom)

    # Select gradient of the corresponding month for each TIME (aligns MONTH -> TIME)
    dTm_dx_t = dTm_dx_monthly.sel(MONTH=m['MONTH'])
    dTm_dy_t = dTm_dy_monthly.sel(MONTH=m['MONTH'])

    # Broadcast Coriolis parameter
    f = f_2d.broadcast_like(tau_x_anom)
    mask = np.abs(f) > fmin  # avoid division near equator

    # Compute Q'_Ek
    Q_ek = c_o * ((tau_x_anom * dTm_dy_t / f) - (tau_y_anom * dTm_dx_t / f))
    Q_ek_x = c_o * (tau_x_anom * dTm_dy_t / f)
    Q_ek_y = c_o * (tau_y_anom * dTm_dx_t / f)
    Q_ek = Q_ek.where(mask)
    Q_ek_x = Q_ek_x.where(mask)
    Q_ek_y = Q_ek_y.where(mask)

    Q_ek.name = "Q_Ek_anom"
    Q_ek.attrs.update({
        "description": "Ekman term anomaly using τ' and monthly mean Tm gradients",
        "formula": "c_o * (τx'/f * dTmbar/dy - τy'/f * dTmbar/dx)",
    })
    return Q_ek, Q_ek_x, Q_ek_y

def month_idx (da: xr.DataArray) -> xr.DataArray:
    """
    Creates a new non-dimensional coordinate 'MONTH' (1-12) for a 366-day time axis.
    """
    month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if da.sizes["TIME"] != sum(month_lengths):
        raise ValueError("TIME length is not 366; check your data or month_lengths.")
    
    month_index = np.repeat(np.arange(1, 13), month_lengths)
    da = da.assign_coords(MONTH=("TIME", month_index))
    return da

def get_monthly_mean(da: xr.DataArray,) -> xr.DataArray:
    if 'TIME' not in da.dims:
        raise ValueError("The DataArray must have a TIME dimension.")
    
    m = month_idx(da['TIME'])
    monthly_mean_da = da.groupby(m).mean('TIME', keep_attrs=True)
    return monthly_mean_da

def get_anomaly(full_field, monthly_mean):
    """
    Calculates the anomaly of a full time-series DataArray
    by subtracting its corresponding monthly mean.
    """
    if 'TIME' not in full_field.dims:
        raise ValueError("The full_field DataArray must have a TIME dimension.")
    
    # Get the month index (1-12) for each item in the full_field
    m = month_idx(full_field)
    anom = full_field.groupby(m['MONTH']) - monthly_mean
    return anom

def coriolis_parameter(lat):
    phi_rad = np.deg2rad(lat)
    f = 2 * omega * np.sin(phi_rad)
    f = xr.DataArray(f, coords={'LATITUDE': lat}, dims=['LATITUDE'])
    f.attrs['units'] = 's^-1'
    return f

def repeat_monthly_field(ds, var_name, n_repeats=15):
    """
    Take a dataset with a monthly 3D field (MONTH, LATITUDE, LONGITUDE)
    and repeat it n_repeats times along the MONTH axis to create a new
    time-like dimension of length 12 * n_repeats.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset. Must contain:
        - coord "MONTH" of length 12
        - coord "LATITUDE"
        - coord "LONGITUDE"
        - data variable `var_name` with dims ("MONTH","LATITUDE","LONGITUDE")
    var_name : str
        Name of the data variable to tile, e.g. "MONTHLY_MEAN_MLD_PRESSURE".
    n_repeats : int, default 15
        How many times to repeat the 12-month cycle.

    Returns
    -------
    xarray.Dataset
        Dataset with:
        - new dim "TIME" of length 12 * n_repeats
        - coords "TIME", "LATITUDE", "LONGITUDE"
        - data variable renamed to the same var_name but
          now on ("TIME","LATITUDE","LONGITUDE")
    """

    month_vals = ds["MONTH"].values  # e.g. [1,2,...,12]

    time_coord = np.tile(month_vals, n_repeats).astype(float)

    for i in range(len(time_coord)):
        time_coord[i] = time_coord[i] + (i // 12) * 12

    time_coord = time_coord - 0.5  # length = 12 * n_repeats

    data_var = ds.values  # shape (12, lat, lon)
    data_tiled = np.tile(data_var, (n_repeats, 1, 1))

    out = xr.Dataset(
        {
            var_name: (
                ("TIME", "LATITUDE", "LONGITUDE"),
                data_tiled,
            )
        },
        coords={
            "TIME": time_coord,
            "LATITUDE": ds["LATITUDE"].values,
            "LONGITUDE": ds["LONGITUDE"].values,
        },
    )
    return out

def tile_monthly_to_daily(gradient_monthly, tile_array):
    daily_list = []
    for i in range(12):
        monthly_field = gradient_monthly.isel(MONTH=i)
        repeated = monthly_field.expand_dims(TIME=tile_array[i]).copy()
        daily_list.append(repeated)
    
    daily_gradient = xr.concat(daily_list, dim="TIME")
    return daily_gradient

def daily_to_monthly_mean_leapyear(da):
    month_lengths = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if da.sizes["TIME"] != sum(month_lengths):
        raise ValueError("TIME length is not 366; check your data or month_lengths.")
    
    month_index = np.repeat(np.arange(12), month_lengths)
    MONTH = xr.DataArray(
        month_index,
        dims=["TIME"],
        coords={"TIME": da["TIME"]},
        name="MONTH",
    )
    monthly = da.groupby(MONTH).mean("TIME")
    monthly = monthly.assign_coords(MONTH=np.arange(1, 13))
    return monthly


if __name__ == "__main__":
    windstress_file_path = '/Users/julia/Desktop/SSTA/datasets/ERA5-ARGO_Mean_Turbulent_Surface_Stress_Daily_2004.nc'
    grad_lat_file_path = '/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature_Gradient_Lat.nc'
    grad_lon_file_path = '/Users/julia/Desktop/SSTA/datasets/Mixed_Layer_Temperature_Gradient_Lon.nc'

    ds_windstress = xr.open_dataset(               # (TIME: 366, LATITUDE: 145, LONGITUDE: 360)
        windstress_file_path,                      # Data variables:
        engine="netcdf4",                       # avg_iews   (TIME, LATITUDE, LONGITUDE) float32 38MB ...
        decode_times=False,                     # avg_inss   (TIME, LATITUDE, LONGITUDE) float32 38MB ...       
        mask_and_scale=True)                    # * TIME       (TIME) float32 720B 0,1,2,3,4,...365
    
    ds_grad_lat = xr.open_dataset(
        grad_lat_file_path,             # (MONTH: 12, LATITUDE: 145, LONGITUDE: 360)
        engine="netcdf4",               # * MONTH      (MONTH) int64 96B 1 2 3 4 5 6 7 8 9 10 11 12
        decode_times=False,             # Data variables: __xarray_dataarray_variable__
        mask_and_scale=True)
    
    ds_grad_lon = xr.open_dataset(
        grad_lon_file_path,             # (MONTH: 12, LATITUDE: 145, LONGITUDE: 360)
        engine="netcdf4",
        decode_times=False,
        mask_and_scale=True)



    ds_tau_x = ds_windstress['avg_iews']           # (TIME: 366, LATITUDE: 145, LONGITUDE: 360)
    ds_tau_y = ds_windstress['avg_inss']           # TIME  0,1,2,3,4,5,6,7,..,365
    dTm_dy_monthly = ds_grad_lat["__xarray_dataarray_variable__"]  # (MONTH, LAT, LON)
    dTm_dx_monthly = ds_grad_lon["__xarray_dataarray_variable__"]  # (MONTH, LAT, LON)
    #%% 
    monthly_mean_tau_x = daily_to_monthly_mean_leapyear(ds_tau_x)
    monthly_mean_tau_y = daily_to_monthly_mean_leapyear(ds_tau_y)
    
    tau_x_anom = get_anomaly(ds_tau_x, monthly_mean_tau_x)  #(TIME: 366, LATITUDE: 145, LONGITUDE: 360, MONTH)
    tau_y_anom = get_anomaly(ds_tau_y, monthly_mean_tau_y)

    #%%
    lat = ds_windstress["LATITUDE"]
    f_2d = coriolis_parameter(lat).broadcast_like(ds_tau_x)
    
    Q_ek_anom, Q_ek_anom_x, Q_ek_anom_y = ekman_current_anomaly(tau_x_anom, tau_y_anom, dTm_dx_monthly, dTm_dy_monthly, f_2d)
    
    # Q_ek_anom.to_netcdf('Ekman Current Anomaly - Daily 2024 - Test')

    print(
        # ds_windstress
        # 'ekmann current anomaly:', Q_ek_anom,
        # 'Q_ek_anom_x', Q_ek_anom_x,
        # 'Q_ek_anom_y', Q_ek_anom_y
        # ds_grad_lat,
        #  'original dataset:\n', ds,
        #'\n ds_tau_x: \n', ds_tau_x,
        #'\n ds_tau_y: \n', ds_tau_y,
        # '\n Monthly mean tau_x: \n', monthly_mean_tau_x,
        # '\n monthly mean tau_y: \n', monthly_mean_tau_y.shape,
        # '\n Full Field Monthly mean tau_x: \n', full_field_monthly_mean_tau_x,
        # '\n Full Field monthly mean tau_y: \n', full_field_monthly_mean_tau_y,
        # '\ntau x anomaly: \n', tau_x_anom['avg_iews'].values,
        # '\n tau y anomaly: \n', tau_y_anom['avg_inss'].values,
        #ds["avg_iews"]
    )


    date = 1
    month_in_year = (date % 12) + 0.5
    year = 2004 


    #---Map Plot for Ekman Current Anomaly on a date ------------------------------------------
    # Q_ek_plot = Q_ek_anom.sel(TIME=f"{date}")

    # plt.figure(figsize=(10, 5))
    # Q_ek_plot.plot(cmap="RdBu_r", vmin=-40, vmax=40, cbar_kwargs={"label": "Ekman Current Anomlay (arbitrary units)"})
    # plt.title(f"Ekman Current Anomaly on ({date})")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.tight_layout()
    # plt.show()

    #---Simulation Map for Ekman Current Anomaly -----------------------------------------------

    anim_times = Q_ek_anom.TIME.values
    lons = Q_ek_anom.LONGITUDE.values
    lats = Q_ek_anom.LATITUDE.values
    vmin, vmax = [-40,40]

    fig = plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    mesh_1 = ax1.pcolormesh(lons, lats, Q_ek_anom.isel(TIME=0), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.coastlines()
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlines = False
    gl1.ylines = False
    gl1.ylocator = LatitudeLocator()
    gl1.xformatter = LongitudeFormatter()
    gl1.yformatter = LatitudeFormatter()
    gl1.ylabel_style = {'size': 12, 'color': 'gray'}
    gl1.xlabel_style = {'size': 12, 'color': 'gray'}
    ax1.set_title('Simulation of Ekman Current Anomaly')

    def update(frame):
        # Update explicit
        Z_1 = Q_ek_anom.isel(TIME=frame).values
        mesh_1.set_array(Z_1.ravel())

        current_time = anim_times[frame]
        month_in_year = (current_time % 12) + 0.5
        year = 2004 

        ax1.set_title(f'Ekman Current Anomaly on Day {current_time} in year {year}')
        return [mesh_1]
    
    animation = FuncAnimation(fig, update, frames=len(anim_times), interval=300, blit=False)
    cbar = fig.colorbar(mesh_1, ax=ax1, orientation="vertical", pad=0.02)
    animation.save('Animation_Ekman_Current_Anomaly_Map_daily_2004.mp4', writer='ffmpeg', fps=10)   
    plt.show()

    #-----Anomaly decomposition plot-------------------------------------------------------
    anom_x = Q_ek_anom_x.sel(TIME=date)
    anom_y = Q_ek_anom_y.sel(TIME=date)
    fig = plt.figure(figsize=(24,4))
    
    ax1 = plt.subplot(1,2,1)
    im1 = anom_x.plot(
        ax=ax1,
        cmap="RdBu_r",
        vmin=-30,
        vmax=30,
        add_colorbar=False,   # we'll add colorbars explicitly to control layout
    )
    ax1.set_title(f"Ekman anomaly — taux-term — on {date}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    ax2 = plt.subplot(1,2,2)
    im2 = anom_y.plot(
        ax=ax2,
        cmap="RdBu_r",
        vmin=-30,
        vmax=30,
        add_colorbar=False,
    )
    ax2.set_title(f"Ekman anomaly — tauy -term  — on {date}")
    ax2.set_xlabel("Longitude")

    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical", pad=0.02)
    cbar.set_label("Q'_Ek (W/m^2)")
    
    plt.suptitle(f"Ekman Current Anomaly by Componenets — on Day {date} in {year}", fontsize=14)
    plt.show()
    # plt.savefig(f"Ekman_Anomaly_Componenets_subplot_{month_in_year}_{year}.png", dpi=600)

    #-----difference plot x-y -------------------------------------------------------
    # dominance = Q_ek_anom_x - Q_ek_anom_y
    # #dominance = get_monthly_mean(dominance)
    
    # # MONTH_idx = 2
    # dominance = dominance.sel(TIME = date)
    # plt.figure(figsize=(10, 5))
    # dominance.plot(
    #     cmap="RdBu_r",
    #     cbar_kwargs={"label": "Difference"},
    #     vmin = -100,
    #     vmax = 100,
    # )
    # plt.title(f"Difference of Ekman Current (x-y) on {month_in_year} ")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.tight_layout()
    # plt.show()

    # -----weighted plot --------------------------------------------------------
    Q_ek_anom_x_monthly = get_monthly_mean(Q_ek_anom_x)
    Q_ek_anom_y_monthly = get_monthly_mean(Q_ek_anom_y)
    weight_x = abs(Q_ek_anom_x_monthly)/ (abs(Q_ek_anom_y_monthly) + abs(Q_ek_anom_x_monthly))
    weight_y = abs(Q_ek_anom_y_monthly)/ (abs(Q_ek_anom_y_monthly) + abs(Q_ek_anom_x_monthly))
    weight_plot_x = weight_x.sel(MONTH = month_in_year)
    weight_plot_y = weight_y.sel(MONTH = month_in_year)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,5), sharey=True)

    # --- Plot for weight_x ---
    im1 = weight_plot_x.plot(
        ax=ax1,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        add_colorbar=False,
    )
    ax1.set_title(f"Weight of Q'_EK x-term: x/(x+y)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # --- Plot for weight_y ---
    im2 = weight_plot_y.plot(
        ax=ax2,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        add_colorbar=False,
    )
    ax2.set_title(f"Weight of Q'_EK y-term: y/(x+y)")
    ax2.set_xlabel("Longitude")

    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical")
    cbar.set_label("Weighted Ratio (0 to 1)")

    plt.suptitle(f"Relative Importance of Ekman Anomaly Terms {date})", fontsize=16)
    # plt.savefig(f"Ekman_Anom_weighted_ratio_{month_in_year}.png", dpi=600)
    plt.show()


    # ---animation----------------------------------------------------------------------------------
    # Get coordinates and the list of months to iterate over
    anim_times = weight_x.MONTH.values  # This will be [1, 2, ..., 12]
    lons = weight_x.LONGITUDE.values
    lats = weight_x.LATITUDE.values
    
    # --- BUG FIX: Define vmin/vmax for 0-1 ratio data ---
    vmin = 0
    vmax = 1
    
    # --- RECOMMENDATION: Use a sequential colormap for 0-1 data ---
    # "viridis" or "cividis" is much clearer than "RdBu_r" (diverging)
    # 
    cmap = "RdBu_r" 

    fig_2 = plt.figure(figsize=(16, 4)) # Give it a bit more width
    
    # --- Axis 1 ---
    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    # Plot initial frame (MONTH index 0, which is Month 1)
    mesh_1 = ax1.pcolormesh(lons, lats, weight_x.isel(MONTH=0), cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.coastlines()
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlines = False
    gl1.ylines = False
    gl1.ylocator = LatitudeLocator()
    gl1.xformatter = LongitudeFormatter()
    gl1.yformatter = LatitudeFormatter()
    gl1.ylabel_style = {'size': 12, 'color': 'gray'}
    gl1.xlabel_style = {'size': 12, 'color': 'gray'}
    # ax1.set_title("Weight of Q'_EK x-term: x/(x+y)") # Title will be set in update

    # --- Axis 2 ---
    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    # Plot initial frame (MONTH index 0)
    mesh_2 = ax2.pcolormesh(lons, lats, weight_y.isel(MONTH=0), cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.coastlines()
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.left_labels = False # No need for y-labels, axis is shared
    gl2.ylines = False
    gl2.xlines = False
    gl2.ylocator = LatitudeLocator()
    gl2.xformatter = LongitudeFormatter()
    gl2.yformatter = LatitudeFormatter()
    gl2.ylabel_style = {'size': 12, 'color': 'gray'}
    gl2.xlabel_style = {'size': 12, 'color': 'gray'}
    # ax2.set_title("Weight of Q'_EK y-term: y/(x+y)") # Title will be set in update
    
    # --- Colorbar and Titles ---
    cbar_2 = fig_2.colorbar(mesh_1, ax=[ax1, ax2], shrink=0.7, label='Weighted Ratio (0 to 1)')
    
    # Add a main title that will be updated
    main_title = fig_2.suptitle('Relative Importance of Ekman Anomaly Terms', fontsize=14)
    
    # --- Corrected Update Function ---
    def update(frame):
        # 'frame' will be an integer from 0 to 11
        
        # Update plot data
        Z_1 = weight_x.isel(MONTH=frame).values
        mesh_1.set_array(Z_1.ravel())

        Z_2 = weight_y.isel(MONTH=frame).values
        mesh_2.set_array(Z_2.ravel())

        # --- BUG FIX 2: Title Logic ---
        # anim_times[frame] is the month number (1, 2, ..., 12)
        current_month = anim_times[frame] 
        
        # The old logic for year/month was from the TIME coordinate,
        # but here we are just iterating over 12 months.
        
        ax1.set_title("Weight of Q'_EK x-term: |x| / (|x| + |y|)")
        ax2.set_title("Weight of Q'_EK y-term: |y| / (|x| + |y|)")
        main_title.set_text(f'Relative Importance of Ekman Anomaly Terms (Month: {current_month})')

        return [mesh_1, mesh_2]

    # --- Create, Save, and Show Animation ---
    animation = FuncAnimation(fig_2, update, frames=len(anim_times), interval=400, blit=False)
    
    # Save the animation (uncommented and with a file name)
    print("Saving animation... This may take a moment.")
    # animation.save('weighted_ratio_animation.mp4', writer='ffmpeg', fps=3) # fps=3 gives a 4-second video
    print("Animation saved as 'weighted_ratio_animation.mp4'")

    plt.show() # This will show the *live* animation *after* saving
# %%
