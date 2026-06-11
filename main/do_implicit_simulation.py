"""
Simulate Sea Surface Temperature Anomalies (SSTA) - implicit scheme.

Christopher O'Sullivan
2026-6-10
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utilities_chris import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter, get_month_from_time
from utilities_chris import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

"""Run the implicit simulation model"""

def run_model(INCLUDE_SURFACE, SPLIT_SURFACE, INCLUDE_RADIATIVE_SURFACE, INCLUDE_TURBULENT_SURFACE,
              INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION, INCLUDE_ENTRAINMENT,
              INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION,
              INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, USE_DOWNLOADED_SSH=False, USE_OTHER_MLD=False,
              USE_MAX_GRADIENT_METHOD=True, USE_LOG_FOR_ENTRAINMENT=False, gamma_0=15.0, DATA_TO_2025=True,
              adjust_mld=0.0, rho_0=1025.0, c_0=4100.0):
    """Calculate the SSTA for every month.
    Args:
        INCLUDE_SURFACE (bool): whether any (radiative or turbulent) air-sea heat flux term should be included
        SPLIT_SURFACE (bool): whether radiative and turbulent air-sea heat fluxes should be separately considered; if FALSE, then both terms are included. This condition has no effect when INCLUDE_SURFACE is FALSE.
        INCLUDE_RADIATIVE_SURFACE (bool): whether radiative air-sea heat flux should be included. This condition has no effect when INCLUDE_SURFACE is FALSE.
        INCLUDE_TURBULENT_SURFACE (bool): whether turbulent air-sea heat flux should be included. This condition has no effect when INCLUDE_SURFACE is FALSE.
        INCLUDE_EKMAN_ANOM_ADVECTION (bool): whether Ekman anomalous advection should be included
        INCLUDE_EKMAN_MEAN_ADVECTION (bool): whether Ekman mean advection should be included
        INCLUDE_ENTRAINMENT (bool): whether entrainment (considering only monthly mean entrainment velocity) should be included; this is the core entrainment term
        INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING (bool): whether entrainment (considering only monthly anomalous entrainment velocity) should be included; this term has little impact, and should generally not be included
        INCLUDE_GEOSTROPHIC_ANOM_ADVECTION (bool): whether geostorphic anomalous advection should be included
        INCLUDE_GEOSTROPHIC_MEAN_ADVECTION (bool): whether geostorphic mean advection should be included
        USE_DOWNLOADED_SSH (bool): whether SSH data (used for the geostrophic terms) should be taken from downloaded satellite altimetry; unreliable, and should generally be FALSE in order to use calculated SSH values
        USE_OTHER_MLD (bool): whether an MLD approach that takes Tsub from the Argo layer below the threshold and MLD from the Argo layer above the threshold should be used; unreliable, and should generally be FALSE.
        USE_MAX_GRADIENT_METHOD (bool): whether Tsub should be determined by a max-gradient method (searching for the maximum gradient of the temperature profile), while MLD continues to be determined by potential density threshold; TRUE by default, as it gives the most reliable results
        USE_LOG_FOR_ENTRAINMENT (bool): whether entrainment should be calculated by a log term, following Liu's paper; generally FALSE due to this approach yielding unreliable results
        gamma_0 (float): value of air-sea damping parameter; 15.0 empirically-determined to yield mean autocorrelation closest to observations
        DATA_TO_2025 (bool): whether the model should run until 2025. If FALSE, then the model stops at 2019.
        adjust_mld (float): diagnostic tool to offset the MLD by a given depth. Positive values offset the MLD to be deeper. 0.0 by default (so MLD is calculated from potential density threshold and not adjusted).

    Outputs:
        model_anomalies_ds (saved to NetCDF): xarray dataset of monthly simulated SSTA
        flux_components_ds (saved to NetCDF): xarray dataset of monthly simulated contribution to SSTA of each forcing term
    """

    # based on the given arguments, get the unique savename for this model; in general, the savenames are not meant to be human-readable, and should always be accessed
    save_name = get_save_name(INCLUDE_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_ENTRAINMENT,
                              INCLUDE_GEOSTROPHIC_ANOM_ADVECTION, USE_DOWNLOADED_SSH, gamma0=gamma_0,
                              INCLUDE_GEOSTROPHIC_DISPLACEMENT=INCLUDE_GEOSTROPHIC_MEAN_ADVECTION,
                              INCLUDE_EKMAN_MEAN_ADVECTION=INCLUDE_EKMAN_MEAN_ADVECTION, OTHER_MLD=USE_OTHER_MLD,
                              MAX_GRAD_TSUB=USE_MAX_GRADIENT_METHOD,
                              ENTRAINMENT_VEL_ANOM_FORC=INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING,
                              LOG_ENTRAINMENT_VELOCITY=USE_LOG_FOR_ENTRAINMENT, SPLIT_SURFACE=SPLIT_SURFACE,
                              INCLUDE_RADIATIVE_SURFACE=INCLUDE_RADIATIVE_SURFACE,
                              INCLUDE_TURBULENT_SURFACE=INCLUDE_TURBULENT_SURFACE, DATA_TO_2025=DATA_TO_2025,
                              adjust_mld=adjust_mld)

    """Gather filepaths of all required datasets. If data to 2025 is required, load those datasets. Otherwise, load datasets that only contain data up to 2019."""

    if DATA_TO_2025:
        HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "datasets/Surface_Heat_Flux-(2004-2025).nc"  # contains air-sea heat flux data
        EKMAN_ANOMALY_DATA_PATH = "datasets/Simulation-Ekman_Heat_Flux-(2004-2025).nc"  # contains Ekman anomalous advection data
        TEMP_DATA_PATH = "datasets/RGARGO/RG_ArgoClim_Temperature_2019.nc"  # contains Argo measured temperature data
        GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "datasets/Simulation-Geostrophic_Heat_Flux-(2004-2025).nc"  # contains geostrophic anomalous advection data
        EKMAN_MEAN_ADVECTION_DATA_PATH = "datasets/2025_ekman_mean_advection.nc"  # contains Ekman mean advection data
        ENTRAINMENT_VEL_ANOMALY_FORCING_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/entrainment_velocity_anomaly_forcing.nc"  # contains monthly anomalous entrainment velocity data
        ENTRAINMENT_VEL_DATA_PATH = "datasets/Mixed_Layer_Entrainment_Velocity-(2004-2025).nc"  # contains monthly mean entrainment velocity data

        if USE_OTHER_MLD:
            """
            CHENGYUN comment: don't turn this on, I didn't give you these files.
            """
            H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/other_h_bar.nc"  # contains MLD when determined by the `other` method (the Argo layer above potential density the threshold)
            T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/other_t_sub_anomaly.nc"  # contains anomalous Tsub when determined by the `other` method (the Argo layer below the potential density threshold)

        else:
            H_BAR_DATA_PATH = "datasets/Mixed_Layer_Depth-(2004-2025).nc"  # contains MLD at the potential density threshold
            T_SUB_DATA_PATH = "datasets/Sub_Layer_Temperature_Anomalies-(2004-2025).nc"  # contains anomalous temperature (Tsub) at the potential density threshold

        if USE_MAX_GRADIENT_METHOD:
            T_SUB_DATA_PATH = "datasets/Sub_Layer_Temperature_Max_Gradient_Method-(2004-2025).nc"  # contains Tsub at the point of maximum thermocline gradient

    else:
        """
        CHENGYUN comment: don't turn this on, I didn't give you these files.
        """
        # data paths contain data as explained above
        HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/heat_flux_interpolated_all_contributions.nc"
        EKMAN_ANOMALY_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/Ekman_Current_Anomaly.nc"
        TEMP_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
        GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/geostrophic_anomaly_downloaded.nc"
        GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/geostrophic_anomaly_calculated.nc"
        EKMAN_MEAN_ADVECTION_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/ekman_mean_advection.nc"
        ENTRAINMENT_VEL_ANOMALY_FORCING_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/entrainment_velocity_anomaly_forcing.nc"
        ENTRAINMENT_VEL_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/Entrainment_Vel_h.nc"

        if USE_OTHER_MLD:
            """
            CHENGYUN comment: don't turn this on, I didn't give you these files.
            """
            H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/other_h_bar.nc"
            T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/other_t_sub_anomaly.nc"

        else:
            """
            CHENGYUN comment: don't turn this on, I didn't give you these files.
            """
            H_BAR_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/hbar.nc"
            T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/t_sub.nc"

        if USE_MAX_GRADIENT_METHOD:
            """
            CHENGYUN comment: don't turn this on, I didn't give you these files.
            """
            T_SUB_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/mld_other_method/Tsub_Max_Gradient_Method_h.nc"

    """Load data from each dataset from each of the filepaths above, formatting where necessary."""
    temperature_ds = load_and_prepare_dataset(
        TEMP_DATA_PATH)  # temperature (and salinity) as recorded by Argo, at every pressure level

    # get heat flux and combine into total (net), radiative, and turbulent parts
    heat_flux_ds = xr.open_dataset(HEAT_FLUX_ALL_CONTRIBUTIONS_DATA_PATH,
                                   decode_times=False)  # four heat flux contributions from ERA5
    heat_flux_ds['NET_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_snlwrf'] + heat_flux_ds[
        'avg_snswrf'] + \
                                    heat_flux_ds['avg_ishf']
    heat_flux_ds['RADIATIVE_HEAT_FLUX'] = heat_flux_ds['avg_snlwrf'] + heat_flux_ds['avg_snswrf']
    heat_flux_ds['TURBULENT_HEAT_FLUX'] = heat_flux_ds['avg_slhtf'] + heat_flux_ds['avg_ishf']

    # get heat flux forcing anomaly in usual way: first get monthly mean, and subtract that from the dataset
    heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['NET_HEAT_FLUX'])
    heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'NET_HEAT_FLUX', heat_flux_monthly_mean)
    surface_flux_da = heat_flux_anomaly_ds['NET_HEAT_FLUX_ANOMALY']

    # radiative heat flux forcing anomaly
    rad_heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['RADIATIVE_HEAT_FLUX'])
    rad_heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'RADIATIVE_HEAT_FLUX', rad_heat_flux_monthly_mean)
    rad_surface_flux_da = rad_heat_flux_anomaly_ds['RADIATIVE_HEAT_FLUX_ANOMALY']

    # turbulent heat flux forcing anomaly
    turb_heat_flux_monthly_mean = get_monthly_mean(heat_flux_ds['TURBULENT_HEAT_FLUX'])
    turb_heat_flux_anomaly_ds = get_anomaly(heat_flux_ds, 'TURBULENT_HEAT_FLUX', turb_heat_flux_monthly_mean)
    turb_surface_flux_da = turb_heat_flux_anomaly_ds['TURBULENT_HEAT_FLUX_ANOMALY']

    # Ekman heat flux is already an anomaly; due to differences in files, the variable name is different for the file containing data up to 2025, necessitating the conditional
    ekman_anomaly_ds = xr.open_dataset(EKMAN_ANOMALY_DATA_PATH, decode_times=False)
    if DATA_TO_2025:
        ekman_anomaly_da = ekman_anomaly_ds['ANOMALY_EKMAN_HEAT_FLUX']
    else:
        ekman_anomaly_da = ekman_anomaly_ds['Q_Ek_anom']
    ekman_anomaly_da = ekman_anomaly_da.where(~np.isnan(ekman_anomaly_da),
                                              0)  # replace NaN with 0 to avoid propagating NaN throughout the model (NaN is forced near the equator due to 1/f factor becoming large)

    # monthly mean MLD (equivalently called hbar), again with differences in whether the data is taken to 2025
    hbar_ds = xr.open_dataset(H_BAR_DATA_PATH, decode_times=False)
    if DATA_TO_2025:
        h_da = hbar_ds["MLD"]
        hbar_da = get_monthly_mean(
            h_da)  # hbar_ds, for the datafile up to 2025, contains the MLD for every month, so to take the monthly mean to get hbar
    else:
        hbar_da = hbar_ds["MONTHLY_MEAN_MLD"]

    hbar_da = (hbar_da + adjust_mld).clip(
        min=2.5)  # require the minimum MLD to be 2.5 m (by default it will be, but if an adjustment to MLD is made, we need to prevent MLD<0)

    # entrainment velocity anomaly

    """
    CHENGYUN comment: I commented the following two lines and assigned a None,
    because they only applies to -2019 simulation.
    """
    # entrainment_vel_anomaly_forcing_ds = xr.open_dataset(ENTRAINMENT_VEL_ANOMALY_FORCING_DATA_PATH, decode_times=False)
    # entrainment_vel_anomaly_forcing = entrainment_vel_anomaly_forcing_ds["ENTRAINMENT_VEL_ANOMALY_FORCING"]
    entrainment_vel_anomaly_forcing = None

    # tsub, containing the sub-layer temperature for every month
    t_sub_ds = xr.open_dataset(T_SUB_DATA_PATH, decode_times=False)
    if USE_OTHER_MLD:
        t_sub_da = t_sub_ds["ANOMALY_SUB_TEMPERATURE"]
    # elif DATA_TO_2025:        # IGNORE; this use case doesn't exist since we consider "OTHER_MLD" to be unhelpful for any practical purpose
    #     t_sub_da = get_anomaly(t_sub_ds, "SUB_TEMPERATURE", get_monthly_mean(t_sub_ds["SUB_TEMPERATURE"]))["SUB_TEMPERATURE_ANOMALY"]
    else:  # get anomaly of the dataset
        t_sub_da = get_anomaly(t_sub_ds, "SUB_TEMPERATURE", get_monthly_mean(t_sub_ds["SUB_TEMPERATURE"]))[
            "SUB_TEMPERATURE_ANOMALY"]

    # entrainment velocity, again with different naming convention for datafile up to 2025. Save only the monthly mean (since the anomalous component is considered within entrainment_vel_anomaly_forcing_ds)
    entrainment_vel_ds = xr.open_dataset(ENTRAINMENT_VEL_DATA_PATH, decode_times=False)
    if DATA_TO_2025:
        entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(entrainment_vel_ds['w_e'])
    else:
        entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN'] = get_monthly_mean(
            entrainment_vel_ds['ENTRAINMENT_VELOCITY'])
    entrainment_vel_da = entrainment_vel_ds['ENTRAINMENT_VELOCITY_MONTHLY_MEAN']

    # filepaths of sea surface height depending on approach (downloaded or simulated SSH data)
    if USE_DOWNLOADED_SSH:  # kept for completion, but there are limited (no?) cases when the downloaded SSH data should be used. It is always better to calculate SSH ourselves.
        """
        CHENGYUN comment: don't turn this on, I didn't give you these files.
        """
        geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_DOWNLOADED_DATA_PATH, decode_times=False)
        SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_interpolated_grad.nc"
        SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_monthly_mean_interpolated_grad.nc"
    else:  # get SSH gradients (in the horizontal and vertical direction), and monthly mean
        geostrophic_anomaly_ds = xr.open_dataset(GEOSTROPHIC_ANOMALY_CALCULATED_DATA_PATH, decode_times=False)
        if DATA_TO_2025:
            SEA_SURFACE_GRAD_DATA_PATH = "datasets/2025_sea_surface_calculated_grad.nc"
            SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "datasets/2025_sea_surface_monthly_mean_calculated_grad.nc"
        else:
            """
            CHENGYUN comment: don't turn this on, I didn't give you these files.
            """
            SEA_SURFACE_GRAD_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_calculated_grad.nc"
            SEA_SURFACE_MONTHLY_MEAN_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/sea_surface_monthly_mean_calculated_grad.nc"

    # geostrophic anomaly, again different naming convention for 2025 data
    if DATA_TO_2025:
        geostrophic_anomaly_da = geostrophic_anomaly_ds["ANOMALY_GEOSTROPHIC_HEAT_FLUX"]
    else:
        geostrophic_anomaly_da = geostrophic_anomaly_ds["GEOSTROPHIC_ANOMALY"]

    # open SSH gradients and monthly mean
    sea_surface_grad_ds = xr.open_dataset(SEA_SURFACE_GRAD_DATA_PATH, decode_times=False)
    sea_surface_monthlymean_ds = xr.open_dataset(SEA_SURFACE_MONTHLY_MEAN_DATA_PATH, decode_times=False)

    # datafile for Ekman mean advection
    ekman_mean_advection = xr.open_dataset(EKMAN_MEAN_ADVECTION_DATA_PATH, decode_times=False)

    def month_to_second(month):
        return month * 30.4375 * 24 * 60 * 60

    delta_t = month_to_second(1)  # number of seconds in a month (our timestep)

    # initialise lists for each month's model anomaly and flux due to entrainment (the latter is directly calculated for each measured anomalous temperature from the formula)
    model_anomalies = []
    entrainment_fluxes = []
    added_baseline = False  # baseline being the initial condition for SST
    for month in heat_flux_anomaly_ds.TIME.values:
        # print the month every 10 months, as a form of status bar (with mean advection, the model runs slow! This can help know if there is time for a coffee before the model completes. But probably a better way exists.)
        if int(month) % 10 == 0:
            if DATA_TO_2025:
                print("Month " + str(int(month)) + " of 264")
            else:
                print("Month " + str(int(month)) + " of 180")

        # find the previous and current month from 1 to 12 to access the monthly-averaged data (hbar, entrainment vel.)
        prev_month = month - 1
        month_in_year = get_month_from_time(
            month)  # takes the "TIME" coordinate (0.5, 1.5, 2.5, ..., 180.5, ...) and determines what month of the year that corresponds to
        prev_month_in_year = get_month_from_time(month - 1)
        if not added_baseline:  # adds the baseline of a whole bunch of zero
            base = temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY'] - \
                   temperature_ds.sel(PRESSURE=2.5, TIME=month)['ARGO_TEMPERATURE_ANOMALY']
            base = base.expand_dims(TIME=[month])
            model_anomalies.append(base)
            added_baseline = True
        else:  # given that a baseline already exists, determine the next month's temperature anomaly.
            prev_anomaly = model_anomalies[-1].isel(
                TIME=-1)  # previous simulated SST, which is to be updated for the current month

            """If desired, calculate mean advection contribution"""
            if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
                # initialise alpha, beta (which represent the x, y current velocities) as 0
                alpha = xr.zeros_like(sea_surface_monthlymean_ds['alpha'].sel(MONTH=prev_month_in_year))
                beta = xr.zeros_like(sea_surface_monthlymean_ds['beta'].sel(MONTH=prev_month_in_year))

                # depending on whether geostrophic and/or Ekman mean advection is required, determine the required x, y current velocity and add those to alpha, beta respectively
                if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
                    alpha += sea_surface_monthlymean_ds['alpha'].sel(MONTH=prev_month_in_year)
                    beta += sea_surface_monthlymean_ds['beta'].sel(MONTH=prev_month_in_year)

                if INCLUDE_EKMAN_MEAN_ADVECTION:
                    alpha = alpha + ekman_mean_advection["ekman_alpha"].sel(MONTH=prev_month_in_year)
                    beta = beta + ekman_mean_advection["ekman_beta"].sel(MONTH=prev_month_in_year)

                # calculate mean advection contributions
                earth_radius = 6371000
                latitudes = np.deg2rad(sea_surface_monthlymean_ds['LATITUDE'])  # any ds to get latitude
                dx = (2 * np.pi * earth_radius / 360) * np.cos(latitudes)  # horizontal gridbox size
                dy = (2 * np.pi * earth_radius / 360) * np.ones_like(latitudes)  # vertical gridbox size (constant)
                dx = xr.DataArray(dx, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values},
                                  dims=['LATITUDE'])  # convert dx, dy to xarray for use below
                dy = xr.DataArray(dy, coords={'LATITUDE': sea_surface_monthlymean_ds['LATITUDE'].values},
                                  dims=['LATITUDE'])

                CFL_x = (abs(alpha) * delta_t / dx).max()  # CFL number
                CFL_y = (abs(beta) * delta_t / dy).max()
                CFL_max = max(float(CFL_x), float(CFL_y))
                substeps = int(np.ceil(
                    CFL_max)) + 1  # require CFL<1 for numerical stability; split the simulation into a number of substeps that ensures this (in effect, reducing delta_t)
                sub_dt = delta_t / substeps  # new timestep that ensures CFL<1

                tm_div_total = xr.zeros_like(prev_anomaly)  # divergence into each cell
                for step in range(substeps):
                    # get upwind flux (temperature at surrounding gridboxes)
                    prev_anom_east = prev_anomaly.roll(LONGITUDE=-1)
                    prev_anom_west = prev_anomaly.roll(LONGITUDE=1)
                    prev_anom_north = prev_anomaly.roll(LATITUDE=-1)
                    prev_anom_south = prev_anomaly.roll(LATITUDE=1)

                    # get alpha/beta at the edges of gridboxes (linearly-interpolate between alpha/beta values of the current cell and the neighbouring cells)
                    alpha_east = (alpha + alpha.roll(LONGITUDE=-1)) / 2
                    alpha_west = (alpha + alpha.roll(LONGITUDE=1)) / 2
                    beta_north = (beta + beta.roll(LATITUDE=-1)) / 2
                    beta_south = (beta + beta.roll(LATITUDE=1)) / 2

                    # check for nans in neighbouring cells and remove as required
                    ocean_mask = ~prev_anomaly.isnull()
                    has_east_ocean = ~prev_anom_east.isnull()
                    has_west_ocean = ~prev_anom_west.isnull()
                    has_north_ocean = ~prev_anom_north.isnull()
                    has_south_ocean = ~prev_anom_south.isnull()

                    # get upwind contributions to divergence (called F for x direction, and G for y direction); the downwind cell cannot contribute to the current cell's temperature, so we discard it
                    F_east = xr.where(alpha_east < 0, -alpha_east * prev_anomaly, -alpha_east * prev_anom_east)
                    F_west = xr.where(alpha_west < 0, -alpha_west * prev_anom_west, -alpha_west * prev_anomaly)
                    G_north = xr.where(beta_north > 0, beta_north * prev_anomaly, beta_north * prev_anom_north)
                    G_south = xr.where(beta_south > 0, beta_south * prev_anom_south, beta_south * prev_anomaly)

                    # ignore flux if advecting from land
                    F_east = xr.where(has_east_ocean, F_east, 0)
                    F_west = xr.where(has_west_ocean, F_west, 0)
                    G_north = xr.where(has_north_ocean, G_north, 0)
                    G_south = xr.where(has_south_ocean, G_south, 0)

                    # ignore flux if current cell is land
                    F_east = xr.where(ocean_mask, F_east, 0)
                    F_west = xr.where(ocean_mask, F_west, 0)
                    G_north = xr.where(ocean_mask, G_north, 0)
                    G_south = xr.where(ocean_mask, G_south, 0)

                    # get flux divergence
                    tm_div = (F_east - F_west) / dx + (G_north - G_south) / dy
                    tm_div_total += tm_div

                    # update working temperature for the substep and apply ocean mask again
                    prev_anomaly = prev_anomaly - sub_dt * tm_div
                    prev_anomaly = prev_anomaly.where(ocean_mask)
                tm_div = tm_div_total / substeps

            # reset prev_anomaly in case it was adjusted by the mean advection
            prev_anomaly = model_anomalies[-1].isel(TIME=-1)

            # get previous data (only hbar is necessary)
            prev_hbar = hbar_da.sel(MONTH=prev_month_in_year)

            # get current data
            cur_tsub_anom = t_sub_da.sel(TIME=month)
            cur_heat_flux_anom = surface_flux_da.sel(TIME=month)
            cur_rad_heat_flux_anom = rad_surface_flux_da.sel(TIME=month)
            cur_turb_heat_flux_anom = turb_surface_flux_da.sel(TIME=month)
            cur_ekman_anom = ekman_anomaly_da.sel(TIME=month)
            cur_entrainment_vel = entrainment_vel_da.sel(MONTH=month_in_year)
            cur_geo_anom = geostrophic_anomaly_da.sel(TIME=month)
            cur_hbar = hbar_da.sel(MONTH=month_in_year)
            if not DATA_TO_2025:  # entrainment velocity anomaly is only implemented for data up to 2019; as far as we know, its contribution is negligible anyway
                cur_entrainment_vel_anomaly_forcing = entrainment_vel_anomaly_forcing.sel(TIME=month)

            # static forcings are the surface flux + Ekman anomolous advection + anomalous entrainment; basically, not the mean advection
            # initialise static forcing array as 0, then we will add to it depending on the forcings selected
            cur_static_forcings = xr.zeros_like(cur_ekman_anom)

            # add air-sea heat flux to static forcings if desired
            if INCLUDE_SURFACE:
                if SPLIT_SURFACE:
                    if INCLUDE_RADIATIVE_SURFACE:
                        cur_static_forcings += cur_rad_heat_flux_anom
                    if INCLUDE_TURBULENT_SURFACE:
                        cur_static_forcings += cur_turb_heat_flux_anom
                else:
                    cur_static_forcings += cur_heat_flux_anom

            # add Ekman anomalous advection to static forcings if desired
            if INCLUDE_EKMAN_ANOM_ADVECTION:
                cur_static_forcings += cur_ekman_anom

            # add entrainment to static forcings if desired, with log formula if desired (but that should not be the default)
            if INCLUDE_ENTRAINMENT:
                if USE_LOG_FOR_ENTRAINMENT:
                    cur_static_forcings += cur_hbar / delta_t * np.log(
                        cur_hbar / prev_hbar) * cur_tsub_anom * rho_0 * c_0
                else:
                    cur_static_forcings += cur_entrainment_vel * cur_tsub_anom * rho_0 * c_0

            # add anomalous entrianment velocity forcing to static forcings if desired
            if INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING:
                cur_static_forcings += cur_entrainment_vel_anomaly_forcing

            # add geostrophic anomalous advection to static forcings if desired
            if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
                cur_static_forcings += cur_geo_anom
                # prev_static_forcings += prev_geo_anom

            # build model components; b and a refer to the coefficients of the SST^0 and SST^1 terms respectively
            cur_b = cur_static_forcings / (rho_0 * c_0 * cur_hbar)

            cur_a = gamma_0 / (rho_0 * c_0 * cur_hbar)
            if INCLUDE_ENTRAINMENT:
                cur_a += cur_entrainment_vel / cur_hbar
            if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION:
                cur_a += (- sea_surface_monthlymean_ds["alpha_grad_long"].sel(MONTH=month_in_year) +
                          sea_surface_monthlymean_ds["beta_grad_lat"].sel(MONTH=month_in_year)).clip(-1e-7, 1e-7)
            if INCLUDE_EKMAN_MEAN_ADVECTION:
                cur_a += (- ekman_mean_advection["ekman_alpha_grad_long"].sel(MONTH=month_in_year) +
                          ekman_mean_advection["ekman_beta_grad_lat"].sel(MONTH=month_in_year)).clip(-1e-7, 1e-7)

            # update anomaly using previous anomaly, timestep (delta_t), a, and b values
            cur_anomaly = (prev_anomaly + delta_t * cur_b) / (1 + delta_t * cur_a)

            # if the mean advection terms were included, we must add a correction term (this originates from the use of flux-form for the mean advection model)
            if INCLUDE_GEOSTROPHIC_MEAN_ADVECTION or INCLUDE_EKMAN_MEAN_ADVECTION:
                cur_anomaly = cur_anomaly - tm_div * delta_t
                cur_anomaly = cur_anomaly.where(ocean_mask, cur_anomaly)

            # reformat a bit
            cur_anomaly = cur_anomaly.drop_vars('MONTH', errors='ignore')
            cur_anomaly = cur_anomaly.expand_dims(TIME=[month])
            model_anomalies.append(cur_anomaly)

            # calculate flux due to entrainment, which is done using the relevant equation
            if INCLUDE_ENTRAINMENT:
                entrainment_flux = rho_0 * c_0 * cur_entrainment_vel * (cur_tsub_anom - cur_anomaly)
                entrainment_fluxes.append(entrainment_flux)

    # concatenate SST from a list into a dataset
    model_anomalies_ds = xr.concat(model_anomalies, 'TIME')
    model_anomalies_ds = model_anomalies_ds.rename("IMPLICIT")
    model_anomalies_ds = model_anomalies_ds.to_dataset(name="IMPLICIT")

    # remove whatever seasonal cycle remains by getting monthly mean and removing it (there was no general requirement that the monthly mean will have remained 0 throughout the modelling process)
    # ignore the first year of data, in order to negate any effect of the initial condition (the autocorrelation after 12 months, ignoring re-entrainment effects, is negligible)
    if DATA_TO_2025:
        monthly_mean = get_monthly_mean(model_anomalies_ds["IMPLICIT"].sel(TIME=slice(12.5, 264.5)))
    else:
        monthly_mean = get_monthly_mean(model_anomalies_ds["IMPLICIT"].sel(TIME=slice(12.5, 180.5)))
    model_anomalies_ds["IMPLICIT"] = get_anomaly(model_anomalies_ds, "IMPLICIT", monthly_mean)["IMPLICIT_ANOMALY"]
    model_anomalies_ds = model_anomalies_ds.drop_vars("IMPLICIT_ANOMALY")  # just a reformatting

    # save the SST dataset into a NETCDF file
    model_anomalies_ds = remove_empty_attributes(
        model_anomalies_ds)  # when doing the seasonality removal, some units are None
    model_anomalies_ds.to_netcdf("datasets/implicit_model/" + save_name + ".nc")

    # save contributions from each forcing
    if INCLUDE_ENTRAINMENT:
        entrainment_fluxes = xr.concat(entrainment_fluxes, 'TIME')
        entrainment_fluxes = entrainment_fluxes.drop_vars(["MONTH", "PRESSURE"])
        entrainment_fluxes = entrainment_fluxes.transpose("TIME", "LATITUDE", "LONGITUDE")
        entrainment_fluxes = entrainment_fluxes.rename("ENTRAINMENT_ANOMALY")

    # merge the relevant fluxes into a single dataset, for each forcing that was included
    # the process for each forcing is to ensure we are dealing with an anomaly, give a human-readable name to the forcing, and then add the dataset into the flux_components_to_merge list. Later, we will concatenate this list into a dataset.
    flux_components_to_merge = []
    variable_names = []
    if INCLUDE_SURFACE:
        surface_flux_da = surface_flux_da.rename("SURFACE_FLUX_ANOMALY")
        flux_components_to_merge.append(surface_flux_da)
        variable_names.append("SURFACE_FLUX_ANOMALY")

        slhrf_heat_flux_anomaly = get_anomaly(heat_flux_ds, 'avg_slhtf', get_monthly_mean(heat_flux_ds['avg_slhtf']))[
            'avg_slhtf_ANOMALY']
        snlwrf_heat_flux_anomaly = \
        get_anomaly(heat_flux_ds, 'avg_snlwrf', get_monthly_mean(heat_flux_ds['avg_snlwrf']))['avg_snlwrf_ANOMALY']
        snswrf_heat_flux_anomaly = \
        get_anomaly(heat_flux_ds, 'avg_snswrf', get_monthly_mean(heat_flux_ds['avg_snswrf']))['avg_snswrf_ANOMALY']
        ishf_heat_flux_anomaly = get_anomaly(heat_flux_ds, 'avg_ishf', get_monthly_mean(heat_flux_ds['avg_ishf']))[
            'avg_ishf_ANOMALY']

        slhrf_heat_flux_anomaly = slhrf_heat_flux_anomaly.rename("SURFACE_LATENT_HF_ANOMALY")
        snlwrf_heat_flux_anomaly = snlwrf_heat_flux_anomaly.rename("SURFACE_LW_RADIATION_FLUX_ANOMALY")
        snswrf_heat_flux_anomaly = snswrf_heat_flux_anomaly.rename("SURFACE_SW_RADIATION_FLUX_ANOMALY")
        ishf_heat_flux_anomaly = ishf_heat_flux_anomaly.rename("SURFACE_SENSIBLE_HF_ANOMALY")

        flux_components_to_merge.append(slhrf_heat_flux_anomaly)
        flux_components_to_merge.append(snlwrf_heat_flux_anomaly)
        flux_components_to_merge.append(snswrf_heat_flux_anomaly)
        flux_components_to_merge.append(ishf_heat_flux_anomaly)

        variable_names.append("SURFACE_LATENT_HF_ANOMALY")
        variable_names.append("SURFACE_LW_RADIATION_FLUX_ANOMALY")
        variable_names.append("SURFACE_SW_RADIATION_FLUX_ANOMALY")
        variable_names.append("SURFACE_SENSIBLE_HF_ANOMALY")

    if INCLUDE_EKMAN_ANOM_ADVECTION:
        ekman_anomaly_da = ekman_anomaly_da.rename("EKMAN_ANOM_ADVECTION_ANOMALY")
        flux_components_to_merge.append(ekman_anomaly_da)
        variable_names.append("EKMAN_ANOM_ADVECTION_ANOMALY")

    if INCLUDE_GEOSTROPHIC_ANOM_ADVECTION:
        geostrophic_anomaly_da = geostrophic_anomaly_da.rename("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY")
        flux_components_to_merge.append(geostrophic_anomaly_da)
        variable_names.append("GEOSTROPHIC_ANOM_ADVECTION_ANOMALY")

    if INCLUDE_ENTRAINMENT:
        flux_components_to_merge.append(entrainment_fluxes)
        variable_names.append("ENTRAINMENT_ANOMALY")

    flux_components_ds = xr.merge(flux_components_to_merge)

    # remove whatever seasonal cycle may remain from the components
    for variable_name in variable_names:
        monthly_mean = get_monthly_mean(flux_components_ds[variable_name])
        flux_components_ds[variable_name] = get_anomaly(flux_components_ds, variable_name, monthly_mean)[
            variable_name + "_ANOMALY"]
        flux_components_ds = flux_components_ds.drop_vars(variable_name + "_ANOMALY")

    # save flux components to NETCDF4
    flux_components_ds = remove_empty_attributes(flux_components_ds)
    print(flux_components_ds)
    flux_components_ds.to_netcdf(
        "datasets/implicit_model/" + save_name + "_flux_components.nc")
