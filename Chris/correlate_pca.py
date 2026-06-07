import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from eofs.tools.standard import correlation_map
from scipy import stats
import cartopy.crs as ccrs
import ssl

from scipy.signal import welch

ssl._create_default_https_context = ssl._create_unverified_context

from Chris.correlation_significance import get_significance
from Chris.utils import make_movie, get_eofs, get_eof, get_eof_with_nan_consideration, get_eof_from_ppca_py, \
    get_save_name, \
    get_month_from_time, format_cartopy, mask_dataset
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from correlation_significance import mutual_information

MASK_TROPICS = True
MASK_TROPICS_LATITUDE = 15

MASK_SOME_REGIONS = False
MASK_REGIONS = [(slice(15, 25), slice(-75, -65)), (slice(-50, -30), slice(125, 150))]

NORTH_ATLANTIC = True
NA_LAT_BOUNDS = slice(0, 80)
NA_LONG_BOUNDS = slice(-80, 10)

ARGO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/RG_ArgoClim_Temperature_2019.nc"
OBSERVATIONS_2025_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/datasets2025/reynolds_sst_Anomalies-(2004-2025)_no_2004.nc"
obs_ds = xr.open_dataset(OBSERVATIONS_2025_DATA_PATH, decode_times=False)
observed_anomaly = obs_ds['ANOMALY_SST']
if MASK_TROPICS:
    observed_anomaly = observed_anomaly.where((observed_anomaly.LATITUDE > MASK_TROPICS_LATITUDE) | (observed_anomaly.LATITUDE < -1 * MASK_TROPICS_LATITUDE), np.nan)
if NORTH_ATLANTIC:
    observed_anomaly = observed_anomaly.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)

save_name_full = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_rad = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=False, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_turb = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=False, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_ekanom = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=False, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_ekmean = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_entrainment = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=False, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_geoanom = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=False, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_geomean = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=True, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=False, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_turb_or_ekman = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=False, INCLUDE_EKMAN=False, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=True, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_meanadv = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=True, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=False, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)
save_name_no_entrainment_or_meanadv = get_save_name(INCLUDE_SURFACE=True, SPLIT_SURFACE=True, INCLUDE_RADIATIVE_SURFACE=True, INCLUDE_TURBULENT_SURFACE=True, INCLUDE_EKMAN=True, INCLUDE_EKMAN_MEAN_ADVECTION=False, INCLUDE_ENTRAINMENT=False, INCLUDE_GEOSTROPHIC=True, INCLUDE_GEOSTROPHIC_DISPLACEMENT=False, USE_DOWNLOADED_SSH=False, gamma0=15.0, OTHER_MLD=False, MAX_GRAD_TSUB=True, ENTRAINMENT_VEL_ANOM_FORC=False, LOG_ENTRAINMENT_VELOCITY=False, DATA_TO_2025=True)

save_names = [save_name_full, save_name_no_rad, save_name_no_turb, save_name_no_ekanom, save_name_no_ekmean, save_name_no_entrainment, save_name_no_geoanom, save_name_no_geomean, save_name_no_turb_or_ekman, save_name_no_meanadv, save_name_no_entrainment_or_meanadv]
readable_labels = ["Full model", "No radiative air-sea flux", "No turbulent air-sea flux", "No Ekman anom. adv.", "No Ekman mean adv.", "No entrainment", "No geostrophic anom. adv.", "No geostrophic mean adv.", "No turbulent air-sea flux or Ekman adv.", "No mean adv.", "No entrainment or mean adv."]

argo_observations_ds = load_and_prepare_dataset(ARGO_DATA_PATH)
map_mask = argo_observations_ds['BATHYMETRY_MASK'].sel(PRESSURE=2.5).drop_vars("PRESSURE")
# map_mask = map_mask.sel(LATITUDE=NEP_LAT_BOUNDS).sel(LONGITUDE=NEP_LONG_BOUNDS)

def get_model(save_name, readable_label, map_mask):
    IMPLICIT_SCHEME_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/implicit_model/" + save_name + ".nc"
    model = xr.open_dataset(IMPLICIT_SCHEME_DATA_PATH, decode_times=False)["IMPLICIT"]
    if MASK_TROPICS:
        model = model.where((model.LATITUDE > MASK_TROPICS_LATITUDE) | (model.LATITUDE < -1 * MASK_TROPICS_LATITUDE),
                            np.nan)
        map_mask = map_mask.where(
            (model.LATITUDE > MASK_TROPICS_LATITUDE) | (map_mask.LATITUDE < -1 * MASK_TROPICS_LATITUDE), np.nan)
    if NORTH_ATLANTIC:
        model = model.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)
        map_mask = map_mask.sel(LATITUDE=NA_LAT_BOUNDS).sel(LONGITUDE=NA_LONG_BOUNDS)
    if readable_label == "No entrainment":
        model = mask_dataset(model, MASK_REGIONS)
        map_mask = mask_dataset(map_mask, MASK_REGIONS)
    return model, map_mask


def plot_pcs(save_names, readable_labels, map_mask, obs=False, plot_eof=False):
    plt.grid()
    for i, save_name in enumerate(save_names):
        model, map_mask = get_model(save_name, readable_labels[i], map_mask)
        EOFPCs = get_eofs(model, 0, 3, map_mask=map_mask, standardise=True)
        pcs = EOFPCs[1][:, 1] * -1
        print(pcs)
        plt.plot(model.TIME.values / 12 + 2004, pcs, label=readable_labels[i])
        if plot_eof:
            eof = EOFPCs[0]
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            eof.plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=-3, vmax=3)
            plt.xlabel("Longitude (º)")
            plt.ylabel("Latitude (º)")
            cbar = plt.gcf().axes[-1]
            cbar.set_ylabel('EOF (standardised)', rotation=90)
            ax = format_cartopy(ax)
            plt.show()

    if obs:
        EOFPCs_obs = get_eofs(observed_anomaly, 0, 3, map_mask=map_mask, invert=False, standardise=True)
        pcs_obs = EOFPCs_obs[1][:, 0]
        plt.plot(observed_anomaly.TIME.values / 12 + 2004, pcs_obs, label="Observations")
        if plot_eof:
            eof = EOFPCs_obs[0]
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            eof.plot(x='LONGITUDE', y='LATITUDE', cmap='RdBu_r', vmin=-3, vmax=3)
            plt.xlabel("Longitude (º)")
            plt.ylabel("Latitude (º)")
            cbar = plt.gcf().axes[-1]
            cbar.set_ylabel('EOF (standardised)', rotation=90)
            ax = format_cartopy(ax)
            plt.show()
    correlation = np.corrcoef(pcs, pcs_obs)[0, 1]
    print(correlation)
    plt.xlabel("Year")
    plt.ylabel("PC (standardised)")
    plt.legend()
    plt.show()


def obs_pc_projected(model_save_names, model_readable_labels, obs, map_mask, n_modes=3):
    # largely from Jason's file
    for i, save_name in enumerate(model_save_names):
        print(model_readable_labels[i])
        model, map_mask = get_model(save_name, model_readable_labels[i], map_mask)
        EOFPCs_obs = get_eofs(obs, 0, n_modes, map_mask, invert=False, standardise=False)
        EOF_obs = EOFPCs_obs[0]
        PC_obs = EOFPCs_obs[1]
        fig, axes = plt.subplots(n_modes, 1, figsize=(12, 4 * n_modes), sharex=True)
        if n_modes == 1:
            axes = [axes]

        for m in range(n_modes):
            ax = axes[m]
            model_pseudo_pc = xr.dot(model.fillna(0), EOF_obs.sel(MODE=m).fillna(0), dims=['LATITUDE', 'LONGITUDE'])    # project obs spatial pattern onto model
            obs_pc_raw = PC_obs[:, m]

            # power spectra
            freqs, psd_obs = welch(obs_pc_raw, fs=1, nperseg=120)
            freqs, psd_model = welch(model_pseudo_pc, fs=1, nperseg=120)
            periods = 1 / freqs[1:]

            # Normalise (Z-score)
            # model_pc_norm = (model_pseudo_pc - model_pseudo_pc.mean()) / model_pseudo_pc.std()
            # obs_pc_norm = (obs_pc_raw - obs_pc_raw.mean()) / obs_pc_raw.std()

            ax.plot(model.TIME.values / 12 + 2004, obs_pc_raw, label="Observations", color='black', linewidth=1.5)
            ax.plot(model.TIME.values / 12 + 2004, model_pseudo_pc, label="Model", linewidth=1.5)

            # Correlation Statistics
            correlation = np.corrcoef(obs_pc_raw, model_pseudo_pc)[0, 1]  # Use pseudo_pc directly for corr
            ax.text(0.02, 0.9, f"Mode {m+1}; Correlation: {correlation:.2f}", transform=ax.transAxes, fontweight='bold')

            # # RMSE Statistics
            # raw_rmse = np.sqrt(np.mean((obs_pc_raw - model_pseudo_pc) ** 2))
            # obs_rms = np.sqrt(np.mean(obs_pc_raw ** 2))
            # nrmse_pc = raw_rmse / obs_rms
            # ax.text(0.02, 0.8, f"PC-RMSE: {nrmse_pc:.3f}", transform=ax.transAxes)
            ax.set_ylabel("Principal Component Amplitude")  # need to change
            # ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
            ax.legend(loc='upper right', fontsize='small')
            # # ax.set_title(f"{scheme_name} | Principal Component Comparison: Mode {m}")

        axes[-1].set_xlabel("Year")
        plt.tight_layout()
        plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/final_results/pc_projected.png", dpi=400,transparent=True)
        plt.show()


def pcs_power_spectrum(model_save_names, model_readable_labels, obs, map_mask, n_modes=3):
    for i, save_name in enumerate(model_save_names):
        print(model_readable_labels[i])
        model, map_mask = get_model(save_name, model_readable_labels[i], map_mask)
        EOFPCs_obs = get_eofs(obs, 0, n_modes, map_mask, invert=False, standardise=False)
        EOF_obs = EOFPCs_obs[0]
        PC_obs = EOFPCs_obs[1]
        fig, axes = plt.subplots(n_modes, 1, figsize=(12, 4 * n_modes), sharex=True)
        if n_modes == 1:
            axes = [axes]

        for m in range(n_modes):
            ax = axes[m]
            model_pseudo_pc = xr.dot(model.fillna(0), EOF_obs.sel(MODE=m).fillna(0), dims=['LATITUDE', 'LONGITUDE'])    # project obs spatial pattern onto model
            obs_pc_raw = PC_obs[:, m]

            # power spectra
            freqs, psd_obs = welch(obs_pc_raw, fs=1, nperseg=48)
            freqs, psd_model = welch(model_pseudo_pc.values, fs=1, nperseg=48)
            periods = 1 / freqs[1:]

            # Normalise (Z-score)
            # model_pc_norm = (model_pseudo_pc - model_pseudo_pc.mean()) / model_pseudo_pc.std()
            # obs_pc_norm = (obs_pc_raw - obs_pc_raw.mean()) / obs_pc_raw.std()

            ax.plot(periods, psd_obs[1:], label="Observed (Reynolds)", color='black', alpha=0.6, linewidth=1.5)
            ax.plot(periods, psd_model[1:], label=f"Model", color='crimson', linestyle='--', linewidth=1.5)

            # # Correlation Statistics
            # correlation = np.corrcoef(psd_obs, psd_model)[0, 1]
            # ax.text(0.02, 0.9, f"Mode {m} | Correlation: {correlation:.2f}", transform=ax.transAxes, fontweight='bold')
            #
            # # RMSE Statistics
            # raw_rmse = np.sqrt(np.mean((psd_obs - psd_model) ** 2))
            # obs_rms = np.sqrt(np.mean(psd_obs ** 2))
            # nrmse_pc = raw_rmse / obs_rms
            # ax.text(0.02, 0.8, f"PC-RMSE: {nrmse_pc:.3f}", transform=ax.transAxes)
            ax.set_yscale('log')
            ax.set_ylabel("PSD (variance / cycle per month)")  # need to change
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
            ax.legend(loc='upper right', fontsize='small')
            # ax.set_title(f"{scheme_name} | Principal Component Comparison: Mode {m}")

        axes[-1].set_xlabel("Periods (months)")
        plt.tight_layout()
        plt.show()

no_entrainment_save_names = [save_name_full, save_name_no_entrainment]
no_entrainment_readable_labels = ["Full model", "No entrainment"]

no_turbEk_save_names = [save_name_full, save_name_no_turb_or_ekman]
no_turbEk_readable_labels = ["Full model", "No turbulent air-sea flux or Ekman"]


# plot_pcs(save_names[0:8], readable_labels[0:8], obs=True, map_mask=map_mask)
plot_pcs(save_names[0:1], readable_labels[0:1], obs=True, map_mask=map_mask, plot_eof=False)
#plot_pcs(save_names[3:5], readable_labels[3:5], obs=True, map_mask=map_mask, plot_eof=False)
#plot_pcs(save_names[5:6], readable_labels[5:6], obs=True, map_mask=map_mask, plot_eof=False)
# plot_pcs(no_entrainment_save_names, no_entrainment_readable_labels, obs=True, map_mask=map_mask, plot_eof=False)
# plot_pcs(no_turbEk_save_names, no_turbEk_readable_labels, obs=True, map_mask=map_mask, plot_eof=False)
#
# plot_pcs(save_names[8:9], readable_labels[8:9], obs=True, map_mask=map_mask, plot_eof=True)
# plot_pcs(save_names[9:10], readable_labels[9:10], obs=True, map_mask=map_mask, plot_eof=False)
# plot_pcs(save_names[10:11], readable_labels[10:11], obs=True, map_mask=map_mask, plot_eof=False)

# obs_pc_projected(save_names[0:1], readable_labels[0:1], observed_anomaly, map_mask)
# pcs_power_spectrum(save_names[0:1], readable_labels[0:1], observed_anomaly, map_mask)

