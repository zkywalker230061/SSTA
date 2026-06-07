"""
Calculate significance of correlation.

Chris O'Sullivan
2026-3-21
"""

import xarray as xr
import numpy as np
from scipy.stats import pearsonr
from scipy import fft
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LatitudeLocator, LongitudeFormatter, LatitudeFormatter

# based on https://journals.ametsoc.org/view/journals/clim/10/9/1520-0442_1997_010_2147_amtets_2.0.co_2.xml


def mutual_information(x, y, mask=False):
    if mask:        # for when this is used in other files
        ocean_mask = np.isfinite(x) & np.isfinite(y)
        if ocean_mask.sum() < 10:
            return np.nan
        x = x[ocean_mask]
        y = y[ocean_mask]
    return mutual_info_regression(x.reshape(-1, 1), y, random_state=1)[0]   # force same random state for reproducibility


def random_phase(da, size):     # create "size" amount of random phases
    da_fourier = fft.rfft(da)
    da_fourier_randomised_phases = np.abs(da_fourier) * np.exp(1j * np.random.uniform(0, 2 * np.pi, (size, len(da_fourier))))
    da_fourier_randomised_phases[:, 0] = da_fourier[0].real        # remove imaginary components from DC signal
    if len(da) % 2 == 0:    # remove imaginary components from Nyquist frequency if present
        da_fourier_randomised_phases[:, -1] = da_fourier[-1].real
    return fft.irfft(da_fourier_randomised_phases, n=len(da))


def get_significance_at_a_gridpoint(model, obs, resamples=100, test_statistic="PEARSON"):
    mask = np.isfinite(model) & np.isfinite(obs)
    if mask.sum() < 10:
        return np.nan, np.nan
    model = model[mask]
    obs = obs[mask]

    random_models = random_phase(model, resamples)

    if test_statistic == "PEARSON":
        statistic = pearsonr(model, obs)[0]
        obs_normalised = (obs - obs.mean()) / obs.std()
        random_model_normalised = (random_models - random_models.mean(axis=1, keepdims=True)) / random_models.std(axis=1, keepdims=True)
        resampled_correlations = (random_model_normalised * obs_normalised).mean(axis=1)
    elif test_statistic == "PEARSON_VARIANCE":
        statistic = pearsonr((model - np.mean(model)) ** 2, (obs - np.mean(obs)) ** 2)[0]
        obs_square = (obs - np.mean(obs)) ** 2
        obs_square_normalised = (obs_square - obs_square.mean()) / obs_square.std()
        random_model_square = (random_models - random_models.mean(axis=1, keepdims=True)) ** 2
        random_model_square_normalised = (random_model_square - random_model_square.mean(axis=1, keepdims=True)) / random_model_square.std(axis=1, keepdims=True)
        resampled_correlations = (random_model_square_normalised * obs_square_normalised).mean(axis=1)
    elif test_statistic == "MI":
        statistic = mutual_information(model, obs)
        resampled_correlations = []
        for i in range(resamples):
            resampled_correlations.append(mutual_information(random_models[i], obs))
        resampled_correlations = np.array(resampled_correlations)
    else:
        print("test statistic invalid")
        return None
    p_value = np.mean(np.abs(resampled_correlations) >= np.abs(statistic))
    return np.array(statistic), np.array(p_value)


def handle_false_detections(p_values, alpha=0.05):
    # checking significance globally therefore expect many false positives
    flat = p_values.values.flatten()
    ocean_mask = np.isfinite(flat)
    corrected_p_values = np.full_like(flat, np.nan)
    corrected_p_values[ocean_mask]= multipletests(flat[ocean_mask], method='fdr_bh')[1]
    corrected_p_values = corrected_p_values.reshape(p_values.shape)
    significant_p_values = corrected_p_values < alpha
    significant_p_values_da = xr.DataArray(significant_p_values, coords=p_values.coords, dims=p_values.dims)
    return significant_p_values_da


def get_significance(model_da, obs_da, resamples, test_statistic="PEARSON", alpha=0.05):
    correlations, p_values = xr.apply_ufunc(get_significance_at_a_gridpoint, model_da, obs_da, kwargs={'resamples': resamples, 'test_statistic': test_statistic}, input_core_dims=[['TIME'], ['TIME']], output_core_dims=[[], []], vectorize=True, dask='parallelized', output_dtypes=[float, float])
    corrected_p_values = handle_false_detections(p_values, alpha=alpha)
    return correlations, corrected_p_values


# simulated = xr.open_dataset(
#     "datasets/Simulation-TA.nc"
# )['ANOMALY_TA_SIMULATED']
# observed = xr.open_dataset(
#     "datasets/reynolds_sst_Anomalies-(2004-2025)_no_2004.nc", decode_times=False
# )['ANOMALY_SST']

simulated = xr.open_dataset(
    "datasets/Simulation-SA.nc"
)['ANOMALY_SA_SIMULATED']
observed = xr.open_dataset(
    "datasets/Mixed_Layer_Salinity_Anomalies-(2004-2025).nc"
)['ANOMALY_ML_SALINITY']

correlations, significant_mask = get_significance(
    simulated, observed,
    resamples=1000, test_statistic="PEARSON", alpha=0.05
)

# corr plot
# ----------------------------------------------------------------------------
corr = correlations

plt.figure(figsize=(10, 5), dpi=600)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(
    corr['LONGITUDE'], corr['LATITUDE'], corr,
    cmap='nipy_spectral',
    vmin=-1, vmax=1
)
ax.contourf(
    corr['LONGITUDE'],
    corr['LATITUDE'],
    significant_mask,
    levels=[0.5, 1],
    hatches=['xx'],
    colors='none',
    transform=ccrs.PlateCarree(),
    zorder=3
)
ax.coastlines()

plt.xlim(-180, 180)
plt.ylim(-90, 90)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=2, color='gray', alpha=0.5, linestyle='--'
)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.xlines = False
gl.ylines = False
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.ylabel_style = {'size': 15, 'color': 'gray'}
gl.xlabel_style = {'size': 15, 'color': 'gray'}

plt.title(f"Correlation: Spatial Mean {corr.mean().item():.2f}", fontsize=20, loc='left')

plt.colorbar(
    shrink=0.75, orientation='horizontal', aspect=40, pad=0.1
)

plt.show()
