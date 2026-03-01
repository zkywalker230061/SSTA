import xarray as xr
import numpy as np
from scipy.stats import pearsonr
from scipy import fft
from statsmodels.stats.multitest import multipletests
import dask
from dask.diagnostics import ProgressBar
from sklearn.metrics import normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer

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
    model_da = model_da.chunk({'LATITUDE': 10, 'LONGITUDE': 10, 'TIME': -1})
    obs_da = obs_da.chunk({'LATITUDE': 10, 'LONGITUDE': 10, 'TIME': -1})
    correlations, p_values = xr.apply_ufunc(get_significance_at_a_gridpoint, model_da, obs_da, kwargs={'resamples': resamples, 'test_statistic': test_statistic}, input_core_dims=[['TIME'], ['TIME']], output_core_dims=[[], []], vectorize=True, dask='parallelized', output_dtypes=[float, float])
    with ProgressBar():
        correlations, p_values = dask.compute(correlations, p_values)
    corrected_p_values = handle_false_detections(p_values, alpha=alpha)
    return correlations, corrected_p_values
