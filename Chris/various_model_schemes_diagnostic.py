import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from SSTA.Chris.utils import make_movie
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset

ALL_SCHEMES_DATA_PATH = "../datasets/all_anomalies.nc"

all_schemes_ds = xr.open_dataset(ALL_SCHEMES_DATA_PATH, decode_times=False)

# make_movie(all_schemes_ds["CHRIS_PREV_CUR"], -10, 10, colorbar_label="Chris Prev-Cur Scheme")
# make_movie(all_schemes_ds["CHRIS_MEAN_K"], -10, 10, colorbar_label="Chris Mean-k Scheme")
# make_movie(all_schemes_ds["CHRIS_PREV_K"], -10, 10, colorbar_label="Chris Prev-k Scheme")
# make_movie(all_schemes_ds["CHRIS_CAPPED_EXPONENT"], -10, 10, colorbar_label="Chris Capped Exponent Scheme")
# make_movie(all_schemes_ds["EXPLICIT"], -10, 10, colorbar_label="Explicit Scheme")
# make_movie(all_schemes_ds["IMPLICIT"], -10, 10, colorbar_label="Implicit Scheme")
# make_movie(all_schemes_ds["SEMI_IMPLICIT"], -10, 10, colorbar_label="Semi-Implicit Scheme")


