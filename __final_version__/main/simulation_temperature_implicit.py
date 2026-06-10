import matplotlib
from Chris.do_implicit_simulation import run_model

matplotlib.use('TkAgg')

"""
Naming scheme:
Surface, Ekman, Entrainment, Geostrophic in that order. 0 if off, 1 if on
e.g.
1001
=> surface and geostrophic on, Ekman and entrainment off
then a bunch of other parameters in the filename. Basically, always use get_save_name to determine the name of a file.
The filenames are not intended to be human-readable.
"""

# Set parameters for the model
INCLUDE_SURFACE = True
INCLUDE_EKMAN_ANOM_ADVECTION = False
INCLUDE_EKMAN_MEAN_ADVECTION = False
INCLUDE_ENTRAINMENT = True
INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING = False
INCLUDE_GEOSTROPHIC_ANOM_ADVECTION = False
INCLUDE_GEOSTROPHIC_MEAN_ADVECTION = False

SPLIT_SURFACE = True
INCLUDE_RADIATIVE_SURFACE = False
INCLUDE_TURBULENT_SURFACE = True

# geostrophic displacement integral: https://egusphere.copernicus.org/preprints/2025/egusphere-2025-3039/egusphere-2025-3039.pdf
USE_DOWNLOADED_SSH = False
USE_OTHER_MLD = False
USE_MAX_GRADIENT_METHOD = True
USE_LOG_FOR_ENTRAINMENT = False
DATA_TO_2025 = True
rho_0 = 1025.0
c_0 = 4100.0
gamma_0 = 15.0
g = 9.81
adjust_mld = 0.0

# run the model from do_implicit_simulation
run_model(INCLUDE_SURFACE, SPLIT_SURFACE, INCLUDE_RADIATIVE_SURFACE, INCLUDE_TURBULENT_SURFACE, INCLUDE_EKMAN_ANOM_ADVECTION, INCLUDE_EKMAN_MEAN_ADVECTION, INCLUDE_ENTRAINMENT, INCLUDE_ENTRAINMENT_VEL_ANOMALY_FORCING, INCLUDE_GEOSTROPHIC_ANOM_ADVECTION, INCLUDE_GEOSTROPHIC_MEAN_ADVECTION, USE_DOWNLOADED_SSH=False, USE_OTHER_MLD=False, USE_MAX_GRADIENT_METHOD=True, USE_LOG_FOR_ENTRAINMENT=False, gamma_0=15.0, DATA_TO_2025=True, adjust_mld=0.0)
