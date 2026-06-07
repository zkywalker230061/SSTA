#---1. Loading Packages and Defining Parameters --------
import numpy as np
import xarray as xr
from utils_ekman import ekman_current_anomaly, ekman_current_anomaly_salinity, coriolis_parameter, repeat_monthly_field
from utils_read_nc import month_idx, get_monthly_mean, get_anomaly



c_o = 4100                         #specific heat capacity of seawater = 4100 Jkg^-1K^-1
omega = 2*np.pi/(24*3600)         #Earth's angular velocity
#f = 2*omega*np.sin(phi)            #Coriolis Parameter


#---2. Loading Files ------------------
file_path = r"C:\Users\jason\MSciProject\era5_interpolated.nc"
derivatives_ds_path = r"C:\Users\jason\MSciProject\Mixed_Layer_Climatology_Derivatives.nc"


ds = xr.open_dataset(               # (TIME: 180, LATITUDE: 145, LONGITUDE: 360)
    file_path,                      # Data variables:
    engine="netcdf4",                       # avg_iews   (TIME, LATITUDE, LONGITUDE) float32 38MB ...
    decode_times=False,                     # avg_inss   (TIME, LATITUDE, LONGITUDE) float32 38MB ...       
    mask_and_scale=True)            # * TIME       (TIME) float32 720B 0.5 1.5 2.5 3.5 ... 176.5 177.5 178.5 179.5

derivatives_ds = xr.open_dataset(
    derivatives_ds_path,             # (MONTH: 12, LATITUDE: 145, LONGITUDE: 360)
    engine="netcdf4",               # * MONTH      (MONTH) int64 96B 1 2 3 4 5 6 7 8 9 10 11 12
    decode_times=False,             
    mask_and_scale=True)


#---3. Defining Windstress terms ------------
ds_tau_x = ds['avg_iews']           # (TIME: 180, LATITUDE: 145, LONGITUDE: 360)
ds_tau_y = ds['avg_inss']           # TIME  0.5 1.5 2.5 3.5 ... 176.5 177.5 178.5 179.5

monthly_mean_tau_x = get_monthly_mean(ds_tau_x)     #(MONTH: 12, LATITUDE: 145, LONGITUDE: 360)
monthly_mean_tau_y = get_monthly_mean(ds_tau_y)

tau_x_anom = get_anomaly(ds_tau_x, monthly_mean_tau_x)  #(TIME: 180, LATITUDE: 145, LONGITUDE: 360)
tau_y_anom = get_anomaly(ds_tau_y, monthly_mean_tau_y)

#---4. Defining Climatology Derivatives
dTm_dy_monthly = derivatives_ds["TEMP_LAT_GRAD"]  # (MONTH, LAT, LON)
dTm_dx_monthly = derivatives_ds["TEMP_LON_GRAD"]  # (MONTH, LAT, LON)

d_newTm_dy_monthly = derivatives_ds["UPDATED_TEMP_LAT_GRAD"]  # (MONTH, LAT, LON)
d_newTm_dx_monthly = derivatives_ds["UPDATED_TEMP_LON_GRAD"]  # (MONTH, LAT, LON)

dSm_dy_monthly = derivatives_ds["SALINITY_LAT_GRAD"]  # (MONTH, LAT, LON)
dSm_dx_monthly = derivatives_ds["SALINITY_LON_GRAD"]  # (MONTH, LAT, LON)

d_newSm_dy_monthly = derivatives_ds["UPDATED_SALINITY_LAT_GRAD"]  # (MONTH, LAT, LON)
d_newSm_dx_monthly = derivatives_ds["UPDATED_SALINITY_LON_GRAD"]  # (MONTH, LAT, LON)


#---5. Defining Coriolis Array -------------------
lat = ds["LATITUDE"]
f_2d = coriolis_parameter(lat).broadcast_like(ds_tau_x)


#---6. Computing Ekman Terms ------------------
# Temperature
Tm_Q_ek_anom, Tm_Q_ek_anom_x, Tm_Q_ek_anom_y = ekman_current_anomaly(tau_x_anom, tau_y_anom, dTm_dx_monthly, dTm_dy_monthly, f_2d)

updated_Tm_Q_ek_anom, updated_Tm_Q_ek_anom_x, updated_Tm_Q_ek_anom_y = ekman_current_anomaly(tau_x_anom, tau_y_anom, d_newTm_dx_monthly, d_newTm_dy_monthly, f_2d)

# Salinity
Sm_Q_ek_anom, Sm_Q_ek_anom_x, Sm_Q_ek_anom_y = ekman_current_anomaly_salinity(tau_x_anom, tau_y_anom, dSm_dx_monthly, dSm_dy_monthly, f_2d)

updated_Sm_Q_ek_anom, updated_Sm_Q_ek_anom_x, updated_Sm_Q_ek_anom_y = ekman_current_anomaly_salinity(tau_x_anom, tau_y_anom, d_newSm_dx_monthly, d_newSm_dy_monthly, f_2d)

#---7. Renaming Datasets -------------------------
# Full Datasets 
Tm_Q_ek_anom = Tm_Q_ek_anom.rename("TEMP_EKMAN_ANOM")
updated_Tm_Q_ek_anom = updated_Tm_Q_ek_anom.rename("UPDATED_TEMP_EKMAN_ANOM")

Sm_Q_ek_anom = Sm_Q_ek_anom.rename("SALINITY_EKMAN_ANOM")
updated_Sm_Q_ek_anom = updated_Sm_Q_ek_anom.rename("UPDATED_SALINITY_EKMAN_ANOM")

# Components
Tm_Q_ek_anom_x = Tm_Q_ek_anom_x.rename("TEMP_EKMAN_ANOM_X")
Tm_Q_ek_anom_y = Tm_Q_ek_anom_y.rename("TEMP_EKMAN_ANOM_Y")

updated_Tm_Q_ek_anom_x = updated_Tm_Q_ek_anom_x.rename("UPDATED_TEMP_EKMAN_ANOM_X")
updated_Tm_Q_ek_anom_y = updated_Tm_Q_ek_anom_y.rename("UPDATED_TEMP_EKMAN_ANOM_Y")


Sm_Q_ek_anom_x = Sm_Q_ek_anom_x.rename("SALINITY_EKMAN_ANOM_X")
Sm_Q_ek_anom_y = Sm_Q_ek_anom_y.rename("SALINITY_EKMAN_ANOM_Y")

updated_Sm_Q_ek_anom_x = updated_Sm_Q_ek_anom_x.rename("UPDATED_SALINITY_EKMAN_ANOM_X")
updated_Sm_Q_ek_anom_y = updated_Sm_Q_ek_anom_y.rename("UPDATED_SALINITY_EKMAN_ANOM_Y")

#---8. Merging Datasets ----------------------------------
ekman_ds = xr.merge([Tm_Q_ek_anom, updated_Tm_Q_ek_anom, Sm_Q_ek_anom, updated_Sm_Q_ek_anom])
ekman_component_ds = xr.merge([Tm_Q_ek_anom_x, Tm_Q_ek_anom_y,
                               updated_Tm_Q_ek_anom_x, updated_Tm_Q_ek_anom_y,
                               Sm_Q_ek_anom_x, Sm_Q_ek_anom_y,
                               updated_Sm_Q_ek_anom_x, updated_Sm_Q_ek_anom_y])


output_path_1 = r"C:\Users\jason\MSciProject\Ekman_Anomaly_Full_Datasets.nc"
output_path_2 = r"C:\Users\jason\MSciProject\Ekman_Anomaly_Decomposed_Datasets.nc"
ekman_ds.to_netcdf(output_path_1)
ekman_component_ds.to_netcdf(output_path_2)