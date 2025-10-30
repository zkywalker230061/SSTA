"""
Pipeline to run Temp/Salinity Data

Edited by JY

"""

#----1. Importing Packages----------------------------
from calculate_Tm_Sm import (
    depth_from_pressure,
    z_to_xarray,
    vertical_integral,
    _full_field
)
from read_nc import fix_rg_time
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


#----2. Config -----------------------------------

@dataclass
class DataChoice:
    data_path: str
    mean_data: str
    anom_data: str
    label: str
    plot_cmap: str
    plot_vmin: Optional[float]
    plot_vmax: Optional[float]
    output_filename: Optional[str]
    plot_title: str

TEMPERATURE = DataChoice(
    data_path="/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Temperature_2019.nc",
    mean_data="ARGO_TEMPERATURE_MEAN",
    anom_data="ARGO_TEMPERATURE_ANOMALY",
    label="Mean Temperature (°C, 0-100m)",
    plot_cmap="RdY1Bu_r",
    plot_vmin=-2,
    plot_vmax=40,
    output_filename="Mean_Temp_0_100m_2004_2018.nc",
    plot_title="Upper 100m Mean Temperature"
)

SALINITY = DataChoice(
    data_path="/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
    mean_data="ARGO_SALINITY_MEAN",
    anom_data="ARGO_SALINITY_ANOMALY",
    label="Mean Salinity",
    plot_cmap="RdY1Bu_r",
    plot_vmin=30,
    plot_vmax=40,
    output_filename="Mean_Sal_0_100m_2004_2018.nc",
    plot_title="Upper 100m Mean Salinity"
)

#----3. Global Params---------------------
h_meters = 100.0
ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
TDIM = "TIME"
SAMPLE_DATE = "2010-06-01"

#---4. 


#---5. Core Pipeline ----------------------
def process_data(
        spec: DataChoice,
        *,
        do_plot: bool = True,
        do_save: bool = False,
        compute_time_mode: str = "datetime",
        save_time_mode: str = "None",
        z_positive_down: bool = True,
        h: float = h_meters
) -> xr.DataArray:
    """
    Fix Later
    """
    raw_data = xr.open_dataset(spec.data_path, engine="netcdf4",
                               decode_times=False,
                               mask_and_scale=True)
    ds = fix_rg_time(raw_data, mode=compute_time_mode)

    # Building the field from the chosen dataset
    field_mean = ds[spec.mean_data]
    field_anom = ds[spec.anom_data]
    f_field = _full_field(field_mean, field_anom)

    # Calculating depth from pressure
    depth_m = depth_from_pressure(ds["PRESSURE"], ds["LATITUDE"])
    z = z_to_xarray(depth_m, f_field)
    z_signed = z if z_positive_down else -z

    # Integrating over vertical
    integral = vertical_integral(f_field, z_signed)

    # Plotting 
    if do_plot and TDIM in integral.dims:
        times_str = integral[TDIM].astype(str)
        if (times_str == SAMPLE_DATE).any():
            t0 = integral.sel({TDIM: SAMPLE_DATE})
            plt.figure(figsize=(10, 5))
            pc = plt.pcolormesh(
                t0[XDIM], t0[YDIM], t0,
                cmap=spec.plot_cmap, shading="auto",
                vmin=spec.plot_vmin, vmax=spec.plot_vmax,
            )
            plt.colorbar(pc, label=spec.label)
            plt.title(f"{spec.plot_title} - {SAMPLE_DATE}")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.show()
    

    return integral


#---- 6. Running Code Function 

def run(batch: Dict[str, DataChoice], **kwargs) -> Dict[str, xr.DataArray]:
    """
    Processes multiple variables in one go
    """
    return {name: process_data(spec, **kwargs) for name, spec in batch.items()}

if __name__ =="__main__":
    results = run(
        {"temperature": TEMPERATURE},
        do_plot = True,
        do_save = False,
        compute_time_mode="datetime",
        save_time_mode="NONE",
        z_positive_down=True,
        h=h_meters
    )

    for a,b in results.items():
        print(a, b.shape, b)







