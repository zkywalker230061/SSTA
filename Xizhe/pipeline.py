"""
Pipeline to run Temp/Salinity Data

"""
#%%
#----1. Importing Packages-----------------------------------------------
from calculate_Tm_Sm import (
    z_to_xarray,
    vertical_integral,
    _full_field,
    depth_dbar_to_meter,
    mld_dbar_to_meter
)
from read_nc import fix_rg_time, fix_longitude_coord
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

#%%
#----2. Config -----------------------------------------------------------

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
    label="Mean Temperature (°C)",
    plot_cmap="RdYlBu_r",
    plot_vmin=-2,
    plot_vmax=35,
    output_filename="Mean_Temp_(2004_2018).nc",
    plot_title="Mean Temperature Using h Field"
)

SALINITY = DataChoice(
    data_path="/Users/xxz/Desktop/SSTA/datasets/RG_ArgoClim_Salinity_2019.nc",
    mean_data="ARGO_SALINITY_MEAN",
    anom_data="ARGO_SALINITY_ANOMALY",
    label="Mean Salinity",
    plot_cmap="RdYlBu_r",
    plot_vmin=30,
    plot_vmax=45,
    output_filename="Mean_Sal_(2004_2018).nc",
    plot_title="Mean Salinity Using h Field"
)

PRESSURE_PATH = "/Users/xxz/Desktop/SSTA/datasets/Mixed_Layer_Depth_Pressure (2004-2018).nc"
PRESSURE_VARNAME = "MLD_PRESSURE"

#%%
#----3. Global Params--------------------------------------------------------
ZDIM = "PRESSURE"
YDIM = "LATITUDE"
XDIM = "LONGITUDE"
TDIM = "TIME"
SAMPLE_DATE = "2010-06-01"

#%%
#---4. Read Pressure Field---------------------------------------------------
def load_pressure_data(path: str, varname: str, *, compute_time_mode: str = "datetime",) -> xr.DataArray:
    """Load MLD in PRESSURE units, fix time, convert to meters (positive down)."""

    ds = xr.open_dataset(path, engine="netcdf4", decode_times=False, mask_and_scale=True)
    ds = fix_rg_time(ds, mode=compute_time_mode)

    pressure = ds[varname] # Coordinates = (TIME: 180, LATITUDE: 145, LONGITUDE: 360)
    lat_1D = ds["LATITUDE"]
    lat_3D = xr.broadcast(lat_1D, pressure)[0].transpose(*pressure.dims)
    depth_m   = mld_dbar_to_meter(pressure, lat_3D)
    depth_m   = fix_longitude_coord(depth_m)

    # print('depth_bar:\n',depth_bar, depth_bar.shape)
    # print(lat_3D)
    # print('depth_m:\n',depth_m)
    # print('depth_m after fix_longitude:\n', depth_m)
    return depth_m

#%%

#---5. Core Pipeline for Tm and Sm-------------------------------------------
def process_data(
        spec: DataChoice,
        *,
        do_plot: bool = True,
        do_save: bool = False,
        compute_time_mode: str = "datetime",
        save_time_mode: str = "None",
        z_positive_down: bool = True,
    ) -> xr.DataArray:

    dataset = xr.open_dataset(spec.data_path, engine="netcdf4",
                               decode_times=False,
                               mask_and_scale=True)
    ds = fix_rg_time(dataset, mode=compute_time_mode)

    # Building the field from the chosen dataset
    field_mean = ds[spec.mean_data]
    field_anom = ds[spec.anom_data]
    f_field = _full_field(field_mean, field_anom)
    f_field = fix_longitude_coord(f_field)

    # Calculating depth from pressure
    z_m = depth_dbar_to_meter(ds["PRESSURE"], ds["LATITUDE"])
    z = z_m.broadcast_like(f_field)
    # z_signed = z if z_positive_down else -z

    # Integrating over vertical
    depth_m = load_pressure_data(PRESSURE_PATH, PRESSURE_VARNAME, compute_time_mode=compute_time_mode)

    print("T dims:", f_field.dims)
    print("z dims:", z.dims, " (is tuple? ", isinstance(z, tuple), ")")
    print("depth_m dims:", depth_m.dims)
    assert isinstance(z, xr.DataArray), "z is a tuple — index [0] or use broadcast_like"
    assert isinstance(depth_m, xr.DataArray), "depth_m must be a DataArray"
    assert YDIM in f_field.dims and ZDIM in f_field.dims and XDIM in f_field.dims and TDIM in f_field.dims
    assert YDIM in z.dims and ZDIM in z.dims
    assert TDIM in depth_m.dims and YDIM in depth_m.dims and XDIM in depth_m.dims

    integral = vertical_integral(f_field, z, depth_m)

    # Plotting 
    if do_plot and TDIM in integral.dims:
        target = pd.Timestamp(SAMPLE_DATE)          # 2010-06-01 00:00:00
        if target in integral[TDIM].values:
            t0 = integral.sel({TDIM: target})
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

#%%
#---- 6. Running Code Function -----------------------------------------------------

def run(batch: Dict[str, DataChoice], **kwargs) -> Dict[str, xr.DataArray]:
    """
    Processes multiple variables in one go
    """
    return {name: process_data(spec, **kwargs) for name, spec in batch.items()}

#-----7. Main-----------------------------------------------------------------------
#%%
if __name__ =="__main__":
    results = run(
    {
        "temperature": TEMPERATURE,
        # "salinity": SALINITY,
    },
        do_plot = True,
        do_save = False,
        compute_time_mode="datetime",
        save_time_mode="NONE",
        z_positive_down=True,
    )

    for a,b in results.items():
        print(a, b.shape, b)
# %%
