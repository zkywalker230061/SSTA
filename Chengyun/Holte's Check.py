"""
Find hbar with Holte's.

Chengyun Zhu
2025-10-11
"""

from IPython.display import display

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
import xesmf as xe  # NOTE: This argolise must be run under conda environment with xesmf installed.

from rgargo_read import load_and_prepare_dataset
# from rgargo_plot import visualise_dataset
from error_analysis import get_mean_of_error

MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3,
    'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9,
    'Oct': 10, 'Nov': 11, 'Dec': 12
}


def main():
    """Main function."""

    holte_ds = xr.open_dataset("../datasets/Argo_mixedlayers_monthlyclim_04142022.nc")
    # holte_ds = xr.open_dataset("../datasets/Argo_mixedlayers_all_04142022.nc")

    # copilot ----------------------------------------------------------------
    # normalize names to simple lat/lon and MONTH
    holte_ds = holte_ds.rename({'iLAT': 'lat', 'iLON': 'lon', 'iMONTH': 'MONTH'})

    # helper: find a variable whose name contains any of the tokens
    def _find_var(ds, tokens):
        for v in ds.variables:
            lname = v.lower()
            for t in tokens:
                if t in lname:
                    return v
        return None

    if 'lat' not in holte_ds.coords:
        lat_var = _find_var(holte_ds, ['lat', 'latitude'])
        if lat_var:
            if lat_var != 'lat':
                holte_ds = holte_ds.rename({lat_var: 'lat'})
            if 'lat' not in holte_ds.coords:
                holte_ds = holte_ds.set_coords('lat')
        else:
            # fallback: create index coords
            nlat = holte_ds.sizes.get('lat', None)
            if nlat is not None:
                holte_ds = holte_ds.assign_coords(lat=('lat', np.arange(nlat)))

    if 'lon' not in holte_ds.coords:
        lon_var = _find_var(holte_ds, ['lon', 'longitude'])
        if lon_var:
            if lon_var != 'lon':
                holte_ds = holte_ds.rename({lon_var: 'lon'})
            if 'lon' not in holte_ds.coords:
                holte_ds = holte_ds.set_coords('lon')
        else:
            nlon = holte_ds.sizes.get('lon', None)
            if nlon is not None:
                holte_ds = holte_ds.assign_coords(lon=('lon', np.arange(nlon)))

    # set CF attributes so cf_xarray/xesmf recognise them
    if 'lat' in holte_ds.coords:
        holte_ds['lat'].attrs.update(
            {'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'}
        )
    if 'lon' in holte_ds.coords:
        holte_ds['lon'].attrs.update(
            {'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'}
        )

    # quick diagnostics (remove or comment out after confirming)
    print("coords:", list(holte_ds.coords))
    print("cf view:", repr(holte_ds.cf))
    # ------------------------------------------------------------------------

    holte_hbar_ds = holte_ds[['mld_da_mean', 'mld_dt_mean']]
    display(holte_hbar_ds)

    argo_hbar_ds = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc"
    )
    display(argo_hbar_ds)

    regridder = xe.Regridder(holte_hbar_ds, argo_hbar_ds, "conservative")
    holte_hbar_ds_interpolated = regridder(holte_hbar_ds)
    display(holte_hbar_ds_interpolated)

    # MONTH = 1

    # holte_hbar_ds['mld_da_mean'].sel(MONTH=MONTH).plot(
    #     figsize=(12, 6), cmap='Blues', levels=200,
    #     vmin=0, vmax=500
    # )
    # plt.show()
    # print(holte_hbar_ds['mld_da_mean'].sel(MONTH=MONTH).max().max().item())

    # holte_hbar_ds['mld_dt_mean'].sel(MONTH=MONTH).plot(
    #     figsize=(12, 6), cmap='Blues', levels=200,
    #     vmin=0, vmax=500
    # )
    # plt.show()
    # print(holte_hbar_ds['mld_dt_mean'].sel(MONTH=MONTH).max().max().item())

    argo_hbar = argo_hbar_ds['MONTHLY_MEAN_MLD_PRESSURE']
    holte_hbar_da = holte_hbar_ds_interpolated['mld_da_mean']
    holte_hbar_dt = holte_hbar_ds_interpolated['mld_dt_mean']
    # set places where argo_hbar is -inf to -inf in holte datasets too
    holte_hbar_da = holte_hbar_da.where(argo_hbar != -np.inf, -np.inf)
    holte_hbar_dt = holte_hbar_dt.where(argo_hbar != -np.inf, -np.inf)

    diff_da = argo_hbar - holte_hbar_da
    diff_dt = argo_hbar - holte_hbar_dt

    diff_da.sel(MONTH=1).plot(
        figsize=(12, 6), cmap='RdBu_r', levels=200,
        vmin=-40, vmax=20
    )
    plt.show()
    print("[", diff_da.max().item(), diff_da.min().item(), "]")
    print(diff_da.mean().item(), "+-", diff_da.std().item(), "dbar")

    diff_dt.sel(MONTH=1).plot(
        figsize=(12, 6), cmap='RdBu_r', levels=200,
        vmin=-40, vmax=20
    )
    plt.show()
    print("[", diff_dt.max().item(), diff_dt.min().item(), "]")
    print(diff_dt.mean().item(), "+-", diff_dt.std().item(), "dbar")

    mean_error_da = get_mean_of_error(diff_da)
    mean_error_da_std = diff_da.std(dim=['LATITUDE', 'LONGITUDE'])
    mean_error_dt = get_mean_of_error(diff_dt)
    mean_error_dt_std = diff_dt.std(dim=['LATITUDE', 'LONGITUDE'])

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        list(MONTHS.values()),
        mean_error_da,
        yerr=mean_error_da_std,
        fmt='o',
        color='#66CCFF',
        alpha=0.5,
        capsize=5,
        label="Std Dev (da)"
    )
    plt.errorbar(
        list(MONTHS.values()),
        mean_error_dt,
        yerr=mean_error_dt_std,
        fmt='o',
        color='#FF9966',
        alpha=0.5,
        capsize=5,
        label="Std Dev (dt)"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
