"""
Find hbar with Holte's.

Chengyun Zhu
2025-10-11
"""

from IPython.display import display

import xarray as xr
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs

# from rgargo_read import load_and_prepare_dataset
# from rgargo_plot import visualise_dataset


def main():
    """Main function."""
    ds = xr.open_dataset("../datasets/Argo_mixedlayers_monthlyclim_04142022.nc")
    # ds = xr.open_dataset("../datasets/Argo_mixedlayers_all_04142022.nc")
    ds = ds.rename({'iLAT': 'LATITUDE', 'iLON': 'LONGITUDE', 'iMONTH': 'MONTH'})
    display(ds)

    MONTH = 1

    ds['mld_da_mean'].sel(MONTH=MONTH).plot(
        figsize=(12, 6), cmap='Blues', levels=200,
        vmin=0, vmax=500
    )
    plt.show()
    print(ds['mld_da_mean'].sel(MONTH=MONTH).max().max().item())

    ds['mld_dt_mean'].sel(MONTH=MONTH).plot(
        figsize=(12, 6), cmap='Blues', levels=200,
        vmin=0, vmax=500
    )
    plt.show()
    print(ds['mld_dt_mean'].sel(MONTH=MONTH).max().max().item())


if __name__ == "__main__":
    main()
