"""
Finding hbar from Mixed_Layer_Depth_Pressure-(2004-2018).nc.

Chengyun Zhu
2024-10-24
"""

# from IPython.display import display

# import xarray as xr
# import numpy as np

from rgargo_read import load_and_prepare_dataset
from rgargo_analysis import get_monthly_mean
from rgargo_plot import visualise_dataset


MAX_DEPTH = float(500)


def save_monthly_mld():
    """Save the monthly mean of the mixed layer depth dataset."""

    ds_mld = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc",
    )
    # display(ds_mld)
    mld = ds_mld['MLD_PRESSURE']
    # display(mld)
    mld_monthly_mean = get_monthly_mean(mld)
    # display(mld_monthly_mean)
    visualise_dataset(
        mld_monthly_mean.sel(MONTH=3, method='nearest'),
        cmap='Blues',
        vmin=0, vmax=MAX_DEPTH
    )
    mld_monthly_mean.to_netcdf("../datasets/Mixed_Layer_Depth_Pressure-Seasonal_Cycle_Mean.nc")


def main():
    """Main function to find hbar."""

    save_monthly_mld()


if __name__ == "__main__":
    main()
