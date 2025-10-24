"""
Plot h dataset from 2004 to 2018.

Chengyun Zhu
2024-10-24
"""

from IPython.display import display

# import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from rgargo_read import load_and_prepare_dataset


MAX_DEPTH = float(500)


def save_h_plot():
    """Save the h plot from 2004 to 2018."""

    ds_h = load_and_prepare_dataset(
        "../datasets/Mixed_Layer_Depth_Pressure-(2004-2018).nc",
    )
    h = ds_h['MLD_PRESSURE']
    display(h)
    for month in range(0, 180):
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.contourf(
            h['LONGITUDE'], h['LATITUDE'], h.isel(TIME=month),
            cmap='Blues',
            levels=200,
            vmin=0,
            vmax=MAX_DEPTH
        )
        ax.coastlines()

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
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

        # plt.colorbar(label=h.attrs.get('units'))
        title_str = (
            h.name
            + f", Time = {month+0.5} months since 2004-01-01"
        )
        plt.title(title_str)

        plt.savefig(f"./h/h_plot_month_{month+1}.png", dpi=600)
        plt.show()


def main():
    """Main function to plot h dataset."""

    save_h_plot()


if __name__ == "__main__":
    main()
