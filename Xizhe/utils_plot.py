import matplotlib as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)

def plot_single(da, title="", vmin=None, vmax=None):    
    plt.figure(figsize=(8, 4))
    im = da.plot(
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={"label": da.name if da.name else ""}
    )
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

def plot_two_subplots(da1, da2, vmin = None, vmax = None, titles=("A", "B"), cmap="RdBu_r"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vmax = vmin
    vmin = vmax

    im1 = da1.plot(ax=axes[0], add_colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(titles[0])

    im2 = da2.plot(ax=axes[1], add_colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])

    # Add shared colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.05)
    cbar.set_label("Units")


def simulation(ds1, ds2, vmin = None, vmax = None, cmap="RdBu_r", titles=["A","B"], cbar_label = ''):
    anim_times = ds1.TIME.values
    lons = ds1.LONGITUDE.values
    lats = ds1.LATITUDE.values

    fig = plt.figure(figsize=(16, 6))

    ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
    mesh_1 = ax1.pcolormesh(lons, lats, ds1.isel(TIME=0), cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.coastlines()
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    gl1.xlines = False
    gl1.ylines = False
    gl1.ylocator = LatitudeLocator()
    gl1.xformatter = LongitudeFormatter()
    gl1.yformatter = LatitudeFormatter()
    gl1.ylabel_style = {'size': 12, 'color': 'gray'}
    gl1.xlabel_style = {'size': 12, 'color': 'gray'}
    ax1.set_title(titles[0])

    ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
    mesh_2 = ax2.pcolormesh(lons, lats, ds1.isel(TIME=0), cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.coastlines()
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    gl2.left_labels = True
    gl2.xlines = False
    gl2.ylines = False
    gl2.ylocator = LatitudeLocator()
    gl2.xformatter = LongitudeFormatter()
    gl2.yformatter = LatitudeFormatter()
    gl2.ylabel_style = {'size': 12, 'color': 'gray'}
    gl2.xlabel_style = {'size': 12, 'color': 'gray'}
    ax2.set_title(titles[1])

    cbar = fig.colorbar(mesh_1, ax=[ax1, ax2], shrink=0.8, label=cbar_label)

    animation = FuncAnimation(fig, update, frames=len(anim_times), interval=600, blit=False)
    animation.save('.mp4', writer='ffmpeg', fps=10)
    plt.show

    def update(frame):
        # Update explicit
        Z_1 = ds1.isel(TIME=frame).values
        mesh_1.set_array(Z_1.ravel())

        Z_2 = ds2.isel(TIME=frame).values
        mesh_2.set_array(Z_2.ravel())

        current_time = anim_times[frame]
        month_in_year = (current_time % 12) + 0.5
        year = 2014 + (current_time // 12)

        ax1.set_title(f'in Month {month_in_year} in year {year}')
        ax2.set_title(f'in Month {month_in_year} in year {year}')

        return [mesh_1, mesh_2]