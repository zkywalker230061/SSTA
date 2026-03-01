"""
Useful utility functions for plotting.

Chengyun Zhu
2026-1-8
"""

# import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation

# matplotlib.use('TkAgg')
plt.rcParams['animation.ffmpeg_path'] = r'C://FFmpeg/bin/ffmpeg.exe'


def make_movie(
    dataset,
    vmin, vmax,
    cmap='nipy_spectral',
    title='SSTA',
    save_path=None
):
    """
    Create an animation of the dataset over time.

    Parameters
    ----------
    dataset : xarray.DataArray
        The dataset to animate. Must have dimensions 'TIME', 'LATITUDE', and 'LONGITUDE'.
    vmin : float
        The minimum value for the color scale.
    vmax : float
        The maximum value for the color scale.
    cmap : str, optional
        The colormap to use for the plot. Default is 'nipy_spectral'.
    title : str, optional
        The title to display on the plot. Default is 'SSTA'.
    save_path : str, optional
        The path to save the animation as a video file. Default is None.
        If None, the animation will not be saved.
    """

    times = dataset.TIME.values

    fig, axes = plt.subplots(
        nrows=1, ncols=1,
        figsize=(40, 10),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    plt.subplots_adjust(wspace=0.1, hspace=0)

    ax1 = axes[0]
    plot1 = ax1.pcolormesh(
        dataset['LONGITUDE'], dataset['LATITUDE'], dataset.isel(TIME=0),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.coastlines()
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_title(f'{title}, Time = 0.5', fontsize=20)

    fig.colorbar(plot1, ax=axes.ravel().tolist())

    def update(frame):
        data1 = dataset.isel(TIME=frame).values
        plot1.set_array(data1.ravel())

        ax1.set_title(f'{title}, Time = {times[frame]}', fontsize=20)

        return plot1

    animation = FuncAnimation(fig, update, frames=len(times), interval=1000, blit=False)
    plt.show()
    if save_path is not None:
        animation.save(save_path, writer='ffmpeg', fps=1)


def make_movie_2(
    dataset1, dataset2,
    vmin, vmax,
    cmap='nipy_spectral',
    title=None,
    save_path=None
):
    """
    Create an animation of two datasets over time, side by side.

    Parameters
    ----------
    dataset1 : xarray.DataArray
        The first dataset to animate. Must have dimensions 'TIME', 'LATITUDE', and 'LONGITUDE'.
    dataset2 : xarray.DataArray
        The second dataset to animate. Must have dimensions 'TIME', 'LATITUDE', and 'LONGITUDE'.
    vmin : float
        The minimum value for the color scale.
    vmax : float
        The maximum value for the color scale.
    cmap : str, optional
        The colormap to use for the plot. Default is 'nipy_spectral'.
    title : list of str, optional
        The titles to display on the plots. Default is ['Reynolds SSTA', 'Simulated SSTA'].
    save_path : str, optional
        The path to save the animation as a video file. Default is None.
        If None, the animation will not be saved.
    """
    if title is None:
        title = ['Reynolds SSTA', 'Simulated SSTA']

    times = dataset1.TIME.values

    fig, axes = plt.subplots(
        nrows=1, ncols=2,
        figsize=(40, 10),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    plt.subplots_adjust(wspace=0.1, hspace=0)

    ax1 = axes[0]
    plot1 = ax1.pcolormesh(
        dataset1['LONGITUDE'], dataset1['LATITUDE'], dataset1.isel(TIME=0),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.coastlines()
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.set_title(f'{title[0]}, Time = 0.5', fontsize=20)

    ax2 = axes[1]
    plot2 = ax2.pcolormesh(
        dataset2['LONGITUDE'], dataset2['LATITUDE'], dataset2.isel(TIME=0),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.coastlines()
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.set_title(f'{title[1]}, Time = 0.5', fontsize=20)

    fig.colorbar(plot2, ax=axes.ravel().tolist())

    def update(frame):
        data1 = dataset1.isel(TIME=frame).values
        data2 = dataset2.isel(TIME=frame).values

        plot1.set_array(data1.ravel())
        plot2.set_array(data2.ravel())

        ax1.set_title(f'{title[0]}, Time = {times[frame]}', fontsize=20)
        ax2.set_title(f'{title[1]}, Time = {times[frame]}', fontsize=20)

        return plot1, plot2

    animation = FuncAnimation(fig, update, frames=len(times), interval=1000, blit=False)
    plt.show()
    if save_path is not None:
        animation.save(save_path, writer='ffmpeg', fps=1)


if __name__ == "__main__":

    pass
