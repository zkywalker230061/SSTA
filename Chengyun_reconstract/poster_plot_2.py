import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs


font = {'size': 30}
matplotlib.rc('font', **font)
norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

EOFs = xr.open_dataset("datasets/Chris/eof_first_mode_model.nc")["__xarray_dataarray_variable__"]
EOFs_obs = xr.open_dataset("datasets/Chris/eof_first_mode_obs.nc")["__xarray_dataarray_variable__"]

CONSIDER_OBSERVATIONS = True
# def plot_spatial_pattern_EOFs():  # plot EOFs (spatial patterns) for the first k modes
#     k_range = 1
#     fig, axs = plt.subplots(
#         k_range, 1,
#         figsize=(20, 10),
#         subplot_kw={'projection': ccrs.PlateCarree()}
#     )
#     # fig.suptitle("EOF of " + "Impl" + " scheme")
#     fig.tight_layout()
#     norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)

#     pcolormesh = axs.pcolormesh(EOFs.LONGITUDE.values, EOFs.LATITUDE.values, EOFs, cmap='RdBu_r', norm=norm)
#     axs.set_xlabel("Longitude")
#     axs.set_ylabel("Latitude")
#     axs.coastlines()
#     cbar = fig.colorbar(pcolormesh, ax=axs, label="EOF spatial pattern (standardised)")
#     # pcolormesh.set_clim(vmin=-2, vmax=2)
#     # plt.savefig("../results/eof_spatial_" + to_plot_name + ".jpg", dpi=400)
#     # EOF_standard.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_poster/eof_first_mode_model.nc")
#     # plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_poster/first_eof.png", dpi=400)
#     plt.show()
#     if CONSIDER_OBSERVATIONS:
#         fig, axs = plt.subplots(
#             k_range, 1, 
#             figsize=(20, 10),
#             subplot_kw={'projection': ccrs.PlateCarree()}
#         )
#         # fig.suptitle("EOF of observations")
#         fig.tight_layout()
#         norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
#         pcolormesh = axs.pcolormesh(EOFs_obs.LONGITUDE.values, EOFs_obs.LATITUDE.values, EOFs_obs,cmap='RdBu_r', norm=norm)
#         axs.set_xlabel("Longitude")
#         axs.set_ylabel("Latitude")
#         axs.coastlines()
#         cbar = fig.colorbar(pcolormesh, ax=axs, label="EOF spatial pattern (standardised)")
#         # pcolormesh.set_clim(vmin=-2, vmax=2)
#         # plt.savefig("../results/eof_spatial_obs.jpg", dpi=400)
#         # EOF_obs_standard.to_netcdf("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_poster/eof_first_mode_obs.nc")
#         # plt.savefig("/Volumes/G-DRIVE ArmorATD/Extension/datasets/results_for_poster/first_eof_obs.png", dpi=400)
#         plt.show()


fig, axes = plt.subplots(
    nrows=1, ncols=2,
    figsize=(50, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)
plt.subplots_adjust(wspace=0.1, hspace=0)

ax1 = axes[0]
plot = ax1.pcolormesh(
    EOFs.LONGITUDE.values, EOFs.LATITUDE.values, EOFs, norm=norm,
    cmap='RdBu_r',
    # levels=200,
)
ax1.coastlines()

ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)
# ax1.set_title(f'Reynolds SSTA, Time = {time_to_see}')

ax2 = axes[1]
plot = ax2.pcolormesh(
    EOFs_obs.LONGITUDE.values, EOFs_obs.LATITUDE.values, EOFs_obs, norm=norm,
    cmap='RdBu_r',
    # levels=200,
)
ax2.coastlines()

ax2.set_xlim(-180, 180)
ax2.set_ylim(-90, 90)
# ax2.set_title(f'Simulated SSTA, Time = {time_to_see}')

fig.colorbar(plot, ax=axes.ravel().tolist(), label="EOF spatial pattern (standardised)")

plt.show()
