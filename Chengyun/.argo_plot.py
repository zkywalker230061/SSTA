"""
ARGO Plot file for literature review. Archive.

Chengyun Zhu
2025-10-03
"""


import matplotlib.pyplot as plt
import numpy as np
from argopy import DataFetcher


# Define what you want to fetch...
# a region:
# ArgoSet = DataFetcher().region([110, 130, 20., 30., 0, 10.])
# floats:
# ArgoSet = DataFetcher().float([6902746, 6902747, 6902757, 6902766])
# or specific profiles:
ArgoSet_tropical = DataFetcher().profile(5906602, 26)
ArgoSet_temperate = DataFetcher().profile(7900887, 8)
ArgoSet_frigid = DataFetcher().profile(3902686, 50)

# then fetch and get data as xarray datasets:
# ds = ArgoSet.load().data
# or
tropical = ArgoSet_tropical.to_xarray()
temperate = ArgoSet_temperate.to_xarray()
frigid = ArgoSet_frigid.to_xarray()

# you can even plot some information:
# ArgoSet.plot('trajectory')
# display(ds)

# Extract pressure and temperature
pres = tropical['PRES'].values.flatten()    # Pressure (dbar)
temp = tropical['TEMP'].values.flatten()    # Temperature (°C)

# Mask NaNs
mask = ~np.isnan(temp) & ~np.isnan(pres)

plt.figure(figsize=(5, 6))
plt.plot(temp[mask], pres[mask], '#66ccff')
plt.xlim(0, 30)
plt.ylim(2000, 0)
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Pressure (dbar)", fontsize=12)
plt.title("ARGO Float 5906602, cycle 26 - 30/09/2025")
plt.grid()
plt.show()

# Extract pressure and temperature
pres = temperate['PRES'].values.flatten()    # Pressure (dbar)
temp = temperate['TEMP'].values.flatten()    # Temperature (°C)

# Mask NaNs
mask = ~np.isnan(temp) & ~np.isnan(pres)

plt.figure(figsize=(5, 6))
plt.plot(temp[mask], pres[mask], '#66ccff')
plt.xlim(0, 30)
plt.ylim(2000, 0)
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Pressure (dbar)", fontsize=12)
plt.title("ARGO Float 7900887, cycle 8 - 29/09/2025")
plt.grid()
plt.show()


# Extract pressure and temperature
pres = frigid['PRES'].values.flatten()    # Pressure (dbar)
temp = frigid['TEMP'].values.flatten()    # Temperature (°C)

# Mask NaNs
mask = ~np.isnan(temp) & ~np.isnan(pres)

plt.figure(figsize=(5, 6))
plt.plot(temp[mask], pres[mask], '#66ccff')
plt.xlim(-2, 5)
plt.ylim(2000, 0)
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Pressure (dbar)", fontsize=12)
plt.title("ARGO Float 3902686, cycle 50 - 29/09/2025")
plt.grid()
plt.show()
