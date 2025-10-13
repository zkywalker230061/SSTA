"""
ARGO. Archive.

Chengyun Zhu
2025-09-30
"""

# from IPython.display import display

# Import the main data fetcher:
# from argopy import DataFetcher


# Define what you want to fetch...
# a region:
# ArgoSet = DataFetcher().region([110, 130, 20., 30., 0, 10.])
# floats:
# ArgoSet = DataFetcher().float([6902746, 6902747, 6902757, 6902766])
# or specific profiles:
# ArgoSet = DataFetcher().profile(6902746, 34)

# then fetch and get data as xarray datasets:
# ds = ArgoSet.load().data
# or
# ds = ArgoSet.to_xarray()

# you can even plot some information:
# ArgoSet.plot('trajectory')
# display(ds)

import argopy
argopy.dashboard()
