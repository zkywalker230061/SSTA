"""
test file.

Chengyun Zhu
2025-10-11
"""

from IPython.display import display

from rgargo_read import load_and_prepare_dataset
# from rgargo_plot import visualise_dataset

ssta = load_and_prepare_dataset(
    "../datasets/Simulated_SSTA-(2004-2018).nc",
)["__xarray_dataarray_variable__"]
display(ssta)
print(ssta.max().item(), ssta.min().item())
print(abs(ssta).mean().item())
