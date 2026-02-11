import matplotlib.pyplot as plt
from utilities import load_and_prepare_dataset
# from utilities import get_monthly_mean

hbar = load_and_prepare_dataset(
    "datasets/Mixed_Layer_Depth-Seasonal_Mean.nc"
)['MONTHLY_MEAN_MLD']

new_hbar = load_and_prepare_dataset(
    "datasets/.test-Mixed_Layer_Depth-Seasonal_Mean.nc"
)['MONTHLY_MEAN_MLD']

hbar.sel(MONTH=1).plot(cmap='Blues')
plt.show()
new_hbar.sel(MONTH=1).plot(cmap='Blues')
plt.show()

difference = hbar - new_hbar
difference.sel(MONTH=1).plot(cmap='RdBu_r')
print(difference.min().item())
plt.show()
