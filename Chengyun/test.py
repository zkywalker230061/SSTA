"""
test file.

Chengyun Zhu
2025-11-06
"""

from IPython.display import display

from rgargo_read import load_and_prepare_dataset
# from rgargo_plot import visualise_dataset

SECONDS = 30.4375 * 24 * 60 * 60  # average seconds in a month


ent_velocity_ds = load_and_prepare_dataset(
    "../datasets/Entrainment_Velocity-(2004-2018).nc",
)
display(ent_velocity_ds)
ent_velocity = ent_velocity_ds['ENTRAINMENT_VELOCITY']
print(ent_velocity.max().item()*SECONDS, ent_velocity.min().item()*SECONDS)
