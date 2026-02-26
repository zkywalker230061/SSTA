import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from Chris.utils import make_movie, get_eof_with_nan_consideration, remove_empty_attributes, get_save_name, \
    coriolis_parameter, get_month_from_time
from utils import get_monthly_mean, get_anomaly, load_and_prepare_dataset
from matplotlib.animation import FuncAnimation
import matplotlib

NAO_DATA_PATH = "/Volumes/G-DRIVE ArmorATD/Extension/datasets/nao.txt"

def read_nao(file):
    nao_list = []
    with open(file, "r") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
            else:
                line = line.strip()
                split_line = line.split()
                year = int(split_line[0])
                if year >= 2004:
                    nao_indices = split_line[1:13]
                    for nao_index in nao_indices:
                        nao_list.append(float(nao_index))
                        if len(nao_list) >= 180:
                            return nao_list
    return nao_list

nao_list = read_nao(NAO_DATA_PATH)

