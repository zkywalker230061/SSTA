"""
test file.

Chengyun Zhu
2025-10-11
"""


from rgargo_read import load_and_prepare_dataset
from rgargo_plot import visualise_dataset


def main():
    t_m = load_and_prepare_dataset(
        "../dataset/Mean Temperature Dataset (2004-2018).nc"
    )
    visualise_dataset(t_m)


if __name__ == "__main__":
    main()
