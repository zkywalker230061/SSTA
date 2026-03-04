"""
SSTA and SSSA model.

Chengyun Zhu
2026-03-03
"""

import prepare_rgargo
import prepare_era5
import prepare_reynolds

import calculate_surf
import calculate_ent
import calculate_ekm
import calculate_geo


def main():
    """
    Main function to run the SSTA and SSSA model.
    """

    with open("logs/datasets.txt", "x", encoding="utf-8"):
        return

    prepare_rgargo.main()
    prepare_era5.main()  # needs xesmf
    prepare_reynolds.main()  # needs xesmf

    calculate_surf.main()
    calculate_ent.main()
    calculate_ekm.main()
    calculate_geo.main()


if __name__ == "__main__":
    main()
