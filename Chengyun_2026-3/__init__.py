"""
SSTA and SSSA model.

Chengyun Zhu
2026-03-03
"""

import prepare_rgargo
import prepare_era5
import prepare_reynolds


def main():
    """
    Main function to run the SSTA and SSSA model.
    """

    prepare_rgargo.main()
    prepare_era5.main()
    prepare_reynolds.main()


if __name__ == "__main__":
    main()
