"""
SST Project - code reconstruction at Christmas 2025

Chengyun, Chris, Jason, Julia
2025-12-06
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def amiya():
    """
    Open Amiya image.

    Returns
    -------
    Amiya_img : array
        Image array of Amiya.
    """

    _amiya_img = mpimg.imread(r"Amiya.png")
    return _amiya_img


def main():
    """
    Main function to display Amiya image.

    # Japanese starts
    これはアーミヤ。
    彼女はいつも私を励ましてくれます。
    この画像は私の21歳の誕生日のプレゼントです。
    だから、私は彼女の画像を表示したいです。

    "おかえり、ドクター！"
    # Japanese ends
    """

    fig, ax = plt.subplots()
    ax.axis('off')
    amiya_img = amiya()
    plt.imshow(amiya_img)
    plt.show()
    print(
        r"おかえり、ドクター！"
    )


if __name__ == "__main__":
    main()
