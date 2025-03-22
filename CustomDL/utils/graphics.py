import math
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from torch import Tensor
from typing import Optional


def draw_filters(
    image: Tensor,
    convolved_filters: Tensor,
    cmap: str = 'gray_r',
    *,
    fig: Optional[Figure] = None,
) -> Figure:
    """
    Draw the original image along with its filtered version. Each filter can be
        interpreted as a channel of the image.

    :param torch.Tensor image: The original image to draw. This should have shape
        (C, H, W) where C is the number of channels, H and W is the height and width
        of the image respectively.
    :param torch.Tensor convolved_filters: A sequence of filtered images as Tensor. This
        should have shape (F, H, W) where F is the number of filters, H and W is the
        height and width of the image respectively.
    :param str cmap: The colormap used to draw the images. This is passed into the `imshow()`
        function. If the original image has colour channels, then `cmap` is not used for to
        draw the image, but is still used to draw the filters.
    :param matplotlib.figure.Figure | None fig: The figure to draw on. This is by default
        `None`.
    :return: The figure drawn on.
    :rtype: matplotlib.figure.Figure
    """
    rows = round(math.sqrt(convolved_filters.shape[0]))
    cols = int(math.ceil(convolved_filters.shape[0] / rows))
    layout = []
    ax_rows = ['Image']
    for i in range(convolved_filters.shape[0]):
        ax_rows.append(i)
        if len(ax_rows) - 1 == cols:
            layout.append(ax_rows)
            ax_rows = ['Image']
    if 1 < len(ax_rows) < cols + 1:
        missing = cols - len(ax_rows) + 1
        ax_rows.extend(('.' for _ in range(missing)))
        layout.append(ax_rows)


    if fig is None:
        fig = plt.figure()
    width_ratios = [cols]
    width_ratios.extend(1 for _ in range(cols))
    axes = fig.subplot_mosaic(layout, width_ratios=width_ratios)

    axes['Image'].imshow(image.permute((1, 2, 0)), cmap=(None if image.shape[0] > 1 else cmap))
    axes['Image'].set_title('Original Image')
    axes['Image'].set_xticks([])
    axes['Image'].set_yticks([])
    for i in range(convolved_filters.shape[0]):
        axes[i].set_title(f'Filter {i}')
        axes[i].imshow(convolved_filters[i], cmap=cmap)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    return fig