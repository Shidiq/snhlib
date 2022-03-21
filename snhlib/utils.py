import re
import sys

import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def COLORs(style=1):
    if style == 1:
        co = ["r", "b", "g", "c", "m", "y", "k", "C0", "C1", "C2"]
    return co


def MARKERs():
    return ["o", "v", "s", "p", "P", "*", "h", "H", "X", "D"]


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"
    FULL = "%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go"

    def __init__(self, total, width=40, fmt=DEFAULT, symbol="=", output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d",
                          r"\g<name>%dd" % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "remaining": remaining,
        }
        print("\r" + self.fmt % args, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
