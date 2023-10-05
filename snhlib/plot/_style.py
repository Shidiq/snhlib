import matplotlib
import matplotlib.pyplot as plt


def reset_style():
    """reset_style
    Reset matplotlib style
    """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def custom_style(loc="best", classic=True, figsize=(10.72, 8.205), single_ax=True):
    """custom_style
    paper based style like OriginLab PRO Figure

    Parameters
    ----------
    loc : str, optional
        legend location, by default "best"
    classic : bool, optional
        plt sytle, by default True
    figsize : tuple, optional
        figure size, by default (10.72, 8.205)
    single_ax : bool, optional
        return single axis, by default True

    Returns
    -------
    fig
        matplotlib figure, and axes if single_ax.
    """
    if classic:
        plt.style.use("classic")

    params = {
        "axes.formatter.useoffset": False,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "axes.labelsize": 28,
        "axes.labelweight": "bold",
        "axes.titlesize": 28,
        "axes.titleweight": "bold",
        "figure.dpi": 300,
        "figure.figsize": figsize,
        "legend.loc": loc,
        "legend.fontsize": 24,
        "legend.fancybox": True,
        "mathtext.fontset": "custom",
        "mathtext.default": "regular",
        "figure.autolayout": True,
        "patch.edgecolor": "#000000",
        "text.color": "#000000",
        "axes.edgecolor": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }

    matplotlib.rcParams.update(params)
    fig = plt.figure()
    fig.patch.set_facecolor("xkcd:white")

    if single_ax:
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax
    else:
        return fig
