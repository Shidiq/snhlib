
import matplotlib
import matplotlib.pyplot as plt


class Style:
    @staticmethod
    def reset():
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    @staticmethod
    def paper(loc='best', classic=True, figsize=[10.72, 8.205]):
        if classic:
            plt.style.use('classic')
        params = {
            "axes.formatter.useoffset": False,
            "font.family": "sans-serif",
            "font.sans-serif": "Arial",
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "axes.labelsize": 28,
            "axes.labelweight": "bold",
            'axes.titlesize': 28,
            'axes.titleweight': "bold",
            "figure.dpi": 300,
            "figure.figsize": figsize,
            "legend.loc": loc,
            "legend.fontsize": 24,
            "legend.fancybox": True,
            "mathtext.fontset": 'custom',
            "mathtext.default": 'regular',
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
        ax = fig.add_subplot(1, 1, 1)
        fig.patch.set_facecolor('xkcd:white')
        return fig, ax
