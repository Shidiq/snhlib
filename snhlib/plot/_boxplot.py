import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from snhlib.plot._style import custom_style


def boxplot(data: pd.DataFrame, id_vars: tuple, value_vars: list, hue_order=None, **options):
    """boxplot _summary_

    Parameters
    ----------
    data : pandas dataframe
        input dataframe X and y
    id_vars : str
        label column name
    value_vars : list
        lisf of column name
    hue_order : list, optional
        label order, by default None

    Returns
    -------
    matplotlib.figure.Figure
    """

    xlabel = options.get("xlabel", None)
    ylabel = options.get("ylabel", None)
    showfliers = options.get("showfliers", False)
    palette = options.get("palette", None)
    loc = options.get("loc", "best")
    rotation = options.get("rotation", 0)
    legend = options.get("legend", True)
    legend_texts = options.get("legend_texts", None)
    per1000 = options.get("per1000", False)
    figsize = options.get("figsize", (10.72, 8.205))

    data_melt = pd.melt(data, id_vars=id_vars, value_vars=value_vars)

    fig, ax = custom_style(figsize=figsize)
    ax = sns.boxplot(
        data=data_melt,
        x="variable",
        y="value",
        hue=id_vars,
        hue_order=hue_order,
        showfliers=showfliers,
        palette=palette,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        if isinstance(legend_texts, list):
            L = ax.legend(loc=loc)
            for i, item in enumerate(legend_texts):
                L.get_texts()[i].set_text(item)
        else:
            ax.legend(loc=loc)

        ax.legend_.set_title(None)

    plt.xticks(rotation=rotation)

    if per1000:
        ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / 1000.0))
        ax.yaxis.set_major_formatter(ticks_y)

    return fig
