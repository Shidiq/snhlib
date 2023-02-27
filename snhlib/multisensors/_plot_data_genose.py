import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

from snhlib.plot._style import custom_style


def plot_data(
    data,
    id_vars="time(s)",
    value_vars=[f"S{i+1}" for i in range(10)],
    palette=None,
    vline=[],
    fill=True,
    xlim=None,
    ylim=None,
    xlabel="Time (s)",
    ylabel="Value",
    legend=True,
    linewidht=4,
    loc=1,
    per1000=True,
    figsize=(10.72, 8.205),
):
    """plot_data
    Plot data multisensors. ex.: GeNose data

    Parameters
    ----------
    data : pandas dataframe
        Pandas dataframe with multi-variables
    id_vars: str
        x-axes data label, by default time(s)
    value_vars: list
        y-axes data label, by default [S1, S2, ..., S10]
    vline : list, optional
        vertical line, by default None
    fill : bool, optional
        fill between vertical line, by default vline
    xlim : list, optional
        x-axes limit, by default None
    xlabel : str, optional
        _description_, by default "Time (s)"
    ylabel : str, optional
        _description_, by default "Value"
    legend : bool, optional
        _description_, by default True
    linewidht : int, optional
        _description_, by default 4
    loc : int, optional
        _description_, by default 1
    per1000 : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    data_melt = pd.melt(data, id_vars=id_vars, value_vars=value_vars)

    fig, ax = custom_style(figsize=figsize)

    if palette is not None:
        palette = sns.color_palette("Paired", 10)

    sns.lineplot(
        data=data_melt,
        x=id_vars,
        y="value",
        hue="variable",
        ax=ax,
        linewidth=linewidht,
        palette=palette,
    )

    if vline is not []:
        for lim in vline:
            ax.axvline(x=lim, color="k", linestyle="--")

    if fill and len(vline) >= 2:
        axes = plt.gca()
        [ylim_min, ylim_max] = axes.get_ylim()

        ax.fill_between(vline, y1=ylim_max, y2=ylim_min, color="#f2f2f2", alpha=0.7)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=28, fontweight="bold")

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=28, fontweight="bold")

    if per1000:
        ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:.2f}".format(x / 1000.0))
        ax.yaxis.set_major_formatter(ticks_y)

    if legend:
        if loc == 5:
            leg = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            leg = ax.legend(loc=loc)

        for legobj in leg.legendHandles:
            legobj.set_linewidth(linewidht)

    return fig
