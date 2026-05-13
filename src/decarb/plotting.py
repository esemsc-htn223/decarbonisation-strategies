import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Iterable

from .terms import Term, TermCollection, TERMS_ALL
from .companies import Company, COS

def plot_mentions(
    df: pd.DataFrame, 
    *, 
    ax: plt.Axes | None = None, figsize: tuple[float]|tuple[int]|None = None,
    plot_n: str|None = 'words', normed_by: str|None = 'words',
    terms: TermCollection = TERMS_ALL
):

    if ax is None: _, ax = plt.subplots(figsize=figsize)

    for term in terms:
        ax.plot(df.index, df[term.alias], marker = term.marker, linestyle = term.dash_ls if term.dash else None, color = term.colour, gapcolor=term.edgecolour)

    if normed_by is not None:
        ax.set_ylabel(f'Term count/No. {normed_by}')
    else:
        ax.set_ylabel('Term count')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index, rotation = 45)

    if not plot_n.startswith('n_'): plot_n = 'n_' + plot_n
    if plot_n in df:
        ax_n = ax.twinx()
        df[plot_n].plot(linestyle = 'None', marker = '2', color = 'k', alpha = 0.9, ax = ax_n)
        ax_n.set_ylabel(f'No. {plot_n.replace("n_", "")}')
        return ax, ax_n
    return ax

def plot_rankings(
        df: pd.DataFrame,
        *,
        ax: plt.Axes | None = None,
        figsize: tuple[float]|tuple[int]|None = None,
        terms: TermCollection = TERMS_ALL,
        lines: bool = True,
        zero_fill: str = 'nan'  # 'nan' or 'min' (fills with min rank + 1). ignored otherwise
):
    if ax is None: _, ax = plt.subplots(figsize=figsize)

    df_ranked = df.loc[:, [term.alias for term in terms]]
    if zero_fill.lower() == 'nan':
        df_ranked = df_ranked.replace(0, np.nan)
    elif zero_fill.lower() == 'min':
        min_rank = df_ranked[df_ranked > 0].min().min()
        df_ranked = df_ranked.replace(0, min_rank + 1)

    df_ranked = df_ranked.rank(ascending=False, method='min', axis=1)

    for term in terms:
        if lines:
            ax.plot(df_ranked.index, df_ranked[term.alias], 
                       marker = None, linestyle = term.dash_ls if term.dash else None, 
                       color = term.colour, gapcolor=term.edgecolour, zorder = 1)
        ax.scatter(df_ranked.index, df_ranked[term.alias], 
                        marker = term.marker, linestyle = 'None', 
                        color = term.colour, edgecolor=term.edgecolour, hatch=term.hatch)

    ax.invert_yaxis()
    ax.set_ylabel('Rank')
    ax.set_xticks(df_ranked.index)
    ax.set_xticklabels(df_ranked.index, rotation = 45)

    return ax

def plot_comparison(
    dfs: dict[str, pd.DataFrame], 
    terms: TermCollection = TERMS_ALL, *,
    cos: Iterable[Company] = COS, 
    rank: bool = False,
    rank_zero_fill: str = 'nan', rank_lines: bool = True,
    plot_n: str|None = 'words', normed_by: str|None = 'words',
    suptitle: str|None = None
):

    years_an = sorted({int(y) for co in cos for y in dfs[co.foldername + '-an'].index})
    years_sr = sorted({int(y) for co in cos for y in dfs[co.foldername + '-sr'].index})

    xmin_an, xmax_an = years_an[0], years_an[-1]
    xmin_sr, xmax_sr = years_sr[0], years_sr[-1]


    fig, axs = plt.subplots(len(cos), 2, figsize = (12, 3*len(cos) + 1))
    if len(cos) == 1:
        axs = np.array([axs])
    for i, co in enumerate(cos):
        axs1 = axs[i][0]
        axs2 = axs[i][1]
        if rank:
            plot_rankings(dfs[co.foldername + '-an'], ax = axs1, terms = terms, lines=rank_lines, zero_fill=rank_zero_fill)
            plot_rankings(dfs[co.foldername + '-sr'], ax = axs2, terms = terms, lines=rank_lines, zero_fill=rank_zero_fill)
        else:
            plot_mentions(dfs[co.foldername + '-an'], ax = axs1, terms = terms, plot_n=plot_n, normed_by=normed_by)
            plot_mentions(dfs[co.foldername + '-sr'], ax = axs2, terms = terms, plot_n=plot_n, normed_by=normed_by)

        axs1.set_xlim(xmin_an, xmax_an)
        axs2.set_xlim(xmin_sr, xmax_sr)

        if xmin_an < dfs[co.foldername + '-an'].index.min(): axs1.axvspan(xmin_an, dfs[co.foldername + '-an'].index.min(), color='lightgrey', alpha=0.5, zorder=-1)
        if xmax_an > dfs[co.foldername + '-an'].index.max(): axs1.axvspan(dfs[co.foldername + '-an'].index.max(), xmax_an, color='lightgrey', alpha=0.5, zorder=-1)
        if xmin_sr < dfs[co.foldername + '-sr'].index.min(): axs2.axvspan(xmin_sr, dfs[co.foldername + '-sr'].index.min(), color='lightgrey', alpha=0.5, zorder=-1)
        if xmax_sr > dfs[co.foldername + '-sr'].index.max(): axs2.axvspan(dfs[co.foldername + '-sr'].index.max(), xmax_sr, color='lightgrey', alpha=0.5, zorder=-1)

        if i == 0:
            axs1.set_title('Annual Reports')
            axs2.set_title('Sustainability Reports')

        if i < len(cos) - 1:
            axs1.set_xticks([])
            axs2.set_xticks([])
            axs1.set_xticklabels([])
            axs2.set_xticklabels([])
        else:
            axs1.set_xticks(np.arange(xmin_an, xmax_an + 1, 2))
            axs2.set_xticks(np.arange(xmin_sr, xmax_sr + 1, 2))
            axs1.set_xticklabels(np.arange(xmin_an, xmax_an + 1, 2))
            axs2.set_xticklabels(np.arange(xmin_sr, xmax_sr + 1, 2))
            axs1.set_xlabel('Year')
            axs2.set_xlabel('Year')


    # construct legend manually
    if rank:
        handles = [plt.Line2D([0], [0], color = term.colour, marker = term.marker if term.marker else 'o', linestyle = term.dash_ls if term.dash else None, label = term.alias, gapcolor=term.edgecolour) for term in terms]
        plt
    else:
        handles = [plt.Line2D([0], [0], color = term.colour, marker = term.marker, linestyle = term.dash_ls if term.dash else None, label = term.alias, gapcolor=term.edgecolour) for term in terms]
    fig.legend(
        handles = handles,
        loc = 'lower center', ncol = (len(terms) + 1)//2, frameon = False, 
        bbox_to_anchor = (0.5, -0.03),
        alignment = 'center'
    )
    extra_labels = [mpl.patches.Patch(color='lightgrey', alpha=0.5, label='No data')]
    if plot_n and 'n_' + plot_n in dfs[cos[0].foldername + '-an'] and not rank:
        extra_labels.append(plt.Line2D([0], [0], color = 'k', marker = '2', linestyle = 'None', label = f'Total {plot_n}'))
    fig.legend(
        handles = extra_labels,
        loc = 'lower center', ncol = len(extra_labels), frameon = False, 
        bbox_to_anchor = (0.5, -0.045),
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize = 18, fontweight='light', y = 1.01)

    fig.tight_layout(pad = 0.75)

    for i, co in enumerate(cos):
        axs1 = axs[i][0]

        col_bot = axs1.get_position().y0
        col_top = axs1.get_position().y1
        if rank:
            offset = 0.055
        else:
            offset = 0.09 if normed_by else 0.065
        x = axs1.get_position().x0 - offset
        y = axs1.get_position().y0 + (col_top - col_bot) / 2
        fig.text(x, y, co.alias, ha='center', va='center', fontweight='light', fontsize = 16, rotation = 90,
                 color = 'dimgrey')
        

    return fig, axs
        