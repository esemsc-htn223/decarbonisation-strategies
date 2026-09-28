import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import Literal
import altair as alt
import tiktoken

from decarb.terms import Term, TermCollection, TERMS_ALL
from decarb.companies import COMPANY_ALIASES, COMPANIES_ALL, CompanyCollection
from decarb.util import REPORT_TYPES_LONG

__all__ = [
    'regex_mentions', 'regex_rankings', 'regex_comparison',
    'term_search_results', 'chunk_size_hist', 'llm_timeseries',
    'llm_term_heatmap', 'llm_terms_heatmap'
]


def regex_mentions(
    df: pd.DataFrame, 
    *,
    company: str|None = None, report_type: str|None = None,
    ax: plt.Axes | None = None, figsize: tuple[float]|tuple[int]|None = None,
    plot_n: Literal['words', 'chunks', 'pages']|None = 'words', 
    normed_by: Literal['words', 'chunks', 'pages']|None = 'words',
    terms: TermCollection = TERMS_ALL
):
    '''
    Plot the number of mentions of each term in `terms` over time, using the data in `df` output from `regex_pipeline`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot. Must have a datetime index and columns for each term 
        in `terms`, as well as a column for the total number of words (or other unit).
    ax : plt.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    figsize : tuple of float, optional
        Size of the figure to create if `ax` is None. Ignored if `ax` is not None.
    plot_n : Literal['words', 'chunks', 'pages']|None, optional
        Column name in `df` to plot as a secondary y-axis. If None, no secondary y-axis will be plotted. Default is 'words'.
    normed_by : Literal['words', 'chunks', 'pages']|None, optional
        Column name in `df` used to normalise the counts of regex matches, used to label the y axis. Default is 'words'.
    terms : TermCollection, optional
        Collection of terms to plot. Default is `TERMS_ALL`.

    Returns
    -------
    ax : plt.Axes
        The axes on which the plot was created.
    ax_n : plt.Axes, optional
        The secondary axes for the `plot_n` column, if `plot_n` is not None. Otherwise, returns None.
    '''

    df_wkg = df.copy()

    if report_type is not None:
        df_wkg = df_wkg[df_wkg.index.get_level_values('report_type') == report_type]
    elif df_wkg.index.get_level_values('report_type').nunique() > 1:
        raise ValueError('report_type must be specified if df contains multiple report types.')

    if company is not None:
        df_wkg = df_wkg[df_wkg.index.get_level_values('company') == company]
    elif df_wkg.index.get_level_values('company').nunique() > 1:
        raise ValueError('company must be specified if df contains multiple companies.')

    if ax is None: _, ax = plt.subplots(figsize=figsize)

    for term in terms:
        ax.plot(df_wkg.index.get_level_values('year').astype(int), df_wkg[term.alias], marker = term.marker, linestyle = term.dash_ls if term.dash else None, color = term.colour, gapcolor=term.edgecolour)

    if normed_by is not None:
        ax.set_ylabel(f'Term count/No. {normed_by}')
    else:
        ax.set_ylabel('Term count')
    ax.set_xticks(df_wkg.index.get_level_values('year').astype(int))
    ax.set_xticklabels(df_wkg.index.get_level_values('year').astype(int), rotation = 45)

    if not plot_n.startswith('n_'): plot_n = 'n_' + plot_n
    if plot_n in df_wkg.columns:
        ax_n = ax.twinx()
        df_wkg[plot_n].plot(linestyle = 'None', marker = '2', color = 'k', alpha = 0.9, ax = ax_n)
        ax_n.set_ylabel(f'No. {plot_n.replace("n_", "")}')
        return ax, ax_n
    return ax

def regex_rankings(
        df: pd.DataFrame,
        *,
        company: str|None = None, report_type: str|None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float]|tuple[int]|None = None,
        terms: TermCollection = TERMS_ALL,
        lines: bool = True,
        zero_fill: Literal['nan', 'min'] = 'nan'  # 'nan' or 'min' (fills with min rank + 1). ignored otherwise
):
    '''
    Plot the rankings of each term in `terms` over time, using the data in `df` output from `regex_pipeline`.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot. Must have a datetime index and columns for each term
        in `terms`, as well as a column for the total number of words (or other unit).
    company : str, optional
        The company to plot. If None, df should only contain one company. 
        If df contains multiple companies, a ValueError will be raised.
    report_type : str, optional
        The report type to plot. If None, df should only contain one report type. 
        If df contains multiple report types, a ValueError will be raised.
    ax : plt.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    figsize : tuple of float, optional
        Size of the figure to create if `ax` is None. Ignored if `ax` is not None.
    terms : TermCollection, optional
        Collection of terms to plot. Default is `TERMS_ALL`.
    lines : bool, optional
        Whether to plot lines connecting the points for each term. Default is True.
    zero_fill : str, optional
        How to handle zero values in the rankings. If 'nan', zero values will be replaced with NaN. 
        If 'min', zero values will be replaced with the minimum rank + 1
        among the non-zero values. Ignored if `lines` is False. Default is 'nan'.
    
    Returns
    -------
    ax : plt.Axes
        The axes on which the plot was created.

    Raises
    ------
    ValueError
        If `company` is None and df contains multiple companies, or if `report_type` is None and df contains multiple report types.
    '''
    if ax is None: _, ax = plt.subplots(figsize=figsize)

    indexer = pd.Series([True] * len(df), index=df.index)
    if company is not None:
        indexer = df.index.get_level_values('company') == company
    elif df.index.get_level_values('company').nunique() > 1:
        raise ValueError('company must be specified if df contains multiple companies.')
    if report_type is not None:
        indexer = indexer & (df.index.get_level_values('report_type') == report_type)
    elif df.index.get_level_values('report_type').nunique() > 1:
        raise ValueError('report_type must be specified if df contains multiple report types.')

    df_ranked = df.loc[indexer, [term.alias for term in terms]]
    if zero_fill.lower() == 'nan':
        df_ranked = df_ranked.replace(0, np.nan)
    elif zero_fill.lower() == 'min':
        min_rank = df_ranked[df_ranked > 0].min().min()
        df_ranked = df_ranked.replace(0, min_rank + 1)

    df_ranked = df_ranked.rank(ascending=False, method='min', axis=1)

    for term in terms:
        if lines:
            ax.plot(df_ranked.index.get_level_values('year').astype(int), df_ranked[term.alias], 
                       marker = None, linestyle = term.dash_ls if term.dash else None, 
                       color = term.colour, gapcolor=term.edgecolour, zorder = 1)
        ax.scatter(df_ranked.index.get_level_values('year').astype(int), df_ranked[term.alias], 
                        marker = term.marker, linestyle = 'None', 
                        color = term.colour, edgecolor=term.edgecolour, hatch=term.hatch)

    ax.invert_yaxis()
    ax.set_ylabel('Rank')
    ax.set_xticks(df_ranked.index.get_level_values('year').astype(int))
    ax.set_xticklabels(df_ranked.index.get_level_values('year').astype(int), rotation = 45)

    return ax

def regex_comparison(
    df: pd.DataFrame, 
    terms: TermCollection = TERMS_ALL, *,
    cos: CompanyCollection = COMPANIES_ALL, 
    rank: bool = False,
    rank_zero_fill: Literal['nan', 'min'] = 'nan', rank_lines: bool = True,
    plot_n: Literal['words', 'chunks', 'pages']|None = 'words', 
    normed_by: Literal['words', 'chunks', 'pages']|None = 'words',
    suptitle: str|None = None
):
    '''
    Create plots for each company in `cos` showing either the number of mentions of each term in `terms` 
    or the rankings of each term over time, using the data in `df` output from `regex_pipeline`.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    terms : TermCollection, optional
        Collection of terms to plot. Default is `TERMS_ALL`.
    cos : CompanyCollection, optional
        Collection of companies to plot. Default is `COMPANIES_ALL`.
    rank : bool, optional
        Whether to plot rankings instead of mentions. Default is False.
    rank_zero_fill : str, optional
        How to handle zero values in the rankings. If 'nan', zero values will be replaced with NaN. 
        If 'min', zero values will be replaced with the minimum rank + 1 among the non-zero values. Ignored if `rank` is False. Default is 'nan'.
    rank_lines : bool, optional
        Whether to plot lines connecting the points for each term when plotting rankings. Ignored if `rank` is False. Default is True.
    plot_n : Literal['words', 'chunks', 'pages']|None, optional
        Column name in `df` to plot as a secondary y-axis when plotting mentions. If None, no secondary y-axis will be plotted. Default is 'words'.
    normed_by : Literal['words', 'chunks', 'pages']|None, optional
        What the data in `df` has been normalised by. Used to label the y-axis. Default is 'words'.
    suptitle : str, optional
        Suptitle for the entire figure. Default is None.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plots.
    axs : np.ndarray
        Array of axes containing the individual plots for each company.
    '''

    df = df.sort_index(level=['company', 'report_type', 'year']).copy()    
    years_an = sorted({int(y) for co in cos for y in df.loc[df.index.get_level_values('report_type') == 'annual-report'].index.get_level_values('year')})
    years_sr = sorted({int(y) for co in cos for y in df.loc[df.index.get_level_values('report_type') == 'sustainability-report'].index.get_level_values('year')})

    xmin_an, xmax_an = years_an[0], years_an[-1]
    xmin_sr, xmax_sr = years_sr[0], years_sr[-1]


    fig, axs = plt.subplots(len(cos), 2, figsize = (12, 3*len(cos) + 1))
    if len(cos) == 1:
        axs = np.array([axs])
    for i, co in enumerate(cos):
        axs1 = axs[i][0]
        axs2 = axs[i][1]
        if rank:
            regex_rankings(
                df, company=co.folder_name, report_type='annual-report', 
                ax=axs1, terms=terms, lines=rank_lines, zero_fill=rank_zero_fill
            )
            regex_rankings(
                df, company=co.folder_name, report_type='sustainability-report', 
                ax = axs2, terms = terms, lines=rank_lines, zero_fill=rank_zero_fill
            )
        else:
            regex_mentions(
                df, company=co.folder_name, report_type='annual-report', 
                ax=axs1, terms=terms, plot_n=plot_n, normed_by=normed_by
            )
            regex_mentions(
                df, company=co.folder_name, report_type='sustainability-report', 
                ax=axs2, terms=terms, plot_n=plot_n, normed_by=normed_by
            )

        axs1.set_xlim(xmin_an, xmax_an)
        axs2.set_xlim(xmin_sr, xmax_sr)

        co_an_yrs = df.loc[(df.index.get_level_values('company') == co.folder_name) & (df.index.get_level_values('report_type') == 'annual-report')].index.get_level_values('year')
        co_sr_yrs = df.loc[(df.index.get_level_values('company') == co.folder_name) & (df.index.get_level_values('report_type') == 'sustainability-report')].index.get_level_values('year')
        if xmin_an < co_an_yrs.min(): axs1.axvspan(xmin_an, co_an_yrs.min(), color='lightgrey', alpha=0.5, zorder=-1)
        if xmax_an > co_an_yrs.max(): axs1.axvspan(co_an_yrs.max(), xmax_an, color='lightgrey', alpha=0.5, zorder=-1)
        if xmin_sr < co_sr_yrs.min(): axs2.axvspan(xmin_sr, co_sr_yrs.min(), color='lightgrey', alpha=0.5, zorder=-1)
        if xmax_sr > co_sr_yrs.max(): axs2.axvspan(co_sr_yrs.max(), xmax_sr, color='lightgrey', alpha=0.5, zorder=-1)

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
    if plot_n and 'n_' + plot_n in df and not rank:
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
        fig.text(x, y, co.full_name, ha='center', va='center', fontweight='light', fontsize = 16, rotation = 90,
                 color = 'dimgrey')
        

    return fig, axs



def term_search_results(search_res: dict, chunks: list, ax = None) -> tuple[plt.Figure, plt.Axes]:
    '''
    Plot the search results for each term in `search_res` over the chunks in `chunks`. Results
    are dispplayed as a heatmap, with green indicating that the term was found in the chunk 
    and white indicating that it was not found.

    Parameters
    ----------
    search_res : dict
        Dictionary containing the search results for each term, as output from `search_chunks`.
    chunks : list of str
        List of chunks to plot the search results for.
    ax : plt.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    fig : plt.Figure
        The figure containing the plot.
    ax : plt.Axes
        The axes on which the plot was created.
    '''
    _im = np.array(len(search_res) * [len(chunks) * [0]])
    for _i, _term in enumerate(search_res.keys()):
        _res = search_res[_term]
        _im[_i] = [1 if i in _res else 0 for i in range(len(chunks))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()
    ax.imshow(_im, aspect='auto', cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(search_res)))
    ax.set_yticklabels(list(search_res.keys()))
    ax.set_xlabel('Chunk Index')
    ax.set_title('Search Results for Terms in Chunks')
    ax.legend(
        handles = [mpl.patches.Patch(color='green', label='Term Found')],
        ncols = 2,
        loc = 'lower center',
        bbox_to_anchor = (0.5, -0.2),
        frameon = False
    )

    return fig, ax

def chunk_size_hist(chunks: list[str], encoding: tiktoken.Encoding|str = 'cl100k_base') -> tuple[plt.Figure, np.ndarray]:
    '''
    Create a histogram of chunk sizes in characters and tokens.

    Parameters
    ----------
    chunks : list[str]
        The list of chunk contents.
    encoding : tiktoken.Encoding | str, optional
        The encoding to use for tokenization. Defaults to 'cl100k_base'.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        A tuple containing the matplotlib figure and axes.
    '''
    if isinstance(encoding, str):
        encoding = tiktoken.get_encoding(encoding)
    chunk_lengths_tokens = [len(encoding.encode(c)) for c in chunks]
    chunk_lengths_chars = [len(c) for c in chunks]
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    axs[0].sharey(axs[1])
    axs[0].hist(chunk_lengths_chars, bins=100, color='green', alpha=0.7)
    axs[0].set_title('Chunk Lengths (Characters)')
    axs[0].set_xlabel('Length of Chunks (Characters)')
    axs[0].set_ylabel('Frequency')
    axs[1].hist(chunk_lengths_tokens, bins=100, color='blue', alpha=0.7)
    axs[1].set_title('Chunk Lengths (Tokens)')
    axs[1].set_xlabel('Length of Chunks (Tokens)')
    axs[1].set_ylabel('Frequency')
    return fig, axs


def llm_timeseries(df:pd.DataFrame, variable: Literal['q1_sentiment', 'q2_maturity'], 
                    *, 
                    company:str, term:Term, report_type:str, 
                    width_px:int = 1000, height_px:int = 250) -> alt.Chart:
    '''
    Create an Altair chart showing sentiment or maturity over time for a given term and report type.
    Missing data is indicated by shaded regions, with black for years where the report was not published, and gray for years where the term was not mentioned.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sentiment analysis results.
    company : str
        The company for which to plot sentiment over time.
    term : Term
        The term for which to plot sentiment over time.
    report_type : str
        The report type for which to plot sentiment over time.
    width_px : int, optional
        Width of the chart in pixels (default is 1000).
    height_px : int, optional
        Height of the chart in pixels (default is 250).

    Returns
    -------
    alt.Chart
    '''

    if variable not in ['q1_sentiment', 'q2_maturity']:
        raise ValueError(f"Variable '{variable}' should be either 'q1_sentiment' or 'q2_maturity'.")
    elif variable == 'q1_sentiment':
        domain = [-1, 1]
    elif variable == 'q2_maturity':
        domain = [1, 9]

    df_co_rt = df[(df["company"] == company) & (df["report_type"] == report_type)].copy()
    all_years = sorted(df['year'].unique())
    chart = alt.Chart(df_co_rt[(df_co_rt['term_alias'] == term.alias)])
    lines = chart.mark_point(color = term.colour).encode(
                x = alt.X('year:O')
                    .scale(domain = all_years)
                    .axis(title = None, labelAngle = -45, offset = -height_px//2, 
                          labelPadding = height_px//2 + 5, # +5 to account for missing ticks (default height of 5px)
                          tickSize = 0
                          )
                ,
                y = alt.Y(f'{variable}:Q', title = variable.split('_')[1].capitalize())
                    .scale(domain = domain)
                ,

                tooltip = [
                    alt.Tooltip('year:O', title = 'Report Year'),
                    alt.Tooltip('q1_sentiment:Q', title = 'Sentiment'),
                    alt.Tooltip('q2_maturity:N', title = 'Maturity'),
                    alt.Tooltip('q3_stage:N', title = 'Stage'),
                ]
            ).properties(
                #title = f'Sentiment over Time for {term.alias}',
                width = width_px,
                height = height_px
        )
    # mark missing data with shaded region
    # get years where report wasn't published
    has_pub_flag = "report_type_published" in df_co_rt.columns
    has_term_flag = "term_regex_found" in df_co_rt.columns

    no_reports_years: list[str] = []
    if has_pub_flag:
        no_reports_years = (
            df_co_rt.loc[df_co_rt["report_type_published"] == False, "year"]  # noqa: E712
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    else:
        no_reports_years = sorted(set(all_years) - set(df_co_rt['year'].unique()))


    missing_term_years: list[str] = []
    if has_term_flag:
        # Missing term = report was published but term regex not found.
        missing_term_years = (
            df_co_rt[
                (df_co_rt["term_alias"] == term.alias)
                & (df_co_rt["report_type_published"] == True)   # noqa: E712
                & (df_co_rt["term_regex_found"] == False)        # noqa: E712
            ]["year"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
    else:
        missing_term_years = sorted(set(all_years) - set(df_co_rt[(df_co_rt['term_alias'] == term.alias) & (df_co_rt['report_type_published'] == True)]['year'].unique()) - set(no_reports_years))

    if no_reports_years:
        no_reports_chart = (
            alt.Chart(pd.DataFrame({"year": no_reports_years}))
            .mark_rect(opacity=0.2, color="black")
            .encode(
                x=alt.X("year:O", title="Report Year"),
                y=alt.value(0),
                y2=alt.value(height_px),
                tooltip=alt.value("Report not published"),
            )
        )
        lines = no_reports_chart + lines

    if missing_term_years:
        missing_chart = (
            alt.Chart(pd.DataFrame({"year": missing_term_years}))
            .mark_rect(opacity=0.2, color="gray")
            .encode(
                x=alt.X("year:O", title="Report Year"),
                y=alt.value(0),
                y2=alt.value(height_px),
                tooltip=alt.value("Term not mentioned"),
            )
        )
        lines = missing_chart + lines

    return lines


def discrete_cmap(bounds:list|np.ndarray, cmap_name:str = 'RdYlGn') -> mpl.colors.ListedColormap:
    '''
    Create a discrete colormap for quantum sentiment/maturity values.

    Parameters
    ----------
    bounds : list | np.ndarray
        The boundaries for the colormap.
    cmap_name : str, optional
        The name of the matplotlib colormap to use. Default is 'RdYlGn'.

    Returns
    -------
    mpl.colors.ListedColormap
        The discrete colormap.
    '''
    from matplotlib import colors as colours
    cmap = mpl.colormaps[cmap_name]
    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)

    # extract all colours
    cmaplist = np.array([cmap(i) for i in range(cmap.N)])

    # create the new map
    cmap = colours.LinearSegmentedColormap.from_list(
        cmap_name + "_quantum", cmaplist, len(bounds))

    return cmap

def llm_term_heatmap(df: pd.DataFrame, term: Term, figsize: tuple = (12, 3)) -> plt.Figure:
    '''
    Plot a heatmap of sentiment and maturity for a given term across companies and report types.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the results of the analysis.
    term : Term
        The term to plot.
    figsize : tuple, optional
        The size of the figure in inches. Default is (12, 3).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the heatmap.
    '''
    if not isinstance(term, Term):
        try:
            term = TERMS_ALL[term]
        except KeyError:
            raise ValueError(f'Term "{term}" not found in TERMS_ALL. Available terms: {list(TERMS_ALL.keys())}')

    fig, axs = plt.subplots(df['report_type'].unique().shape[0], 2, figsize=figsize, sharey=True, sharex=True)

    cmaps = {
        'q1_sentiment': 'RdYlGn',
        'q2_maturity': discrete_cmap([1, 2, 3, 4, 5, 6, 7, 8, 9], cmap_name='cividis'),
        'q3_stage': discrete_cmap([1, 2, 3, 4, 5], cmap_name='plasma'),
    }

    for i, rt in enumerate(df['report_type'].unique()):
        for j, question in enumerate(('q1_sentiment', 'q2_maturity')):
            df_question = df.loc[(df['report_type'] == rt) & (df['term_alias'] == term.alias)].pivot(index='company', columns='year', values=question).astype(float)

            im = axs[i, j].imshow(
                df_question.replace(-1, np.nan).values,
                cmap=cmaps[question]
            )
            axs[i, j].set_title(f"{REPORT_TYPES_LONG.get(rt, rt).title()} - {question.split('_')[1].capitalize()}")
            axs[i, j].set_yticks(np.arange(len(df_question.index)), df_question.reset_index()['company'].apply(lambda x: COMPANY_ALIASES.get(x, x)))
            axs[i, j].set_xticks(np.arange(len(df_question.columns)),minor = True)
            axs[i, j].set_xticks(
                np.arange(len(df_question.columns), step = 5),
                np.arange(df_question.columns.astype('int').min(), df_question.columns.astype('int').max() + 1, 5), 
                minor = False, 
                rotation = 90, #ha = 'right'
            )

    fig.suptitle(f"{term.alias.upper()}", fontweight='bold', fontfamily = 'monospace', fontsize = 16)
    fig.set_constrained_layout(True)

    return fig, axs

def llm_terms_heatmap(
        df: pd.DataFrame, question: Literal['q1_sentiment', 'q2_maturity', 'q3_stage', 'sentiment', 'maturity', 'stage'], *, 
        terms: TermCollection = 'all', 
        figsize: tuple = (12, 15),
        cmap: str|None = None,
        return_type: Literal['figure', 'axs', 'both'] = 'both'
    ) -> plt.Figure:
    '''
    Plot a heatmap of sentiment and maturity for all terms across companies and report types.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the results of the analysis.
    question : Literal['q1_sentiment', 'q2_maturity', 'sentiment', 'maturity']
        The question to plot.
    terms : TermCollection, optional
        The collection of terms to plot. Default is TERMS_ALL.
    figsize : tuple, optional
        The size of the figure in inches. Default is (12, 15).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the heatmap.
    '''

    if question not in ['q1_sentiment', 'q2_maturity', 'q3_stage']:
        if question.lower() in ['sentiment', 'maturity', 'stage']:
            question = f'q{1 if question.lower() == "sentiment" else 2 if question.lower() == "maturity" else 3}_{question.lower()}'
        else:
            raise ValueError(f"Question '{question}' should be either 'q1_sentiment', 'q2_maturity', 'q3_stage', 'sentiment', 'maturity', or 'stage'.")
    if not isinstance(terms, TermCollection):
        if isinstance(terms, Term):
            terms = TermCollection([terms])
        elif isinstance(terms, str) and terms.lower() == 'all':
            terms = df['term_alias'].unique()
            terms = [TERMS_ALL[t] for t in terms]
        elif isinstance(terms, list):
            terms = [TERMS_ALL[t] for t in terms]
        else:
            raise ValueError(f"Terms should be a TermCollection or 'all', got {type(terms)} instead.")

    #fig, axs = plt.subplots(len(terms), df['report_type'].unique().shape[0], figsize=figsize, sharey=True, sharex=True)

    fig = plt.figure(layout="constrained", figsize=figsize)

    outer = fig.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[60, 1],
        width_ratios = [1, 18, 1]
    )

    n_rts = df['report_type'].unique().shape[0]
    main = outer[0, :].subgridspec(len(terms), n_rts)

    axs = np.array([
        [fig.add_subplot(main[i, j]) for j in range(n_rts)]
        for i in range(len(terms))
    ])

    cmaps = {
        'q1_sentiment': 'RdYlGn',
        'q2_maturity': discrete_cmap([1, 2, 3, 4, 5, 6, 7, 8, 9], cmap_name='cividis'),
        'q3_stage': discrete_cmap([1, 2, 3, 4, 5], cmap_name='plasma'),
    }
    if cmap is not None:
        cmaps[question] = cmap


    for i, term in enumerate(terms):
        # add term alias as a y-axis label for the first column
        axs[i, 0].set_ylabel(term.alias.replace(' ', '\n'), rotation=25, labelpad=40, fontweight='bold', va='center', fontfamily='monospace')
        for j, rt in enumerate(df['report_type'].unique()):
            df_question = df.loc[(df['report_type'] == rt) & (df['term_alias'] == term.alias)].pivot(index='company', columns='year', values=question)
            if question == 'q3_stage':
                df_question = df_question.replace({
                    "research and development": 1,
                    "pilot": 2,
                    "demonstration": 3,
                    "commercial-deployment": 4,
                    "phasing-out": 5,
                    "unclear": np.nan,
                    None: np.nan
                }).astype(float)
            else:
                df_question = df_question.astype(float)
            im = axs[i, j].imshow(
                df_question.values,
                cmap=cmaps[question],
                vmin = 1 if question in ('q2_maturity', 'q3_stage') else -1,
                vmax = 9 if question == 'q2_maturity' else 5 if question == 'q3_stage' else 1
            )
            if i == 0:
                axs[i, j].set_title(f"{REPORT_TYPES_LONG.get(rt, rt).title()}s", fontweight='bold', fontfamily='monospace')
            axs[i, j].set_yticks(np.arange(len(df_question.index)),
                                  df_question.reset_index()['company'].apply(lambda x: COMPANY_ALIASES.get(x, x)),
                                  rotation = 0,# va = 'center', 
                                  fontfamily='monospace',
                                  fontsize = 8)
            axs[i, j].set_xticks(np.arange(len(df_question.columns)),minor = True)
            axs[i, j].set_xticks(
                np.arange(len(df_question.columns), step = 5),
                np.arange(df_question.columns.astype('int').min(), df_question.columns.astype('int').max() + 1, 5), 
                minor = False, 
                rotation = 90,
            )

    for ax in axs.flat:
        ax.label_outer()

    # add colorbar to the bottom of the figure, spanning all columns
    #cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.01])
    if 'sentiment' in question.lower():
        ticks = [-1, -0.5, 0, 0.5, 1]
        min_max = (-1, 1)
        cbar_label = 'Sentiment'
    elif 'maturity' in question.lower():
        ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        tick_labels = ticks
        min_max = (0, 9)
        cbar_label = 'Technology Readiness Level'
    elif 'stage' in question.lower():
        ticks = [1, 2, 3, 4, 5]
        tick_labels = ['R&D', 'Pilot', 'Demonstration', 'Deployment', 'Phasing Out']
        min_max = (0, 5)
        cbar_label = 'Commercialisation Stage'

    # cr = mpl.colorizer.Colorizer(cmap=cmaps[question], norm=mpl.colors.Normalize(vmin=min_max[0], vmax=min_max[1]))
    # cbar = fig.colorbar(mpl.colorizer.ColorizingArtist(cr), cax=cbar_ax, orientation='horizontal')

    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=min_max[0], vmax=min_max[1]),
        cmap=cmaps[question],
    )
    mappable.set_array([])

    cax = fig.add_subplot(outer[1, 1])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")

    if 'sentiment' in question.lower():    
        cbar.set_ticks(ticks)
    else:
        bounds = [0] + ticks
        cbar.set_ticks([(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds) - 1)])
        cbar.set_ticklabels(tick_labels)
        cbar.minorticks_off()
    cbar.set_label(cbar_label, rotation=0, fontweight='bold', fontfamily='monospace')
    #fig.subplots_adjust(bottom=0.20)

    fig.suptitle(f"{question.split('_')[1].upper()}", fontsize=16, fontweight='bold', fontfamily='monospace')
    fig.set_constrained_layout(True)

    if return_type.lower() == 'figure':
        return fig
    elif return_type.lower() == 'axs' or return_type.lower() == 'axes':
        return axs
    elif return_type.lower() == 'both':
        return fig, axs
