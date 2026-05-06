import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0. Imports & Setup
    """)
    return


@app.cell
def _():
    import pandas as pd
    import re
    import os
    import pymupdf
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from typing import Iterable
    from dataclasses import dataclass

    pymupdf.TOOLS.mupdf_display_errors(on = None)
    pymupdf.TOOLS.mupdf_display_warnings(on = None)
    return Iterable, dataclass, mo, np, os, pd, plt, pymupdf, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Functions
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class Term:
        alias: str
        regex: str
        colour: str


    TERMS = [
        Term('CCS', 'carbon capture|ccs', 'tab:gray'),
        Term('hydrogen', 'hydrogen', 'tab:cyan'),
        Term('wind', 'wind', 'tab:green'),
        Term('solar', 'solar', 'tab:orange')
    ]
    return TERMS, Term


@app.cell
def _(Iterable, TERMS, Term, pymupdf, re):
    def search_file(
        file:str|pymupdf.Document, 
        terms: Iterable[Term] = TERMS, 
        *, 
        count_type: str = 'all', norm_type: str = 'words', norm:bool = True
    ) -> dict[str, int] | dict[str, int | float]:

        if isinstance(file, str): 
                doc = pymupdf.open(file) 
        elif isinstance(file, pymupdf.Document): 
                doc = file
        else:
            raise TypeError(f'file must be a filepath or Document instance. Got {type(file)}.')

        out= {term.alias: 0 for term in terms}

        n_words = 0

        for page in doc:
            txt = page.get_text().lower()

            if norm_type == 'words':
                n_words += len(re.findall(r"[^\W_]+", txt, re.M))

            for term in terms:
                matches = re.findall(term.regex, txt, re.I)
                if count_type == 'pages':
                    out[term.alias] += 1
                else:
                    out[term.alias] += len(matches)

        n_pages = len(doc)

        if norm_type.lower() == 'pages' and norm:
            out = {term: out[term]/n_pages for term in out}
        elif norm_type.lower() == 'words':
            if norm:
                out = {term: out[term]/n_words for term in out}
            out['n_words'] = n_words


        out['n_pages'] = n_pages

        return out

    search_file('reports/Shell/Annual/shell-annual-report-2025.pdf', norm_type='words', norm = True)
    return (search_file,)


@app.cell
def _(Iterable, TERMS, Term, os, pd, re, search_file):
    def extract_year(filename: str) -> int:
        match = re.findall(r"\d{4}", filename)
        n_matches = len(match)
        if n_matches == 0:
            raise ValueError(f'filename does not contain a 4-digit year: {filename}')
        elif n_matches > 1:
            raise ValueError(f'more than 1 4-digit number found in filename: {filename}')
        else:
            return int(match[0])

    def search_files(
        folder:str, 
        terms: Iterable[Term] = TERMS, 
        *,
        count_type:str = 'all', norm: bool = True,
        year_index: bool = True
    ) -> pd.DataFrame:
        if not os.path.isdir(folder):
            raise ValueError('folder is not a folder!')

        out = {}
        for file in os.listdir(folder):
            if file.endswith('.pdf'):
                out[file] = search_file(os.path.join(folder, file), terms, count_type=count_type, norm=norm)
                out[file]['year'] = extract_year(file)

        out = pd.DataFrame(out).T
        out.index.name = 'filename'
        if year_index: out = out.reset_index().set_index('year')
        return out.sort_index()


    return (search_files,)


@app.cell
def _(Iterable, TERMS, Term, pd, plt):
    def plot_mentions(
        df: pd.DataFrame, 
        *, 
        ax: plt.Axes | None = None, figsize: tuple[float]|tuple[int]|None = None,
        linestyle: str|None = '--', marker: str|None = 'x',
        plot_n: str = 'words', normed_by: str = 'words',
        terms: Iterable[Term] = TERMS
    ):

        if ax is None: _, ax = plt.subplots(figsize=figsize)

        for term in terms:
            ax.plot(df.index, df[term.alias], marker = marker, linestyle = linestyle, color = term.colour)
        ax.set_ylabel(f'Term count/No. {normed_by}')
        ax.set_xticks(df.index)
        ax.set_xticklabels(df.index, rotation = 45)

        if not plot_n.startswith('n_'): plot_n = 'n_' + plot_n
        if plot_n in df:
            ax_n = ax.twinx()
            df[plot_n].plot(linestyle = 'None', marker = '2', color = 'k', alpha = 0.9, ax = ax_n)
            ax_n.set_ylabel(f'No. {plot_n.replace("n_", "")}')
            return ax, ax_n
        return ax

    return (plot_mentions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Pipeline run
    """)
    return


@app.cell
def _(Term):
    Term()
    return


@app.cell
def _(
    np,
    os,
    plot_mentions,
    plt,
    search_files,
    term_aliases,
    term_colours,
    terms,
):


    company_aliases = {
        'shell': 'Shell',
        'bp': 'BP',
        'exxon': 'ExxonMobil',
        'chevron': 'Chevron',
        'total': 'TotalEnergies'
    }



    cos = os.listdir('reports')
    if '.DS_Store' in cos: cos.remove('.DS_Store')
    report_types = {'annual': 'an', 'sustainability-report': 'sr'}

    dfs = {}
    for co in cos:
        if co.lower() == '.ds_store': continue
        for report_type in report_types:
            dfs[f'{co}-{report_types[report_type]}'] = search_files(
                os.path.join('reports', co, report_type),
                terms = [term.alias for term in terms]
            )


    years_an = sorted({int(y) for co in cos for y in dfs[co + '-an'].index})
    years_sr = sorted({int(y) for co in cos for y in dfs[co + '-sr'].index})

    xmin_an, xmax_an = years_an[0], years_an[-1]
    xmin_sr, xmax_sr = years_sr[0], years_sr[-1]


    _fig, _axs = plt.subplots(len(cos), 2, figsize = (12, 3*len(cos) + 1))
    for i, _co in enumerate(cos):
        _axs1 = _axs[i][0]
        _axs2 = _axs[i][1]

        plot_mentions(dfs[_co+'-an'], ax = _axs1, linestyle=None, marker = None, terms = term_colours)
        plot_mentions(dfs[_co+'-sr'], ax = _axs2, linestyle=None, marker = None, terms = term_colours)

        _axs1.set_xlim(xmin_an, xmax_an)
        _axs2.set_xlim(xmin_sr, xmax_sr)

        if i == 0:
            _axs1.set_title('Annual Reports')
            _axs2.set_title('Sustainability Reports')

        if i < len(cos) - 1:
            _axs1.set_xticks([])
            _axs2.set_xticks([])
            _axs1.set_xticklabels([])
            _axs2.set_xticklabels([])
        else:
            _axs1.set_xticks(np.arange(xmin_an, xmax_an + 1, 2))
            _axs2.set_xticks(np.arange(xmin_sr, xmax_sr + 1, 2))
            _axs1.set_xticklabels(np.arange(xmin_an, xmax_an + 1, 2))
            _axs2.set_xticklabels(np.arange(xmin_sr, xmax_sr + 1, 2))
            _axs1.set_xlabel('Year')
            _axs2.set_xlabel('Year')


    _handles = [plt.Line2D([0], [0], color = term_colours[term], marker = None, linestyle = None, label = term_aliases.get(term, term.capitalize())) for term in term_colours]
    _handles += [plt.Line2D([0], [0], color = 'k', marker = '2', linestyle = 'None', label = 'Total words')]
    _fig.legend(
        handles = _handles,
        loc = 'lower center', ncol = len(term_colours), bbox_to_anchor = (0.5, -0.03),
        alignment = 'center'

    )

    _fig.tight_layout(pad = 0.75)



    for i, _co in enumerate(cos):
        _axs1 = _axs[i][0]

        col_bot = _axs1.get_position().y0
        col_top = _axs1.get_position().y1
        x = _axs1.get_position().x0 - 0.1
        y = _axs1.get_position().y0 + (col_top - col_bot) / 2
        _fig.text(x, y, company_aliases.get(_co, _co.capitalize()), ha='center', va='center', fontweight='light', fontsize = 16, rotation = 90)

    plt.show()
    return


if __name__ == "__main__":
    app.run()
