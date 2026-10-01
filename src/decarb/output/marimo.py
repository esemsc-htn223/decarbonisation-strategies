__all__ = [
    'search_results_chunks',
    'min_max_chunks',
    'random_chunks',
    'search_summary',
    'results_tabs',
]

import marimo as mo
import re
import numpy as np
import matplotlib as mpl
import os
from typing import Literal
import pandas as pd

from decarb.companies import Report, COMPANY_ALIASES
from decarb.util import REPORT_TYPES_LONG
from decarb.terms import (
    Term, TermCollection, TERMS_ALL
)
from decarb.output.plot import llm_timeseries

COLOUR_STG = '#8AC4FF'
COLOUR_MAT = '#E5B8FF'
COLOUR_SENT = '#FF8A8A'
COLOR_CMT = "#E8FF8A"

STG_MAP = {
    "research and development": {
        'alias': "1/4 - Research and Development",
        'alias_short': "1/4 - R+D",
        'style': {'color': '#8AC4FF'}
    },
    "pilot": {
        'alias': "2/4 - Pilot",
        'alias_short': "2/4 - Pilot",
        'style': {'color': "#1F8FFF"}
    },
    "demonstration": {
        'alias': "3/4 - Demonstration",
        'alias_short': "3/4 - Demo",
        'style': {'color': '#0080FF'}
    },
    "commercial-deployment": {
        'alias': "4/4 - Deployment",
        'alias_short': "4/4 - Deploy",
        'style': {'color': '#0052A3'}
    },
    "unclear": {
        'alias': "-/4 - Unclear",
        'alias_short': "-/4 - Unclear",
        'style': {'color': '#cccccc'}
    },
    "phasing-out": {
        'alias': "0/4 - Phasing Out",
        'alias_short': "0/4 - Phase Out",
        'style': {'color': '#e65054'}
    },
    'unclear': {
        'alias': "-/4 - Unclear",
        'alias_short': "-/4 - Unclear",
        'style': {'color': '#cccccc'}
    }
}

MAT_MAP = {
    1: {
        'alias': "1/9 - Initial idea reported",
        'alias_short': "1/9 - Idea",
        'style': {'color': '#F8CDF9'}
    },
    2: {
        'alias': "2/9 - Concept development",
        'alias_short': "2/9 - Concept",
        'style': {'color': '#F5BCF7'}
    },
    3: {
        'alias': "3/9 - Early-stage testing",
        'alias_short': "3/9 - Early tests",
        'style': {'color': '#F1ABF5'}
    },
    4: {
        'alias': "4/9 - Pilot implementation",
        'alias_short': "4/9 - Pilot",
        'style': {'color': '#ED9AF3'}
    },
    5: {
        'alias': "5/9 - Scale-up planning",
        'alias_short': "5/9 - Scale-up",
        'style': {'color': '#E988F0'}
    },
    6: {
        'alias': "6/9 - Commercial deployment",
        'alias_short': "6/9 - Deploy",
        'style': {'color': '#E475EE'}
    },
    7: {
        'alias': "7/9 - Market penetration",
        'alias_short': "7/9 - Market",
        'style': {'color': '#DF61EB'}
    },
    8: {
        'alias': "8/9 - Optimisation",
        'alias_short': "8/9 - Optimise",
        'style': {'color': '#D94AE9'}
    },
    9: {
        'alias': "9/9 - Mature technology",
        'alias_short': "9/9 - Mature",
        'style': {'color': '#D32CE6'}
    },
    'unclear':{
        'alias': "-/9 - Unclear",
        'alias_short': "-/9 - Unclear",
        'style': {'color': '#cccccc'}
    }
}


def search_results_chunks(report: Report, terms: TermCollection, term: Term, return_type: Literal['dict', 'list'] = 'list') -> dict[str, str]|list[str]:
    '''
    Get the chunks where a given term was found in a report, based on the search results for the given terms.

    Parameters
    ----------
    report : Report
        The report object containing the search results and chunks.
    terms : TermCollection
        The collection of terms that were searched for in the report.
    term : Term
        The specific term for which to retrieve the chunks.
    return_type : Literal['dict', 'list'], optional
        The format in which to return the chunks. If 'dict', returns a dictionary where keys are 
        chunk indices and values are chunk content. If 'list', returns a list of chunk content. Defaults to 'list'.

    Returns
    -------
    dict[str, str] | list[str]
        The chunks where the term was found, either as a dictionary or a list, depending on the specified return_type.
    '''
    search_res = report.search_results(terms)
    if search_res is None:
        raise FileNotFoundError(f'Search results not found for report {report.name} and terms {terms.folder_name}. Please run `decarb.regex.search` first to generate search results.')
    term_alias = term.alias if isinstance(term, Term) else term  # allowing str input
    if return_type == 'dict':
        return {i: open(os.path.join(report.chunks_dir, f'chunk-{i}.md')).read() for i in search_res[term_alias]}
    elif return_type == 'list':
        return [open(os.path.join(report.chunks_dir, f'chunk-{i}.md')).read() for i in search_res[term_alias]]

def min_max_chunks(chunks: list[str]) -> mo.Html:
    '''
    Create a marimo accordion showing the minimum and maximum chunks based on length.

    Parameters
    ----------
    chunks : list[str]
        The list of chunk contents.

    Returns
    -------
    mo.Html
        The marimo accordion containing the minimum and maximum chunks.
    '''
    min_len = min(len(c) for c in chunks)
    max_len = max(len(c) for c in chunks)

    # identify chunk with minimum length
    min_chunk = min(chunks, key=lambda c: len(c))
    min_chunk_index = chunks.index(min_chunk)
    max_chunk = max(chunks, key=lambda c: len(c))
    max_chunk_index = chunks.index(max_chunk)

    return mo.accordion(
        {
            f'Chunk {min_chunk_index - 1}': mo.md(chunks[min_chunk_index - 1]),
            f'**Min Chunk** ({min_len} chars): Chunk {min_chunk_index}': mo.md(min_chunk),
            f'Chunk {min_chunk_index + 1}': mo.md(chunks[min_chunk_index + 1]),
            f'Chunk {max_chunk_index - 1}': mo.md(chunks[max_chunk_index - 1]),
            f'**Max Chunk** ({max_len} chars): Chunk {max_chunk_index}': mo.md(max_chunk),
            f'Chunk {max_chunk_index + 1}': mo.md(chunks[max_chunk_index + 1]),
        },
        multiple=True
    )

def random_chunks(chunks: list[str], n: int = 5) -> mo.Html:
    '''
    Create a marimo accordion showing a random sample of chunks.

    Parameters
    ----------
    chunks : list[str]
        The list of chunk contents.
    n : int, optional
        The number of random chunks to display. Defaults to 5.

    Returns
    -------
    mo.Html
        The marimo accordion containing the random sample of chunks.
    '''
    indices = np.random.choice(len(chunks), size=n, replace=False)
    return mo.accordion(
        {f'Chunk {i}': mo.md(chunks[i]) for i in indices},
        multiple=True
    )

def search_summary(report: Report, terms: Term|TermCollection = TERMS_ALL) -> mo.Html:
    '''
    Create a marimo accordian showing search results for the given terms and chunks.

    Parameters
    ----------
    chunks : list[str] | dict[int, str] | str
        The markdown chunks that were searched, either as a list of chunk content (sorted by chunk index) or a dictionary where the keys are chunk indices and the values are the chunk content.
        Or a string path to a directory containing chunk markdown files named in the format `chunk-{index}.md`.
    terms : Term | TermCollection
        The term or collection of terms that were searched for.
    search_res : dict[str, list[int]]
        The search results, as returned by `search_chunks`.
    terms : Term | TermCollection
        The term or collection of terms that were searched for.

    Returns
    -------
    mo.Html
        A marimo accordion showing the search results, with terms as top-level accordions and chunks
    '''
    search_res = report.search_results(terms)
    if search_res is None:
        raise FileNotFoundError(f'Search results not found for report {report.name} and terms {terms.folder_name}. Please run `search_chunks` first to generate search results.')

    # enforce dict format
    chunks = report.chunks_dir
    if chunks is None:
        raise FileNotFoundError(f'Chunks directory not found for report {report.name}. Please run `search_chunks` first to generate search results and ensure chunks are saved in the expected directory structure.')
    chunks_dict = {int(f.split('-')[1].split('.')[0]): open(os.path.join(chunks, f)).read() for f in sorted(os.listdir(chunks)) if f.endswith('.md')}
    
    master_accordion_content = {}
    for term in terms:
        # display the chunks where the term was found, with the term highlighted
        term_accordion_content = {f'Chunk {i}': chunks_dict[i] for i in search_res[term.alias]}
        term_accordion_content = dict(sorted(term_accordion_content.items(), key=lambda x: int(x[0].split()[1])))  # sort by chunk index
        n_matches = {}
        for label, chunk in term_accordion_content.items():
            matches = re.finditer(term.regex, chunk, re.I)
            prev_shift = 0
            n_matches[label] = 0
            for match in matches:
                start, end = match.span()
                chunk = chunk[:start + prev_shift] + '<mark>' + chunk[start + prev_shift:end + prev_shift] + '</mark>' + chunk[end + prev_shift:]
                prev_shift += len('<mark></mark>')
                n_matches[label] += 1
            term_accordion_content[label] = mo.md(chunk)

        # rename with number of matches and formatting
        for label in list(term_accordion_content.keys()):
            if label in term_accordion_content:
                term_accordion_content[f'>**{label}:** {n_matches[label]} match{n_matches[label] > 1 and "es" or ""}'] = term_accordion_content.pop(label)
        
        # add to master accordion with number of chunks where term was found
        n_chunks = len(search_res[term.alias])
        _s = 's' if n_chunks != 1 else ''
        master_accordion_content[f'**{term.alias}:** {n_chunks} chunk{_s}'] = mo.vstack([
            #mo.md(f"Term: **{_term.alias}**"),
            mo.md(f"`{term.regex}`"),
            mo.accordion(term_accordion_content, multiple=True)
        ])


    return mo.accordion(master_accordion_content, multiple=False)


def _mark_term_mentions(md: str, term: Term):
    matches = re.finditer(term.regex, md, re.I)
    prev_shift = 0
    for match in matches:
        start, end = match.span()
        md = md[:start + prev_shift] + f'<u><mark>{md[start + prev_shift:end + prev_shift]}</mark></u>' + md[end + prev_shift:]
        prev_shift += 20  # len('<u><mark></mark></u>')
    return md

def _mark_quote(md: str, quotes: list, colour: str = '#ffcccb', add_label:str = '') -> str:

    for quote in quotes:
        if isinstance(quote,dict):
            quote = quote['quote']
        if '...' in quote:
            for q in quote.split('...'):
                if q.strip():
                    md = md.replace(q, f'<mark style="background-color: {colour}">{q}{add_label}</mark>')
        md = md.replace(quote, f'<mark style="background-color: {colour}">{quote}{add_label}</mark>')
    return md

def _quote_stack(quotes: list) -> mo.Html:
    quotes_stk = []
    for q in quotes:
        quotes_stk.append(
            mo.vstack([
                mo.md(
                    f"> *\"{q['quote']}\"*"
                ),
                mo.ui.text_area(
                    q['justification'],
                    full_width = True,
                    rows = 3,
                ),
            ])
            )
    return mo.vstack(quotes_stk)

def commitment_accordion(commitments: list) -> mo.Html:
    quotes_acc = {}
    for c in commitments:
        quotes_acc[f"*{c['commitment']}*"] = mo.vstack([
                mo.md(
                    f"> *\"{c['quote']}\"*"
                ),
                mo.ui.text_area(
                    c['justification'],
                    full_width = True,
                    rows = 3,
                ),
                mo.ui.button(
                    label = 'Verifiable' if c['verifiable'] else 'Not Verifiable',
                    kind = 'success' if c['verifiable'] else 'danger',
                    disabled = True, full_width = True
                )
            ])
    return mo.accordion(quotes_acc)

def _entry_styles(entry: pd.Series, short: bool = False) -> tuple:
    '''Look up the maturity/stage display details and inline styles for an entry. If `short` is True, use the short alias for maturity/stage.'''
    mat_details = MAT_MAP.get(entry['q2_maturity'], MAT_MAP['unclear'])
    stg = entry['q3_stage']
    stg_details = STG_MAP.get(stg, STG_MAP['unclear'])
    style = lambda details: '; '.join(f"{k}: {v}" for k, v in details["style"].items())
    if short:
        mat_details['alias'] = mat_details.get('alias_short', mat_details['alias'])
        stg_details['alias'] = stg_details.get('alias_short', stg_details['alias'])
    return mat_details, style(mat_details), stg_details, style(stg_details)

def _term_year_key(entry: pd.Series) -> str:
    '''
    Build the accordion label for a single term-year entry.

    Parameters
    ----------
    entry : pd.Series
        A pandas Series representing a single row of the term-year analysis DataFrame.

    Returns
    -------
    str
        A string representing the label for the accordion, including year, sentiment, stage, maturity, 
        and number of commitments.
    '''
    yr = entry['year']
    if not entry['report_type_published']:
        return f'<span style="color: lightcoral">**{yr}:** Report type "{entry["report_type"]}" not published</span>'

    mat_details, mat_style, stg_details, stg_style = _entry_styles(entry, short = True)
    snt = entry['q1_sentiment']

    # handle missing / non-numeric sentiment
    if pd.isna(snt):
        snt_display = "  - "
        snt_colour = "#cccccc"
    else:
        snt_display = f'{snt}' if snt < 0.0 else f' {snt}'
        try:
            snt_cmap = mpl.colormaps['RdYlGn'].resampled(21)
            snt_colour = mpl.colors.to_hex(snt_cmap((float(snt) + 1) / 2))
        except (TypeError, ValueError):
            snt_colour = "#cccccc"
    frm = entry['q1_framing']  # either 'opportunity', 'neutral', or 'risk'
    if pd.isna(frm):
        frm = "unclear"
    else:
        frm_display = {
            'opportunity': '↗',
            'neutral': '→',
            'risk': '↘',
        }[frm] 
        frm_colour = {
            'opportunity': 'green',
            'neutral': 'grey',
            'risk': 'red',
        }[frm]

    key = f'**{yr}:**|{snt_display} <span style="color: {snt_colour}">█</span>{frm_display} \t| <span style="{stg_style}">{stg_details["alias"]}</span> \t| <span style="{mat_style}">{mat_details["alias"]}</span>'
    n_commitments = len(entry['q4_commitments'])
    if n_commitments > 0:
        key += f'\t| {n_commitments} commitment{"s" if n_commitments != 1 else ""}'
    return f'<span style="font-family: monospace; white-space: pre">{key}</span>'

def term_year_analysis(entry: pd.Series, terms: TermCollection, *, return_key: bool = False, lazy_chunks: bool = True) -> mo.Html | tuple[str, mo.Html]:
    '''
    Display the analysis results for a single term-year entry, 
    including sentiment, stage, maturity, commitments, and the relevant chunks.
    
    Parameters
    ----------
    entry : pd.Series
        A pandas Series representing a single row of the term-year analysis DataFrame.
    terms : TermCollection
        The collection of terms used in the analysis.
    return_key : bool, optional
        Whether to return the term-year key along with the analysis output. Default is False.
    lazy_chunks : bool, optional
        Whether to load the chunks lazily (only when the user expands the accordion). Default is True.

    Returns
    -------
    mo.Html or tuple[str, mo.Html]
        A vstack of mds showing the sentiment, stage, maturity, commitments, 
        along with their respective quotes, as well as the relevant chunks for the term-year entry.
        If `return_key` is True, returns a tuple containing the term-year key and the vstack of mds.
    '''

    if not entry['report_type_published']:
        out = mo.md(f"> Report type '{entry['report_type']}' not published in {entry['year']}")
        if return_key:
            out = (_term_year_key(entry), out)
        return out

    co = entry['company_alias']
    rt = entry['report_type']
    yr = entry['year']
    term = terms[entry['term_alias']]

    mat_details, mat_style, stg_details, stg_style = _entry_styles(entry)
    snt = entry['q1_sentiment']

    if not entry['report_type_published']:
        chunks = [f"Report type '{rt}' not published in {yr}"]
    else:
        try:
            chunks = search_results_chunks(Report.from_metadata(co, rt, yr), terms, entry['term_alias'])
        except (FileNotFoundError, KeyError, ValueError) as e:
            # backup output in case of error
            out = mo.md(f'> <span style="color: lightcoral">Chunks unavailable for {co} {rt} {yr} / {entry["term_alias"]}: {e}</span>')
            if return_key:
                out = (_term_year_key(entry), out)
            return out

    if 'notes' in entry and pd.notna(entry['notes']):
        notes = f"> {entry['notes']}"
    else:
        notes = None

    # mark all quotes
    chunks = [_mark_quote(chunk, entry['q1_sentiment_quotes'], add_label=' (q1)', colour = COLOUR_SENT) for chunk in chunks]
    chunks = [_mark_quote(chunk, entry['q2_maturity_quotes'], add_label=' (q2)', colour = COLOUR_MAT) for chunk in chunks]
    chunks = [_mark_quote(chunk, entry['q3_stage_quotes'], add_label=' (q3)', colour = COLOUR_STG) for chunk in chunks]
    chunks = [_mark_quote(chunk, entry['q4_commitments'], add_label=' (q4)', colour = COLOR_CMT) for chunk in chunks]
    # and the mentions of the term
    chunks = [mo.md(_mark_term_mentions(chunk, term)) for chunk in chunks]
    n_commitments = len(entry['q4_commitments'])

    out =  mo.vstack([
                mo.vstack([
                    mo.md(notes) if notes else '',
                    mo.md('---'),
                    mo.hstack([
                        mo.md(f'**Sentiment:** {snt}'),
                        mo.md('|'),
                        mo.md(f'**Stage:** <span style="{stg_style}">{stg_details["alias"]}</span>'),
                        mo.md('|'),
                        mo.md(f'**Maturity:** <span style="{mat_style}">{mat_details["alias"]}</span>'),
                    ], justify = 'center')
                ]),
                mo.accordion({
                    f'**Q1. Sentiment** {snt}': _quote_stack(entry['q1_sentiment_quotes']),
                    f'**Q2. Maturity** <span style="{mat_style}">{mat_details["alias"]}</span>': _quote_stack(entry['q2_maturity_quotes']),
                    f'**Q3. Stage** <span style="{stg_style}">{stg_details["alias"]}</span>': _quote_stack(entry['q3_stage_quotes']),
                    f'**Q4. Commitments** ({n_commitments})': commitment_accordion(entry['q4_commitments']),
                }, multiple = True),
                mo.md('---'),

                mo.accordion({
                    f'{len(chunks)} Chunk{"s" if len(chunks) != 1 else ""}:': mo.vstack(chunks)
                }, lazy = lazy_chunks),
            ])
    if return_key:
        out = (_term_year_key(entry), out)
    return out

def company_report_results(df: pd.DataFrame, company_fname: str, terms: TermCollection, report_type: str, lazy: bool = True) -> mo.Html:
    '''
    Get the results for a given company and report type, with tabs for each term containing the sentiment timeseries, regex, and year-by-year LLM analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the results of the analysis.
    company_fname : str
        The filename of the company to get the results for. Should be one of the recognised companies, as in the keys of `companies.COMPANY_ALIASES`.
    terms : TermCollection
        The collection of terms used in the analysis.
    report_type : str
        The report type to analyse. Should be one of the recognised report types, as in the keys of `util.REPORT_TYPES_LONG`.

    Returns
    -------
    mo.Html
        A mo.ui.tabs object containing the results for the given company and report type, with tabs for each term.
    '''

    if company_fname not in df['company'].unique():
        raise ValueError(f'Company "{company_fname}" not found in dataframe. Available companies: {df["company"].unique()}')
    if report_type not in df['report_type'].unique():
        raise ValueError(f'Report type "{report_type}" not found in dataframe. Available report types: {df["report_type"].unique()}')

    company_alias = COMPANY_ALIASES.get(company_fname, company_fname)
    term_tabs = {}
    rt_long = REPORT_TYPES_LONG.get(report_type, report_type).title()
    for term in terms:
        df_term = df[(df['report_type'] == report_type) & (df['term_alias'] == term.alias) & (df['company'] == company_fname)]

        if df_term['prompt'].isna().all():
            term_tabs[f'<span style="color: lightcoral">{term}</span>'] = mo.vstack([
                mo.ui.text_area(value = term.regex, full_width = True, label = '**Regex:**'),
                mo.md(f'> <span style="color: lightcoral"> Term "{term}" not found in {company_alias}\'s {rt_long.lower()}s.</span>'),
            ])
            continue

        def make_term_panel(term=term, df_term=df_term):
            year_accordion = {
                _term_year_key(entry): (lambda entry=entry: term_year_analysis(entry, terms, return_key=False))
                for entry in (df_term[df_term['year'] == year].iloc[0] for year in df_term['year'].unique()) if entry['term_regex_found']
            }

            return mo.vstack([
                mo.ui.text_area(value = term.regex, full_width = True, label = '**Regex:**'),
                mo.hstack([
                    llm_timeseries(df_term, 'q1_sentiment', company=company_fname, term=term, report_type=report_type, width_px=550, height_px=250),
                    llm_timeseries(df_term, 'q2_maturity', company=company_fname, term=term, report_type=report_type, width_px=550, height_px=250)
                ], gap = 0, widths = 'equal', align = 'center'),
                mo.accordion(year_accordion, lazy = lazy)
            ])

        term_tabs[term.alias] = make_term_panel

    return mo.ui.tabs(term_tabs, label = 'Term', lazy = lazy)


def results_tabs(df: pd.DataFrame, terms: TermCollection = TERMS_ALL, lazy: bool = True) -> mo.Html:
    '''
    Company -> report type -> term -> year drill-down over the analysis results.

    Every level is lazy, so a cell only ever renders the branch that is currently
    open. Building the whole tree eagerly exceeds marimo's `output_max_bytes`.
    '''
    def make_company_tabs(co, terms=terms, lazy=lazy):
        # Each value is a plain callable, wrapped in mo.lazy once by lazy=True.
        rt_tabs = {
            REPORT_TYPES_LONG.get(report_type, report_type).title():
                (lambda co=co, rt=report_type: company_report_results(
                    df, company_fname=co, terms=terms, report_type=rt, lazy=lazy
                ))
            for report_type in df['report_type'].unique()
        }
        return mo.ui.tabs(rt_tabs, label = 'Report Type', lazy = lazy)

    co_tabs = {
        COMPANY_ALIASES.get(co, co): (lambda co=co: make_company_tabs(co))
        for co in df['company'].unique()
    }

    return mo.ui.tabs(co_tabs, label = 'Company', lazy = lazy)