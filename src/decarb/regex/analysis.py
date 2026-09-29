__all__ = [
    'chunk_position_metrics', 'report_term_metrics', 
    'df_report_term_metrics'
]

import re

import numpy as np
import pandas as pd

from decarb.companies import Company, CompanyCollection, Report, co_files
from decarb.terms import TermCollection, TERMS_ALL
from decarb.util import logger, record_id


SECTIONS = ('front', 'middle', 'back')

# Headings are lowercase with all non-alphanumeric characters removed to 
# account for markdown emphasis and whitespace added in converison
_FRONT_HEADINGS = (
    'chairman', 'chiefexecutive', 'chairmansstatement', 'letterfromour', 'letterfromthe',
    'ceoletter', 'messagefrom', 'ourstrategy', 'strategicreport', 'businessmodel', 'ataglance',
    'highlights', 'aboutthisreport', 'whoweare', 'ourapproach', 'introduction', 'ourpurpose',
)
_BACK_HEADINGS = (
    'notestoconsolidated', 'notestothe', 'consolidatedstatement', 'consolidatedbalance',
    'financialstatements', 'independentauditor', 'auditorsreport', 'assurancestatement',
    'cautionarystatement', 'forwardlooking', 'glossary', 'definitions', 'abbreviations',
    'investorinformation', 'shareholderinformation', 'stockholderandinvestor', 'boardofdirectors',
    'executivecommittee', 'annualmeeting', 'dividendpayment', 'electronicaccess',
    'supplementalinformation', 'supplementaryinformation', 'criticalaccounting',
    'newaccountingstandards', 'transactionswithrelatedparties', 'remunerationreport',
)


def chunk_sections(chunks: list[str]) -> list[str | None]:
    '''
    Classify each chunk as front, middle or back matter using its heading.
    Chunks with no heading or with headings not present in `_FRONT_HEADINGS`/`_BACK_HEADINGS`
    are classified as "middle". Chunks with no heading are classified as None.

    Parameters
    ----------
    chunks : list[str]
        The report's chunks, in document order.

    Returns
    -------
    list[str | None]
        One entry per chunk: "front", "middle", "back", or None where no heading was found.
    '''
    out = []
    for chunk in chunks:
        heading = re.search(r'^#{1,6}\s*(.+)$', chunk, re.M)
        if heading is None:
            out.append(None)
            continue
        norm = re.sub(r'[^a-z0-9]', '', heading.group(1).lower())  # normalise heading to match against the lists
        if any(pat in norm for pat in _FRONT_HEADINGS):
            out.append('front')
        elif any(pat in norm for pat in _BACK_HEADINGS):
            out.append('back')
        else:
            out.append('middle')
    return out

def chunk_position_metrics(indices: list[int], n_chunks: int) -> dict:
    '''
    Positional statistics for a term's chunk hits within a report.
    Returns a dict with the following keys:
    - `first_pos`, `mean_pos`, `last_pos`: the first, mean and last chunk indices containing the term, expressed as a fraction of the report's length (0 is the start, 1 is the end).
    - `prominence`: the hit rate with each hit weighted by how early it appears.
    - `front_decile_frac`: the proportion of this term's hits falling in the first tenth of the document.
    
    Parameters
    ----------
    indices : list[int]
        The indices of the chunks containing the term.
    n_chunks : int
        The total number of chunks in the report.

    Returns
    -------
    dict
        Positional statistics as described above. If `n_chunks` is 0 or `indices` is empty, returns NaN for 
        the positions and 0 for prominence.
    '''
    if n_chunks <= 0 or not indices:
        return {'first_pos': np.nan, 'mean_pos': np.nan, 'last_pos': np.nan,
                'prominence': 0.0 if n_chunks > 0 else np.nan, 'front_decile_frac': np.nan}

    pos = np.asarray(indices, dtype=float) / n_chunks
    return {
        'first_pos': float(pos.min()),
        'mean_pos': float(pos.mean()),
        'last_pos': float(pos.max()),
        'prominence': float((1.0 - pos).sum() / n_chunks),
        'front_decile_frac': float((pos < 0.1).mean()),
    }

def _centred_log_ratio(counts, pseudocount: float = 0.5) -> np.ndarray:
    '''
    Centred log-ratio transform of a compositional vector. Allows shares to be used in linear models
    by removing the unit-sum constraint. `pseudocount` is added before normalising to avoid division by zero.
    '''
    x = np.asarray(counts, dtype=float) + pseudocount
    total = x.sum()
    if total <= 0:
        return np.full(x.shape, np.nan)
    log_x = np.log(x / total)
    return log_x - log_x.mean()

def report_term_metrics(report: Report, terms: TermCollection = TERMS_ALL) -> dict | None:
    '''
    Measures of term locations and counts for a single report. Counts are 
    of chunks containing a term, not occurrences of the term. 
    Returns None if the report has no search results for the collection.


    Parameters
    ----------
    report : Report
        The report to measure. Its search results for `terms` must already exist on disk.
    terms : TermCollection
        The term collection to measure over. Default is `TERMS_ALL`.

    Returns
    -------
    dict | None
        A dict keyed by term alias. Has the following keys:
       - `n_chunks`: The number of chunks in the report.
       - `n_hits`: The number of chunks containing the term.
       - `rate`: The proportion of chunks containing the term.
       - `share`: The proportion of all term hits in the report that are for this term.
       - `first_pos`, `mean_pos`, `last_pos`: The first, mean and last chunk indices containing the term, expressed as a fraction of the report's length.
       - `prominence`: The hit rate weighted by how early the hits appear in the report.
       - `front_decile_frac`: The proportion of this term's hits falling in the first tenth of the document.
       - `n_hits_front`, `n_hits_middle`, `n_hits_back`: The number of hits in the front, middle, and back sections of the document.
        `n_hits_middle`, `n_hits_back`) with `front_frac` giving the front-matter proportion.
        None if no search results were found.
    '''
    search_results = report.search_results(terms)
    if search_results is None:
        return None

    n_chunks = report.n_chunks
    hits = {term.alias: list(search_results.get(term.alias, [])) for term in terms}
    total_hits = sum(len(idx) for idx in hits.values())
    sections = chunk_sections(report.chunks)

    out = {}
    for term in terms:
        idx = hits[term.alias]
        section_counts = {f'n_hits_{s}': 0 for s in SECTIONS}
        for i in idx:
            section = sections[i] if i < len(sections) else None
            if section is not None:
                section_counts[f'n_hits_{section}'] += 1
        n_classified = sum(section_counts.values())
        out[term.alias] = {
            'n_chunks': n_chunks,
            'n_hits': len(idx),
            'rate': len(idx) / n_chunks if n_chunks else np.nan,
            'share': len(idx) / total_hits if total_hits else np.nan,
            **chunk_position_metrics(idx, n_chunks),
            **section_counts,
            'front_frac': section_counts['n_hits_front'] / n_classified if n_classified else np.nan,
        }

    return out


def df_report_term_metrics(
        terms: TermCollection = TERMS_ALL, *,
        companies: CompanyCollection | Company | None = None,
        report_types: list[str] | None = None,
        groups: dict[str, str] | None | bool = None,
        clr: bool = True, clr_pseudocount: float = 0.5
) -> pd.DataFrame:
    '''
    Build a long dataframe of attention measures, with one row per report and term.

    Rows are indexed by the same id convention as the LLM results dataframe
    (`{company}_{report_type}_{year}_{term}`), so the two can be joined directly. Every report
    with search results contributes a row for every term in the collection, including terms that
    matched nothing, so the frame has no missing cells.

    `share` is computed across every term in the collection. `group_share` is computed across only
    the terms sharing a row's group, so decarbonisation shares are unaffected by whether fossil
    terms are present in the collection.

    Parameters
    ----------
    terms : TermCollection
        The term collection to measure over. Default is `TERMS_ALL`.
    companies : CompanyCollection | Company | None
        The companies to include. If None, all companies in the data directory are included.
    report_types : list[str] | None
        The report types to include (e.g. "annual-report"). If None, all report types are included.
    groups : dict[str, str] | None | bool
        A mapping of term alias to group name, used to compute `group_share`. If None, each term's
        `group` attribute is used ("fossil" or "decarbonisation"). Pass False to omit the `group`
        and `group_share` columns. Aliases absent from a supplied mapping get a null group and a
        null `group_share`.
    clr : bool
        Whether to add a centred log-ratio column for `n_hits` within each report.
    clr_pseudocount : float
        Added to every count before normalising, to avoid division by zero. Default is 0.5.
        Ignored if `clr` is False.

    Returns
    -------
    pd.DataFrame
        Columns: company, company_alias, report_type, report_name, year, term_alias, term_fname,
        n_chunks, n_hits, rate, share, first_pos, mean_pos, last_pos, prominence,
        front_decile_frac, n_hits_front, n_hits_middle, n_hits_back, front_frac, and (unless
        `groups` is False) group and group_share.
    '''
    rows = []
    n_reports = n_skipped = 0

    for report in co_files('Report', companies=companies, report_types=report_types):
        attention = report_term_metrics(report, terms)
        if attention is None:
            n_skipped += 1
            logger.warning(f'NO SEARCH RESULTS: {report.report_name} for terms "{terms}". Skipping.')
            continue
        n_reports += 1

        for term in terms:
            rows.append({
                'id': record_id(report.company.folder_name, report.report_type, report.year, term.prompt_fname),
                'company': report.company.folder_name,
                'company_alias': report.company.full_name,
                'report_type': report.report_type,
                'report_name': report.report_name,
                'year': int(report.year),
                'term_alias': term.alias,
                'term_fname': term.prompt_fname,
                **attention[term.alias],
            })

    logger.debug(f'ATTENTION DF BUILT: {n_reports} reports, {n_skipped} skipped, {len(rows)} rows, terms="{terms}"')
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index('id').sort_values(by=['company', 'report_type', 'year', 'term_alias'])
    if groups is not False:
        groups = {term.alias: term.group for term in terms} if groups is None else groups
        df['group'] = df['term_alias'].map(groups)
        totals = df.groupby(['report_name', 'group'], dropna=False)['n_hits'].transform('sum')
        df['group_share'] = (df['n_hits'] / totals).where(totals > 0)
    
    # add clr column to allow shares to be used in linear models
    if clr:
        df['clr'] = (
            df.groupby(['company', 'report_type', 'year'], sort=False)['n_hits']
            .transform(lambda s: _centred_log_ratio(s.to_numpy(), pseudocount=clr_pseudocount))
        )
    return df