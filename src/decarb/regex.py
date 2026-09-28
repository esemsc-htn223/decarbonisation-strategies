__all__ = ['search_file', 'search_files', 'regex_pipeline']

import pandas as pd
import re
import os
import pymupdf
from typing import Literal

from decarb.terms import TermCollection, TERMS_ALL
from decarb.companies import COMPANIES_ALL, CompanyCollection, Report, co_files
from decarb.util import logger_regex

pymupdf.TOOLS.mupdf_display_errors(on = False)
pymupdf.TOOLS.mupdf_display_warnings(on = False)

def search_report(
    report: Report, 
    terms: TermCollection = TERMS_ALL, 
    *, 
    norm:bool = False,
    norm_type: Literal['pages','chunks', 'words'] = 'words') -> dict[str, int | float]:
    '''
    Search the given text for the given terms, returning a dictionary of term counts.

    Parameters
    ----------
    report : Report
        The report to search.
    terms : TermCollection, optional
        The terms to search for, by default TERMS_ALL.
    norm : bool, optional
        Whether to normalise the counts by the number of pages or words (as specified by norm_type). 
        Defaults to False.
    norm_type : Literal['pages', 'words'], optional
        The type of normalisation to apply. If 'pages', divide the counts by the number of pages. 
        If 'words', divide the counts by the number of words. Only applied if norm is True. Defaults to 'words'.
    '''

    if not isinstance(report, Report):
        raise TypeError(f'Invalid report. Expected a Report instance, got {type(report)}.')
    if norm and norm_type.lower() not in ['pages', 'words', 'chunks']:
        raise ValueError(f'Invalid norm_type: {norm_type}. Must be "pages", "words" or "chunks".')

    if report.n_chunks == 0:
        raise ValueError(
            f'Cannot search report with 0 chunks: {report.pdf_path} - run decaarb.llm.pipeline.stage_chunk() first.'
        )  # TODO: regex should not rely upon llm pipeline - if anything it should be the other way around

    out = {term.alias: 0 for term in terms}
    #out.update({term.alias + '_chunk_indices': [] for term in terms})
    
    n_words = 0

    for i, chunk in enumerate(report.chunks_iter()):

        n_words += len(re.findall(r"[^\W_]+", chunk, re.M))

        for term in terms:
            out[term.alias] += len(re.findall(term.regex, chunk, re.I))
#            out[term.alias + '_chunk_indices'].append(i)   # same as llm search

    n_pages = len(pymupdf.open(report.pdf_path))

    if norm:
        if norm_type.lower() == 'pages':
            if n_pages == 0:
                raise ValueError(f'Cannot normalise by pages: report has 0 pages: {report.pdf_path}')
            out = {term.alias: out[term.alias]/n_pages for term in terms}
            out['normed_by'] = 'n_pages'
        elif norm_type.lower() == 'chunks':
            if report.n_chunks == 0:
                raise ValueError(f'Cannot normalise by chunks: report has 0 chunks: {report.pdf_path}')
            out = {term.alias: out[term.alias]/report.n_chunks for term in terms}
            out['normed_by'] = 'n_chunks'
        elif norm_type.lower() == 'words':
            if n_words == 0:
                raise ValueError(f'Cannot normalise by words: report has 0 words: {report.pdf_path}')
            out = {term.alias: out[term.alias]/n_words for term in terms}
            out['normed_by'] = 'n_words'
        else:
            # can't happen
            raise ValueError(f'Invalid norm_type: {norm_type}. Must be "pages" or "words".')
    else:
        out['normed_by'] = None

    out['n_words'] = n_words
    out['n_pages'] = n_pages
    out['n_chunks'] = report.n_chunks
    out['company'] = report.company.folder_name
    out['report_type'] = report.type
    out['year'] = report.year

    return out

def regex_pipeline(
        terms: TermCollection, cos: CompanyCollection = COMPANIES_ALL, *, 
        norm_type: Literal['pages','chunks', 'words'] = 'words',
        overwrite: bool = False, save: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Run the regex pipeline on a collection of reports.

        Parameters
        ----------
        terms : TermCollection
            The terms to search for.
        cos : CompanyCollection
            The companies whose reports to search.
        norm_type : Literal['pages','chunks', 'words'], optional
            The type of normalisation to apply. Default is 'words'.
        overwrite : bool, optional
            Whether to overwrite existing results. Default is False.
        save : bool, optional
            Whether to save the results. Default is True.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the normalised and unnormalised results.
        '''
        
        logger_regex.debug(f'REGEX PIPELINE START: terms={terms.folder_name}, {cos=}, {save=}, {overwrite=}, {norm_type=}')

        regex_dir = os.path.join(terms.out_folder, 'regex')

        if not os.path.exists(regex_dir):
            os.makedirs(regex_dir)

        results_normed = []
        results_unnormed = []
        for report in co_files('Report', companies = cos):
            logger_regex.debug(f'Processing report: {report}')

            results_normed.append(
                search_report(report, terms, norm=True, norm_type=norm_type)
            )
            results_unnormed.append(
                search_report(report, terms, norm=False)
            )

        results_normed = pd.DataFrame(results_normed).set_index(['company', 'report_type', 'year'])
        results_unnormed = pd.DataFrame(results_unnormed).set_index(['company', 'report_type', 'year'])

        if save:
            out_fname_normed = os.path.join(regex_dir, f'normed-{norm_type}.csv')
            out_fname_unnormed = os.path.join(regex_dir, 'unnormed.csv')
            if not os.path.exists(out_fname_normed) or overwrite:
                results_normed.to_csv(out_fname_normed)
            if not os.path.exists(out_fname_unnormed) or overwrite:
                results_unnormed.to_csv(out_fname_unnormed)
        return results_normed, results_unnormed