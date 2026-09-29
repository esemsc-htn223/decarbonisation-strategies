__all__ = [
    'pdf_to_md', 'conversion_quality', 'chunk_md',
    'search', 'search_with_metadata',
    'pipeline',
    'REGEX_BASELINES', 'LabellingSession'
]

import re
import os
import json
from typing import Literal, Iterable
import hashlib
import time
import random

from dotenv import load_dotenv
import pandas as pd
import pymupdf
import pymupdf4llm
import tiktoken
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from decarb.terms import Term, TermCollection, TERMS_ALL
from decarb.companies import Company, CompanyCollection, COMPANIES_ALL, Report, co_files
from decarb.util import logger_regex, tidy_md, resolve_path, DEFAULT_ENCODER

load_dotenv()  # in case TESSDATA_PREFIX is in .env - required for ocr of troublesome docs
pymupdf.set_messages(path = os.devnull)  # limit PyMuPDF spam (note doesnt apply to tessaract messages, which are the bulk when using OCR)
pymupdf.TOOLS.mupdf_display_errors(on = False)
pymupdf.TOOLS.mupdf_display_warnings(on = False)


MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

def pdf_to_md(report: Report, *,
              use_ocr:bool = False,
              layout_fallback:bool = True,
              save:bool = True,
              save_as:str|None = None,
              overwrite:bool=False) -> str:
    '''
    Convert the given pdf to markdown.

    Parameters
    ----------
    report : Report 
        The report to convert. The pdf path will be inferred from report.pdf_path.
    use_ocr : bool
        Whether to use OCR to extract text from the PDF, passed to pymupdf4llm.to_markdown(). Default is False.
    layout_fallback : bool
        Whether to retry a failed conversion with pymupdf4llm's layout engine disabled. Default is True.
        The classic engine performs no OCR, so a retry of a document needing OCR will not use it.
    save : bool
        Whether to save the converted markdown as a file. If False, the markdown content will be returned as a string.
    save_as : str | None, optional
        The path to save the converted markdown file to. If not provided, the markdown will be saved 
        to report._md_path, (i.e., in the same directory as the PDF but with a .md extension). 
        Only relevant if save=True.
    overwrite : bool
        Whether to overwrite existing markdown files. If False and a markdown file already exists at `save_path`, the function will skip conversion and return the existing markdown content.

    Returns
    -------
    str
        If save=True, the directory where the chunked markdown files are saved. Otherwise, the markdown content as a string.
    '''

    pdf_path = report.pdf_path
    if pdf_path is None:
        raise ValueError(f'PDF file not found for report at expected location: {report._pdf_path}')
    
    # get default save dir
    check_chunks = False
    if save:
        if save_as is None:
            save_as = report._md_path
            check_chunks = True
        else:
            if save_as == report._md_path:
                check_chunks = True
            if not save_as.endswith('.md'):
                raise ValueError(f'save_as must be a path to a markdown file ending in .md. Got {save_as}')
            

    existing_md = save_as is not None and os.path.isfile(save_as)
    if not overwrite and (existing_md or (check_chunks and report.n_chunks > 0)):
        logger_regex.debug(f'NOT CONVERTED: "{os.path.basename(pdf_path)}". overwrite=False, md file exists. pdf path: {pdf_path}')
        if save:
            return save_as
        else:
            if existing_md:
                return open(save_as).read()
            elif check_chunks and report.n_chunks > 0:
                return '\n'.join(report.chunks)


    try:
        md = pymupdf4llm.to_markdown(pdf_path, use_ocr=use_ocr)
    except Exception as e:
        if not layout_fallback:
            logger_regex.error(f'MUPDF ERROR: use_ocr={use_ocr}, not reattempting. Full error: {type(e).__name__}: {e}')
            raise
        logger_regex.warning(f'MUPDF ERROR: use_ocr={use_ocr}, retrying without the layout engine. Full error: {type(e).__name__}: {e}')
        if use_ocr:
            logger_regex.warning(f'OCR UNAVAILABLE: the layout engine is the only one that performs OCR, so "{os.path.basename(pdf_path)}" is being converted without it.')
        was_layout = getattr(pymupdf4llm, '_use_layout', True)
        try:
            pymupdf4llm.use_layout(False)
            md = pymupdf4llm.to_markdown(pdf_path)
        except Exception as e2:
            logger_regex.error(f'MUPDF ERROR: failed without the layout engine too. Full error: {type(e2).__name__}: {e2}')
            raise
        finally:
            pymupdf4llm.use_layout(was_layout)
        logger_regex.debug(f'CONVERTED WITHOUT LAYOUT ENGINE: "{os.path.basename(pdf_path)}".')
    md = tidy_md(md)
    logger_regex.debug(f'CONVERTED "{os.path.basename(pdf_path)}". len: {len(md)}. pdf path: {pdf_path}.')
    
    if save:
        if not os.path.exists(os.path.dirname(save_as)):
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
        with open(save_as, 'w') as f:
            f.write(md)
        logger_regex.debug(f'MD SAVED "{os.path.basename(save_as)}". full path: {save_as}. pdf path: {pdf_path}.')
        return save_as
    else:
        return md

def conversion_quality(
        *,
        reports: Iterable[Report] | None = None,
        companies: CompanyCollection | Company | None = None,
        report_types: list[str] | None = None,
        encoder_name: str = DEFAULT_ENCODER,
        check_type: Literal['pre-conversion', 'post-conversion'] = 'post-conversion',
        progress_bar: bool = True,
) -> pd.DataFrame:
    '''
    Measure how much text was recovered from each report's PDF, for QA of the conversion stage.

    The returned DataFrame has one row per report, with the following columns:
    - `company`: the company folder name
    - `report_type`: the report type
    - `year`: the report year
    - `n_pages`: the number of pages in the PDF
    - `n_chunks`: the number of chunks in the report (0 if not chunked)
    - `n_tokens`: the number of tokens in the report's markdown or chunked markdown
    - `tokens_per_page`: the number of tokens per page in the report's markdown or chunked markdown
    - `n_no_text_layer`: the number of pages with no text layer
    - `frac_no_text_layer`: the fraction of pages with no text layer
    - `pdf_md5`: the MD5 hash of the PDF file


    Parameters
    ----------
    reports : Iterable[Report] | None
        The reports to measure. If given, `companies` and `report_types` are ignored.
        If None, the reports are collected from `companies` and `report_types`.
    companies : CompanyCollection | Company | None
        The companies to include. If None, all companies in the data directory are included.
    report_types : list[str] | None
        The report types to include. If None, all report types are included.
    encoder_name : str
        The tiktoken encoding used to count tokens. Default is "cl100k_base".
    check_type : {"pre-conversion", "post-conversion"}
        What sort of check to run.
        - "pre-conversion" checks the PDF text layer before conversion allowing documents that need OCR to be identified.
        - "post-conversion" checks the markdown or chunked markdown after conversion, allowing documents that lost text during conversion to be identified.


        For "post-conversion", the `n_tokens` column is the token count of the markdown or chunked markdown.
        If the report has no markdown or chunks, `n_tokens` will be None and a warning will be logged.
        For "pre-conversion", `n_tokens` will not be calculated and will be None to save computation time.

    Returns
    -------
    pd.DataFrame
        One row per report, indexed by report name, sorted by `tokens_per_page` ascending so the
        worst conversions appear first.
    '''
    check_type = check_type.lower()
    if check_type not in ('pre-conversion', 'post-conversion'):
        raise ValueError(f'check_type must be "pre-conversion" or "post-conversion". Got {check_type}')

    encoder = tiktoken.get_encoding(encoder_name)
    rows = []

    if reports is None:
        reports = co_files('Report', companies=companies, report_types=report_types)
    reports = list(reports)

    n_reports = len(reports)
    with tqdm(total=n_reports, desc="checking conversion quality", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for report in reports:
            if report.pdf_path is None:
                pbar.update(1)
                continue
            try:
                doc = pymupdf.open(report.pdf_path)
                n_pages = doc.page_count
                n_no_text = sum(1 for page in doc if not page.get_text().strip())
                doc.close()
            except Exception as e:
                logger_regex.error(f'CONVERSION QA FAILED: {report.report_name}. {type(e).__name__}: {e}')
                pbar.update(1)
                continue

            if check_type == 'post-conversion' and n_pages > 0:
                if report.md_path and os.path.isfile(report.md_path):
                    n_tokens = len(encoder.encode(open(report.md_path).read()))
                elif report.n_chunks > 0:
                    n_tokens = sum(len(encoder.encode(chunk)) for chunk in report.chunks)
                else:
                    logger_regex.warning(f'CONVERSION QA: {report.report_name} has no markdown or chunks. Skipping.')
                    n_tokens = None
            else:
                n_tokens = None

            with open(report.pdf_path, 'rb') as f:
                pdf_md5 = hashlib.md5(f.read()).hexdigest()

            rows.append({
                'report_name': report.report_name,
                'company': report.company.folder_name,
                'report_type': report.report_type,
                'year': int(report.year),
                'n_pages': n_pages,
                'n_chunks': report.n_chunks,
                'n_tokens': n_tokens,
                'tokens_per_page': n_tokens / n_pages if n_tokens is not None and n_pages > 0 else None,
                'n_no_text_layer': n_no_text,
                'frac_no_text_layer': n_no_text / n_pages if n_pages > 0 else None,
                'pdf_md5': pdf_md5,  # for identifying duplicate PDFs
            })
            pbar.update(1)

    logger_regex.debug(f'CONVERSION QA: measured {len(rows)} reports.')
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('report_name').sort_values('tokens_per_page')

def chunk_md(
        report:Report, 
        *, 
        max_chunk_size:int=MAX_CHUNK_SIZE, 
        min_chunk_size:int=MIN_CHUNK_SIZE, 
        save:bool=True,
        save_as:str|None=None,
        overwrite_chunks:bool=False,
        remove_original_md:bool=False,
        encoder_name:str=DEFAULT_ENCODER,
    ) -> list[str]:
    '''
    Split the given markdown into chunks via content-aware splitting.
    Content is first split on markdown headers (##, and ###)*. Each chunk is then split to
    enforce a maximum chunk size (in the token space). Finally, small chunks are added to the
    next chunk to enforce a minimum chunk size.

    `* note '#' is not included, as this is usually reserved for the document title

    Parameters
    ----------
    report : Report
        The report object containing the markdown content to split.
    max_chunk_size : int
        Maximum chunk size, in the token space. Note due to token character length being variable,
        chunks may have a __len__ that is longer than `max_chunk_size`
    min_chunk_size : int
        Minimum chunk size. Note this is calculated by characters (not tokens).
    save : bool
        Whether to save the chunked markdown as files. Default is True. 
        If False, the chunked markdown will be returned as a list of strings.
        Will save to `report._chunks_dir` by default, or to `save_as` if provided.
    save_as : str | None
        The directory to save the chunked markdown files to. If None (default), the chunked files will 
        be saved to `report._chunks_dir` (i.e., in a 'chunks' subdirectory of the report directory). 
        Only relevant if `save=True`.
    overwrite_chunks : bool
        Whether to overwrite existing chunk files in `dir_to_save`. If False and chunk files already exist,
        the function will skip conversion.
    remove_original_md : bool
        Whether to remove the original markdown file at `md` (if `md` is a file path). If False and a markdown file already exists at `md`, 
        this file will be retained.

    Returns
    -------
    list[str]
        List of markdown chunks.
    '''
    # ensure we have md path, for consistency of removal, even if we don't need to read it
    md_path = report.md_path
    if md_path is None:
        raise ValueError(f'Markdown file not found for report at expected location: {report._md_path}')
    
    if save_as is not None:
        dir_to_save = save_as
    else:
        dir_to_save = report._chunks_dir
    
    # check if we're overwriting
    if os.path.isdir(dir_to_save):
        chunk_files = [os.path.join(dir_to_save, f) for f in os.listdir(dir_to_save) if f.endswith('.md') and f.startswith('chunk-')]
    else:
        chunk_files = []
    
    if save and not overwrite_chunks and len(chunk_files) > 0:
            logger_regex.debug(f'NOT CHUNKED: {os.path.basename(dir_to_save)}. overwrite_chunks=False, chunk files exist. Full path: {dir_to_save}')
            out = [open(f).read() for f in sorted(chunk_files)]
    else:
        # read md
        with open(md_path, 'r') as f:
            md = f.read()

        headers_to_split_on = [
            #("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        chunks = md_splitter.split_text(md)

        # enforce maximum in the token space, with some overlap, and a minimum to avoid too-short chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoder_name,
            #separators=["\n\n", "\n", " ", ""],
            chunk_size=max_chunk_size, 
            #chunk_overlap=chunk_overlap, 
            keep_separator=True
        )
        chunks = text_splitter.split_documents(chunks)

        # enforce minimum chunk size by merging short chunks with the next one
        i = 0
        while i < len(chunks):
            if len(chunks[i].page_content) < min_chunk_size:
                if i + 1 < len(chunks):
                    chunks[i] = Document(page_content=chunks[i].page_content + chunks[i + 1].page_content)
                    chunks.pop(i + 1)
                else:
                    chunks.pop(i)
            else:
                i += 1
        
        if dir_to_save is not None:
            os.makedirs(dir_to_save, exist_ok=True)
            
            for i, chunk in enumerate(chunks):
                with open(os.path.join(dir_to_save, f'chunk-{i}.md'), 'w') as f:
                    f.write(chunk.page_content)
            logger_regex.debug(f'CHUNKED to {os.path.basename(dir_to_save)}. {len(chunks)} chunks created. Full path: {dir_to_save}')
        
        out = [chunk.page_content for chunk in chunks]
        
    if remove_original_md and md_path is not None and os.path.isfile(md_path):
        os.remove(md_path)
        logger_regex.debug(f'REMOVED md file {os.path.basename(md_path)}. Full path: {md_path}')
    
    return out

def search(
    report: Report, 
    terms: TermCollection = TERMS_ALL, 
    *, 
    return_counts: bool = False,
    return_chunk_indices: bool = True,
    count_all_words: bool = False,
    save:bool = False,
    save_as:str|None = None,
    overwrite: bool = False
) -> dict[str, int | list[int]]:
    '''
    Search the given text for the given terms, returning a dictionary of term counts.
    Only a plain search - for a search with report metadata and normalisation, 
    see `search_with_metadata`.

    Parameters
    ----------
    report : Report
        The report to search.
    terms : Term|TermCollection, optional
        The term(s) to search for, by default TERMS_ALL.
    return_counts : bool, optional
        Whether to return the total number of mentions of each term. Defaults to False.
    return_chunk_indices : bool, optional
        Whether to return the indices of the chunks where each term is found. Defaults to True.
    count_all_words : bool, optional
        Whether to count all words in the report. Defaults to False.
    save : bool, optional
        Whether to save the search results as a json file - see `save_as` for location details. Defaults to False.
    save_as : str | None, optional
        The path to save the search results as a json file. 
        If None, and only return_chunk_indices is True, the search results will be saved in the default location. 
        Otherwise, must be provided if save is True.
        Defaults to None.
    overwrite : bool, optional
        Whether to overwrite the existing json file if it exists. Defaults to False.

    Returns
    -------
    dict[str, int | float]
        A dictionary mapping each term alias to its count in the report, and chunk indices if requested. 
    '''

    if not isinstance(report, Report):
        raise TypeError(f'Invalid report. Expected a Report instance, got {type(report)}.')
    
    if report.n_chunks == 0:
        raise ValueError(
            f'Cannot search report with 0 chunks: {report.pdf_path} - run decaarb.llm.pipeline.stage_chunk() first.'
        )

    if isinstance(terms, Term):
        terms = [terms]

    if return_chunk_indices and return_counts:
        # If both counts and chunk indices are requested, we need to store the chunk indices separately.
        out = {term.alias: 0 for term in terms}
        out.update({term.alias + '_chunk_indices': [] for term in terms})
    elif return_chunk_indices:
        # only chunk indices requested - store in the main alias key.
        out = {term.alias: [] for term in terms}
    else:
        # only counts requested - store in the main alias key.
        out = {term.alias: 0 for term in terms}

    if count_all_words:
        out['n_words'] = 0

    logger_regex.debug(f'SEARCHING in {report.pdf_path} for terms: {[term.alias for term in terms]}')

    for i, chunk in enumerate(report.chunks_iter()):
        if count_all_words:
            out['n_words'] += len(re.findall(r"[^\W_]+", chunk, re.M))

        for term in terms:
            if re.search(term.regex, chunk, re.I):
                if return_chunk_indices and return_counts:
                    # need to store both counts and chunk indices separately
                    out[term.alias] += len(re.findall(term.regex, chunk, re.I))
                    out[term.alias + '_chunk_indices'].append(i)
                elif return_chunk_indices:
                    out[term.alias].append(i)
                else:  # only counts requested
                    out[term.alias] += len(re.findall(term.regex, chunk, re.I))

    if save:
        # only allow saving to default if output is in the expected format (only chunk indices)
        if return_chunk_indices and not return_counts and not count_all_words:
            # if only chunk indices are requested, save to the default location if save_as is not provided
            save_path = report._search_results_path(terms) if save_as is None else save_as
        else:
            if save_as is None:
                raise ValueError(f'Can only save search results to default location if only chunk indices are requested. Got save_as=None and {return_counts=}, {return_chunk_indices=}.')
            save_path = save_as
        
        if not overwrite and os.path.isfile(save_path):
            logger_regex.debug(f'SEARCH NOT SAVED. overwrite=False, file exists: {save_path}')
        else:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(out, f, indent=4)
            logger_regex.debug(f'SEARCH SAVED to {save_path}.')

    return out

def search_with_metadata(
    report: Report, 
    terms: TermCollection = TERMS_ALL, 
    *, 
    return_counts: bool = True,
    return_chunk_indices: bool = True,
    count_all_words: bool = True,
    norm:bool = False,
    norm_type: Literal['pages','chunks', 'words'] = 'words',
    save_as:str|None = None,
    overwrite: bool = False
) -> dict[str, int | list[int]| float]:
    '''
    Search the given text for the given terms, returning a dictionary of term counts or 
    term indices. Term counts can be normalised by the number of pages or words in the report.
    Additionally returns metadata about the report.

    Parameters
    ----------
    report : Report
        The report to search.
    terms : Term|TermCollection, optional
        The term(s) to search for, by default TERMS_ALL.
    norm : bool, optional
        Whether to normalise the counts by the number of pages or words (as specified by norm_type). 
        Defaults to False.
    return_counts : bool, optional
        Whether to return the total number of mentions of each term. Defaults to True.
    return_chunk_indices : bool, optional
        Whether to return the indices of the chunks where each term is found. Defaults to False.
    count_all_words : bool, optional
        Whether to count all words in the report. Defaults to True.
    norm_type : Literal['pages', 'words'], optional
        The type of normalisation to apply. If 'pages', divide the counts by the number of pages. 
        If 'words', divide the counts by the number of words. Only applied if norm is True. Defaults to 'words'.

    Returns
    -------
    dict[str, int | float]
        A dictionary mapping each term alias to its count (or normalised count) in the report, and chunk indices if requested. 
        Also includes metadata about the report and the normalisation applied.
    '''

    norm_type = norm_type.lower()
    if norm_type not in ['pages', 'chunks', 'words']:
        raise ValueError(f'Invalid norm_type: {norm_type}. Must be one of "pages", "chunks", or "words".')

    out = search(
        report, 
        terms, 
        return_counts=return_counts, 
        return_chunk_indices=return_chunk_indices, 
        count_all_words=count_all_words
    )

    out['n_pages'] = len(pymupdf.open(report.pdf_path))
    out['n_chunks'] = report.n_chunks

    # normalise if requested
    if norm:
        normed_by = 'n_' + norm_type
        if out[normed_by] == 0:
            raise ValueError(f'Cannot normalise by {norm_type}: report has 0 {norm_type}: {report.pdf_path}')

        out.update({term.alias: out[term.alias]/out[normed_by] for term in terms})
        out['normed_by'] = normed_by
    else:
        out['normed_by'] = None

    # add metadata
    out['company'] = report.company.folder_name
    out['report_type'] = report.type
    out['year'] = report.year

    if save_as is not None:
        if not overwrite and os.path.isfile(save_as):
            logger_regex.debug(f'SEARCH WITH METADATA NOT SAVED. overwrite=False, file exists: {save_as}')
        else:
            if not os.path.exists(os.path.dirname(save_as)):
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as, 'w') as f:
                json.dump(out, f, indent=4)
            logger_regex.debug(f'SEARCH WITH METADATA SAVED to {save_as}.')

    return out

def pipeline(
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
                search_with_metadata(
                    report, terms, norm=True, norm_type=norm_type,
                    return_counts=True, return_chunk_indices=True, count_all_words=True
                )
            )
            results_unnormed.append(
                search_with_metadata(
                    report, terms, norm=False, 
                    return_counts=True, return_chunk_indices=True, count_all_words=True
                )
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


REGEX_BASELINES = {
    'Oil': r'\boil\b|petroleum|\bcrude\b',
    'Natural gas': r'\bgas\b|\bLNG\b|\bNGLs?\b',
    'Nature': r'nature|forest|offset|carbon\s+credit',
    'EVs': r'\bEVs?\b|electric\s+vehicle|charg(ing|e)',
    'Storage': r'\bbatter(y|ies)\b|\bstorage\b',
    'CCS': r'carbon\s+capture|\bCCS\b|\bCCUS\b|sequestrat|\bCO2\b',
}

class LabellingSession:
    '''
    A class for labelling chunks by the terms they mention, along with the type of mention.
    Used to validate regex patterns for precision and recall, 
    and to produce a labelled dataset for training/testing ML classifiers.

    The class is designed for use in a marimo notebook. At a minimimum, it requires two cells: 
    one to create the session and buttons, and one to display the current chunk and controls:

    **Cell 1:**
    ```
    session = LabellingSession(TERMS_SIMPLE, sample_path, log_path)
    get_pos, set_pos = mo.state(0)
    get_err, set_err = mo.state(None)
    buttons = session.buttons(set_pos, set_err)
    ```
    **Cell 2:**
    ```
    session.view(get_pos(), get_err(), buttons)
    ```

    Labels are recorded in a jsonl log, allowing labelling sessions to be interrupted and resumed.
    Duplicate entries are allowed - only the latest entry for a given chunk is loaded when the session is created, 
    allowing chunks to be re-labeled.

    
    Parameters
    ----------
    terms : TermCollection
        The collection whose regexes are being validated.
    sample_path : str
        The csv holding the sample. Read if it exists, otherwise drawn and written.
    log_path : str
        The jsonl log of labels. Created on the first label if absent.
    n_precision : int
        Regex hits drawn per term, for estimating precision. Default is 25.
    n_recall : int
        Baseline-only chunks drawn per term that has a baseline regex, for estimating recall.
        Default is 40.
    baseline_regexes : dict[str, str] | None
        Overly-broad regex patterns in the form alias: pattern. Used to identify potential false negatives.
        Default is `REGEX_BASELINES`.
    seed : int
        Random seed for sampling and shuffling. Default is 0.
    redraw : bool
        Whether to draw a new sample even when `sample_path` exists. Default is False.
    '''

    MENTION_TYPES = ('primary', 'comparison', 'table', 'boilerplate', 'uninterpretable')
    MENTION_RULES = {
        'primary': 'the chunk discusses the technology in its own right',
        'comparison': 'the term appears only as a benchmark for a different technology',
        'table': 'the mention is a row or cell of a quantitative table',
        'boilerplate': 'a glossary entry, definition, or other standing text',
        'uninterpretable': 'the term is present but the chunk lacks the context to read it, '
                           'such as a table stripped of its headers',
    }
    BASELINE_COLOUR = '#bfe3ff'

    # key: (button label, kind, keyboard shortcut, label, mention type)
    _LABEL_BUTTONS = {
        'primary': ('Match: primary', 'success', 'Ctrl-j', 1, 'primary'),
        'comparison': ('Match: comparison', 'success', 'Ctrl-k', 1, 'comparison'),
        'table': ('Match: table', 'success', 'Ctrl-l', 1, 'table'),
        'boilerplate': ('Match: boilerplate', 'success', 'Ctrl-u', 1, 'boilerplate'),
        'uninterpretable': ('Match: uninterpretable', 'warn', 'Ctrl-y', 1, 'uninterpretable'),
        'miss': ('Miss', 'danger', 'Ctrl-h', 0, None),
    }

    def __init__(self, terms, sample_path, log_path, *, n_precision=25, n_recall=40,
                 baseline_regexes=None, seed=0, redraw=False):

        self.terms = terms
        self.sample_path = sample_path
        self.log_path = log_path
        self.n_precision = n_precision
        self.n_recall = n_recall
        self.baseline_regexes = REGEX_BASELINES if baseline_regexes is None else baseline_regexes
        self.seed = seed

        if redraw or not os.path.isfile(resolve_path(sample_path)):
            self.sample = self.draw()
        else:
            self.sample = pd.read_csv(resolve_path(sample_path))
        # rename probe -> baseline for backwards compatibility
        if 'probe_only' in self.sample.columns and 'baseline_only' not in self.sample.columns:
            self.sample = self.sample.rename(columns={'probe_only': 'baseline_only'})
        self._keys = [self.label_key(r.term_alias, r.report_name, int(r.chunk_index))
                      for r in self.sample.itertuples()]

    def __len__(self):
        return len(self.sample)

    # ------------------------------------------------------------------ #
    #                           sampling
    # ------------------------------------------------------------------ #

    def draw(self):
        '''
        Draw the samples for labelling and write them to `sample_path`.
        Samples are drawn in two passes: first, `n_precision` regex hits for every term, then
        `n_recall` baseline-only chunks for every term with a baseline regex. 
        For a given seed, raising either count extends the sample rather than redrawing it allowing
        samples to be built up over time.
        '''
        precision = self.sample_for_labelling(
            self.terms, n_per_term=self.n_precision, hit_fraction=1.0, seed=self.seed)
        recall = self.sample_for_labelling(
            self.terms, hit_fraction=0.0, seed=self.seed, baseline_regexes=self.baseline_regexes,
            n_per_term={alias: self.n_recall for alias in self.baseline_regexes})
        df = pd.concat([precision, recall], ignore_index=True)
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        df.to_csv(resolve_path(self.sample_path), index=False)
        return df

    @staticmethod
    def sample_for_labelling(
            terms=TERMS_ALL, *,
            n_per_term=12,
            hit_fraction=0.5,
            miss_pool='same_report',
            baseline_regexes=None,
            seed=0,
            companies=None,
            report_types=None,
            save_as=None,
    ):
        '''
        Draw a stratified sample of chunks for hand-coding, to estimate regex precision and recall.

        The returned frame carries an empty `label` column for the coder to fill with 1 (the chunk
        does discuss the term) or 0 (it does not).

        Parameters
        ----------
        terms : TermCollection
            The term collection to sample for. Default is `TERMS_ALL`.
        n_per_term : int | dict[str, int]
            Total chunks to draw per term. If an int, the same number is drawn for each term.
            If a dict, it maps term alias to a count, and terms absent from it are not sampled.
            Default is 12.
        hit_fraction : float
            Proportion of each term's sample drawn from matching chunks. Default is 0.5.
        miss_pool : {"same_report", "random"}
            How non-matching chunks are drawn. "same_report" draws them from the reports in which
            the term matched, and "random" from any report. Default is "same_report".
        baseline_regexes : dict[str, str] | None
            Regexes, keyed by term alias, used as a baseline. For a term with a baseline, the
            non-matching chunks are drawn from those the baseline matched but the term's own regex
            did not. For a term without one, they are drawn from those the term's regex did not
            match.
        seed : int
            Seed for the sampling RNG. Default is 0.
        companies : CompanyCollection | Company | None
            The companies to include. If None, all companies are included.
        report_types : list[str] | None
            The report types to include. If None, all report types are included.
        save_as : str | None
            Path to write the sample to as a csv. If None, the frame is returned without being saved.

        Returns
        -------
        pd.DataFrame
            Columns: term_alias, company, report_type, year, report_name, chunk_index, regex_hit,
            baseline_only, matched_text, chunk, label.

        Raises
        ------
        ValueError
            If `miss_pool` is not "same_report" or "random".
        '''

        if miss_pool not in ('same_report', 'random'):
            raise ValueError(f'miss_pool must be "same_report" or "random". Got {miss_pool}')

        corpus = []
        for report in co_files('Report', companies=companies, report_types=report_types):
            search_results = report.search_results(terms)
            if search_results is None:
                continue
            corpus.append((report, report.chunks, search_results))

        baseline_regexes = baseline_regexes or {}
        rows = []

        for term in terms:
            n_term = n_per_term.get(term.alias, 0) if isinstance(n_per_term, dict) else n_per_term
            if n_term <= 0:
                continue
            n_hits_wanted = int(round(n_term * hit_fraction))
            baseline = re.compile(baseline_regexes[term.alias], re.I) if term.alias in baseline_regexes else None

            hits, misses = [], []
            for report, chunks, search_results in corpus:
                hit_idx = set(search_results.get(term.alias, []))
                if baseline is None and miss_pool == 'same_report' and not hit_idx:
                    continue
                for i, chunk in enumerate(chunks):
                    entry = (report, i, chunk)
                    if i in hit_idx:
                        hits.append(entry)
                    elif baseline is None or baseline.search(chunk):
                        misses.append(entry)

            for pool, is_hit, n_wanted in ((hits, True, n_hits_wanted),
                                           (misses, False, n_term - n_hits_wanted)):
                # shuffle then slice, so that raising n_per_term adds chunks rather than changing them
                pool = list(pool)
                random.Random(f'{seed}|{term.alias}|{is_hit}').shuffle(pool)
                for report, i, chunk in pool[:n_wanted]:
                    match = re.search(term.regex, chunk, re.I)
                    rows.append({
                        'term_alias': term.alias,
                        'company': report.company.folder_name,
                        'report_type': report.report_type,
                        'year': int(report.year),
                        'report_name': report.report_name,
                        'chunk_index': i,
                        'regex_hit': is_hit,
                        'baseline_only': bool(baseline is not None and not is_hit),
                        'matched_text': match.group(0) if match else '',
                        'chunk': chunk,
                        'label': '',
                    })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        logger_regex.debug(f'LABELLING SAMPLE: {len(df)} chunks, terms="{terms}", miss_pool={miss_pool}')

        if save_as is not None:
            df.to_csv(save_as, index=False)
            logger_regex.debug(f'LABELLING SAMPLE SAVED to {save_as}.')
        return df

    # ------------------------------------------------------------------ #
    #                            label log
    # ------------------------------------------------------------------ #

    @staticmethod
    def label_key(term_alias, report_name, chunk_index):
        '''Identify one labelling decision, so a log entry survives the sample being redrawn.'''
        return f'{term_alias}|{report_name}|{chunk_index}'

    def append_label(self, term_alias, report_name, chunk_index, label, mention_type=None):
        '''
        Append one labelling decision to the log.

        The log is append-only, allowing sessions to be interrupted without losing progress.
        Duplicate entries for the same chunk are allowed, and `load_labels` keeps the latest.

        Parameters
        ----------
        term_alias : str
            The alias of the term being labelled.
        report_name : str
            The name of the report the chunk belongs to.
        chunk_index : int
            The index of the chunk within that report.
        label : int
            1 if the chunk discusses the term, 0 if it does not.
        mention_type : str | None
            How the term is mentioned, for chunks labelled 1. One of `MENTION_TYPES`, described in
            `MENTION_RULES`. Recorded for sensitivity analysis of attention share, and does not
            affect precision.

        Raises
        ------
        ValueError
            If `mention_type` is not one of `MENTION_TYPES` or None.
        '''

        if mention_type is not None and mention_type not in self.MENTION_TYPES:
            raise ValueError(f'mention_type must be one of {self.MENTION_TYPES} or None. Got {mention_type}')
        path = resolve_path(self.log_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {
            'key': self.label_key(term_alias, report_name, chunk_index),
            'term_alias': term_alias,
            'report_name': report_name,
            'chunk_index': int(chunk_index),
            'label': int(label),
            'mention_type': mention_type,
            'labelled_at': time.time(),
        }
        with open(path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def load_labels(self):
        '''
        Read the log, keyed by `label_key`. Keeps only the latest label for each chunk.
        Returns an empty dict if the log does not exist.

        Returns
        -------
        dict[str, dict]
            A dict mapping `label_key` to the corresponding log entry, with keys `key`,
            `term_alias`, `report_name`, `chunk_index`, `label`, `mention_type` and `labelled_at`.
        '''

        path = resolve_path(self.log_path)
        if not os.path.isfile(path):
            return {}
        labels = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger_regex.warning(f'LABEL LOG: skipping unreadable line in {path}.')
                    continue
                held = labels.get(entry['key'])
                if held is None or entry.get('labelled_at', 0) >= held.get('labelled_at', 0):
                    labels[entry['key']] = entry
        return labels

    def apply_labels(self, df=None):
        '''
        Fill the `label` and `mention_type` columns of the given sample from the log. 
        If a chunk has no entry in the log, it will have an empty label.
        Returns a copy.
        
        Parameters
        ----------
        df : pd.DataFrame | None
            The sample to fill. If None, the session's own sample is used.
        
        Returns
        -------
        pd.DataFrame
            A copy of df with `label` and `mention_type` columns filled from the log.
        '''
        df = (self.sample if df is None else df).copy()
        labels = self.load_labels()
        entries = [labels.get(self.label_key(r.term_alias, r.report_name, int(r.chunk_index)))
                   for r in df.itertuples()]
        df['label'] = [e['label'] if e else '' for e in entries]
        df['mention_type'] = [(e.get('mention_type') or '') if e else '' for e in entries]
        return df

    def scores(self):
        '''
        Precision and false-negative rate per term, from the labels logged so far. `precision` is
        the binary regex validity. The `MENTION_TYPES` columns give the share of true matches of
        each kind, for sensitivity analysis of attention share. Empty if nothing is labelled.
        '''
        labelled = self.apply_labels()
        labelled = labelled[labelled.label != '']
        if not len(labelled):
            return pd.DataFrame()

        def per_term(d):
            hits = d[d.regex_hit == True]
            true = hits[hits.label.astype(int) == 1]
            out = {
                'n_hits_labelled': len(hits),
                'precision': hits.label.astype(int).mean() if len(hits) else float('nan'),
                'n_misses_labelled': int((d.regex_hit == False).sum()),
                'false_neg_rate': d.loc[d.regex_hit == False, 'label'].astype(int).mean(),
            }
            for t in self.MENTION_TYPES:
                out[t] = (true.mention_type == t).mean() if len(true) else float('nan')
            return pd.Series(out)

        return labelled.groupby('term_alias').apply(per_term, include_groups=False).round(3)

    # ------------------------------------------------------------------ #
    #                       marimo interface
    # ------------------------------------------------------------------ #

    def _record(self, pos, label, mention_type, set_err):
        '''Log a label for the chunk at `pos` and return the next position. Holds position on failure.'''
        if pos < len(self):
            row = self.sample.iloc[pos]
            try:
                self.append_label(row.term_alias, row.report_name, int(row.chunk_index), label, mention_type)
            except Exception as e:
                set_err(f'{type(e).__name__}: {e}')
                return pos
        set_err(None)
        return min(pos + 1, len(self))

    def _next_unlabelled(self, pos):
        '''The next position after `pos` with no entry in the log, wrapping round the end.'''
        labels = self.load_labels()
        for offset in range(1, len(self) + 1):
            j = (pos + offset) % len(self)
            if self._keys[j] not in labels:
                return j
        return len(self)

    def buttons(self, set_pos, set_err, *, shortcuts=True):
        '''
        Build the labelling buttons. Must be called in a cell that does not read the position
        getter, and the result held in a global and passed to `view`:
        ```
        set_pos, get_pos = mo.state(0)
        set_err, get_err = mo.state(None)
        buttons = session.buttons(set_pos, set_err)
        ```

        Parameters
        ----------
        set_pos : callable
            The setter of a `mo.state` holding the current position in the sample.
        set_err : callable
            The setter of a `mo.state` holding the last error, or None.
        shortcuts : bool
            Whether to bind keyboard shortcuts to the label buttons. Default is True.

        Returns
        -------
        dict[str, mo.ui.button]
            The buttons, keyed "primary", "comparison", "table", "boilerplate",
            "uninterpretable", "miss", "skip", "back" and "next_unlabelled".
        '''
        import marimo as mo
        # setters take an updater rather than a value, so this cell never reads the getter: a
        # cell that consumes the state it sets is re-run on every click, recreating the buttons
        out = {}
        for key, (text, kind, shortcut, label, mention_type) in self._LABEL_BUTTONS.items():
            out[key] = mo.ui.button(
                label=text, kind=kind, keyboard_shortcut=shortcut if shortcuts else None,
                on_click=lambda _, label=label, mention_type=mention_type: set_pos(
                    lambda i: self._record(i, label, mention_type, set_err)))
        out['skip'] = mo.ui.button(label='Skip', on_click=lambda _: set_pos(lambda i: min(i + 1, len(self))))
        out['back'] = mo.ui.button(label='Back', on_click=lambda _: set_pos(lambda i: max(i - 1, 0)))
        out['next_unlabelled'] = mo.ui.button(
            label='Next unlabelled', keyboard_shortcut='Ctrl-g' if shortcuts else None,
            on_click=lambda _: set_pos(self._next_unlabelled))
        return out

    def _highlight(self, row):
        '''
        The chunk with its regex matches (either the term regex or the baseline regex) highlighted. 
        Baseline chunks are highlighted in a different colour (BASELINE_COLOUR) to keep them distinct 
        from real matches.

        Parameters
        ----------
        row : pd.Series
            A row of the sample, with columns term_alias, report_name, chunk_index, regex_hit, baseline_only, matched_text, chunk, label.
        
        Returns
        -------
        str
            The chunk with its regex matches highlighted via HTML <mark> tags.
        '''
        chunk = str(row.chunk)
        baseline = self.baseline_regexes.get(row.term_alias)
        if row.get('baseline_only') and baseline:
            return re.sub(baseline, lambda m: f'<mark style="background:{self.BASELINE_COLOUR}">{m.group(0)}</mark>',
                          chunk, flags=re.I)
        return re.sub(self.terms[row.term_alias].regex, lambda m: f'<mark>{m.group(0)}</mark>', chunk, flags=re.I)

    def view(self, pos, err, buttons):
        '''
        View the current chunk and controls for labelling it.
        Must be called in a cell that reads the position getter to allow it to re-render on each click:
        ```
        session.view(get_pos(), get_err(), buttons)
        ```

        Parameters
        ----------
        pos : int
            The current position in the sample.
        err : str | None
            The last error from `buttons`, shown as a banner if not None.
        buttons : dict[str, mo.ui.button]
            As returned by `buttons`.

        Returns
        -------
        mo.Html
        '''
        import marimo as mo
        labels = self.load_labels()
        n = len(self)
        n_in_sample = sum(1 for k in self._keys if k in labels)

        controls = mo.vstack([
            mo.hstack([buttons[k] for k in self._LABEL_BUTTONS], justify='start', gap=0.5),
            mo.hstack([buttons[k] for k in ('skip', 'back', 'next_unlabelled')], justify='start', gap=0.5),
        ], gap=0.5)
        rules = mo.md(' &nbsp;·&nbsp; '.join(f'**{t}** {r}' for t, r in self.MENTION_RULES.items()))

        if pos >= n:
            body = mo.md(f'### Sample complete\n\n{len(labels)} chunks labelled. Raise `n_precision` '
                         'or `n_recall` and redraw to extend the sample.')
        else:
            row = self.sample.iloc[pos]
            if row.get('baseline_only'):
                pool = f'baseline-only candidate (<mark style="background:{self.BASELINE_COLOUR}">baseline hit</mark>, no term match)'
            elif row.regex_hit:
                pool = '<mark>term regex hit</mark>'
            else:
                pool = 'non-match'
            entry = labels.get(self._keys[pos])
            if entry is None:
                state = '**unlabelled**'
            elif entry['label'] == 0:
                state = 'labelled **miss**'
            else:
                state = f'labelled **match: {entry.get("mention_type") or "untyped"}**'
            body = mo.vstack([
                mo.md(f'# {row.term_alias}'),
                mo.md(f'`{row.report_name}` &nbsp;|&nbsp; chunk {row.chunk_index} &nbsp;|&nbsp; {pool} &nbsp;|&nbsp; {state}'),
                mo.md(f'**Does this chunk discuss {row.term_alias.lower()}?**'),
                controls,
                rules,
                mo.md('---'),
                mo.callout(mo.md(self._highlight(row)), kind='neutral'),
            ])

        frac = pos / n if n else 0
        bar = '█' * int(frac * 40) + '░' * (40 - int(frac * 40))
        banner = [mo.callout(mo.md(
            f'**Label not saved** - {err}\n\nIf this mentions `mention_type`, the kernel is running an '
            'older `decarb.regex`; restart it. Position has been held so nothing is skipped.'
        ), kind='danger')] if err else []

        return mo.vstack(banner + [
            mo.md(f'`{bar}` position **{pos} / {n}** &nbsp;|&nbsp; **{n_in_sample}** of this sample '
                  f'labelled, **{n - n_in_sample}** still open &nbsp;|&nbsp; **{len(labels)}** labelled in total'),
            body,
        ])