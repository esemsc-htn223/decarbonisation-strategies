import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sys
from typing import Iterable, Any, Literal
import json
from tqdm import tqdm

from decarb.regex import search, pdf_to_md, chunk_md, conversion_quality
from decarb.llm.prompt import (
    construct_prompt,
    Job,
    DEFAULT_MODEL,
    RESPONSE_SCHEMAS
)
from decarb.companies import Company, COMPANIES_ALL, CompanyCollection, co_files, REPORT_TYPES
from decarb.util import logger_llm
from decarb.terms import TermCollection, TERMS_ALL

def stage_convert(cos: Company|CompanyCollection = COMPANIES_ALL, report_types: Iterable[str]|str = REPORT_TYPES,
                  *,
                  num_workers: int = 2,
                  ocr_frac_no_text_threshold: float = 0.5,
                  overwrite: bool = False,
                  progress_bar: bool = True,
                  ):
    '''
    Convert all company pdfs to markdown. Returns the names of any reports that failed to convert.

    Parameters
    ----------
    cos : Company | CompanyCollection
        The company/companies to convert. Defaults to BP, Chevron, Exxon, Shell, and Total.
    report_types : Iterable[str]
        The report type(s) to convert, given as folder names to be found in each Company's data folder.
        Defaults to annual reports and sustainability reports.
    num_workers : int
        The number of worker processes to use for parallel conversion, using ProcessPoolExecutor. Defaults to 2. Note that more workers may speed up conversion, 
        but will also use more memory, as each worker loads the PDF into memory to convert it.
    ocr_frac_no_text_threshold : float
        The fraction of the PDF pages that must have no text layer for OCR to be used. Defaults to 0.5.
        Any PDF with more than this fraction of pages with no text layer will be converted using OCR, 
        which is slower but more reliable for scanned documents.
    overwrite : bool
        Whether to overwrite existing markdown files. Defaults to False.
    progress_bar : bool
        Whether to show a progress bar during conversion. Defaults to True.

    Returns
    -------
    list[str]
        The names of any reports that raised during conversion.
    '''
    if isinstance(report_types, str):
        report_types = [report_types]

    reports = []
    for report in co_files(yield_level='Report', companies=cos, report_types=report_types):
        if report.pdf_path is None:
            logger_llm.warning(f'PDF NOT FOUND: {report.report_name}. Skipping pdf->md conversion. Full path: {report._pdf_path}')
            continue
        if not overwrite and (os.path.isfile(report._md_path) or report.n_chunks > 0):
            logger_llm.debug(f'ALREADY CONVERTED: {report.report_name}. Skipping pdf->md conversion.')
            continue
        reports.append(report)

    if not reports:
        logger_llm.debug('NOTHING TO CONVERT: every report already has markdown.')
        if not progress_bar:
            print('nothing to convert')
        return []

    if not progress_bar:
        print('determining which reports need OCR')
    df_quality = conversion_quality(reports = reports, check_type='pre-conversion',
                                    progress_bar = progress_bar)

    # longest first to maximise job overlap when working in parallel
    reports.sort(key=lambda r: df_quality['n_pages'].get(r.report_name, 0), reverse=True)

    if not progress_bar:
        print('converting pdfs to markdown')
    failed = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for report in reports:
            frac_no_text = df_quality['frac_no_text_layer'].get(report.report_name)
            use_ocr = frac_no_text is not None and frac_no_text > ocr_frac_no_text_threshold
            futures[executor.submit(pdf_to_md, report=report, save=True, overwrite=overwrite, use_ocr=use_ocr)] = report

        with tqdm(
            total=len(futures), desc="converting pdfs to md", disable=not progress_bar,
            bar_format='{desc}: {bar} {n}/{total}', file=sys.stdout) as pbar:
            for future in as_completed(futures):
                report = futures[future]
                try:
                    future.result()  # discard result, but surface any exceptions
                except Exception as e:
                    failed.append(report.report_name)
                    logger_llm.error(f'CONVERSION FAILED: {report.report_name}. {type(e).__name__}: {e}')
                sys.stdout.flush()  # ensure tqdm output is printed immediately
                sys.stderr.flush()  # mupdf loves to throw to stderr, which obscures tqdm output
                pbar.update(1)

    if failed:
        logger_llm.error(f'CONVERSION FAILED for {len(failed)} of {len(futures)} reports: {failed}')

    if not progress_bar:
        if not failed:
            print('conversion complete')
        else:
            print(f'conversion complete, {len(failed)} of {len(futures)} failed: {failed}')
    return failed
    
def stage_chunk(*,
                num_workers: int = 5,
                overwrite: bool = False,
                progress_bar: bool = True,
                **chunk_kwargs: Any):
    '''
    Chunk the markdown files created in stage_convert, and save the chunks as separate markdown files.

    Parameters
    ----------
    num_workers : int
        The number of worker threads to use for parallel chunking, using ThreadPoolExecutor. Defaults to 5. 
        Note that more workers may speed up chunking, but will also use more memory, as each worker loads the markdown file into memory to chunk it.
    overwrite : bool
        Whether to overwrite existing chunk files. Defaults to False.
    progress_bar : bool
        Whether to show a progress bar during chunking. Defaults to True.
    chunk_kwargs : dict
        Additional keyword arguments to pass to the chunk_md function, such as chunk_size and chunk_overlap.
    '''

    # chunk in parallel using threads (no OCR, so not limited to processes)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for report in co_files(yield_level='Report'):
            if report.md_path is None:
                logger_llm.warning(f'MARKDOWN NOT FOUND: {report.report_name}. Skipping chunking. Full path: {report._md_path}')
                continue
            futures.append(executor.submit(chunk_md, report=report, overwrite_chunks=overwrite, save=True, remove_original_md=True, **chunk_kwargs))
        
        with tqdm(total=len(futures), desc="chunking md files", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
            for future in as_completed(futures):
                future.result()  # discard result, but raise any exceptions
                pbar.update(1)
    if not progress_bar:
        print('chunking complete')

def stage_search(terms: TermCollection=TERMS_ALL, 
                 *, 
                 overwrite: bool = False, 
                 progress_bar: bool = True):
    '''
    Search the markdown files created in stage_convert for the given terms, and save the results as csvs.

    Parameters
    ----------
    terms : TermCollection
        The collection of terms to search for.
    overwrite : bool
        Whether to overwrite existing search results. Defaults to False.
    progress_bar : bool
        Whether to display a progress bar. Defaults to True. If False, a message will be printed when the search is complete.
    '''
    # get total by counting the number of md files to be searched, for progress bar

    total = sum([report.n_chunks for report in co_files(yield_level='Report')])

    with tqdm(total=total, desc="Processing report chunks", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for report in co_files(yield_level='Report'):
            if report.n_chunks == 0:
                logger_llm.warning(f'NO CHUNKS FOUND for {report.report_name}. Skipping search. Full path: {report._chunks_dir}')
                continue

            search(
                report=report, terms=terms,
                return_counts=False, return_chunk_indices=True, 
                count_all_words=False,
                save=True,  # save to default location in report dir
                overwrite=overwrite
            )
            pbar.update(report.n_chunks)
    
    if not progress_bar:
        print('search complete')

def stage_draft(terms: TermCollection=TERMS_ALL, *, 
                schema_version: int = 1,
                overlap_strategy: Literal["before", "after", "both", ]|None = 'both',
                overlap_amount: float = 0.2,
                max_extracts: int|None = None,
                mask: bool = False,
                overwrite: bool = False,
                progress_bar: bool = True):
    '''
    Use the markdown files and search results from the previous stages to draft LLM prompts for each article.

    Parameters
    ----------
    terms : TermCollection
        The collection of terms to draft prompts for.
    schema_version : int
        The schema version to use for the prompts. Must be one of the available schema versions. Defaults to 1.
    overlap_strategy : ["before", "after", "both"] | None
        The strategy to use for overlapping chunks when drafting prompts. Defaults to 'both'.
    overlap_amount : float
        The fraction of the chunk to overlap when drafting prompts. Defaults to 0.2.
    max_extracts : int | None
        The maximum number of extracts to include in each prompt. If None, includes all extracts.
    mask : bool
        Whether to mask references to the company and year in the prompt. Defaults to False.
    overwrite : bool
        Whether to overwrite existing prompt files. If using a new schema, set to True. Defaults to False.
    progress_bar : bool
        Whether to display a progress bar. Defaults to True.
    '''

    if schema_version not in RESPONSE_SCHEMAS:
        raise ValueError(f'Invalid schema_version: {schema_version}. Must be one of {list(RESPONSE_SCHEMAS.keys())}')

    skip_counter = 0
    with tqdm(total=sum(1 for _ in co_files(yield_level='chunk_folder'))*len(terms), desc="Drafting prompts", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for report in co_files(yield_level='Report'):
                    
            if report.search_results_path(terms) is None:
                logger_llm.warning(f'SEARCH RESULTS NOT FOUND for {report:full}. Skipping prompt construction.')
                pbar.update(len(terms))
                skip_counter += len(terms)
                continue

            if report.n_chunks == 0:
                logger_llm.warning(f'NO CHUNKS FOUND for {report:full}. Skipping prompt construction. Full path: {report._chunks_dir}')
                pbar.update(len(terms))
                skip_counter += len(terms)
                continue

            if skip_counter > 10:
                raise ValueError(f'More than 10 prompts skipped due to missing chunks or search results - try running stage_chunk and/or stage_search. Last missing report dir: {report.dir}')

            logger_llm.debug(f'PROCESSING: {report.company} - {report.report_type} - {report.search_results_path(terms)}')
            
            search_results = json.load(open(report.search_results_path(terms)))
            for term_alias, chunk_inds in search_results.items():
                if chunk_inds is None or len(chunk_inds) == 0:
                    logger_llm.debug(f'NO CHUNKS for {term_alias} in {report:full}. Skipping prompt construction for this term.')
                    pbar.update(1)
                    continue
                
                if term_alias not in terms.aliases:
                    # the search results cover the parent collection, which may be wider than `terms`
                    pbar.update(1)
                    continue
                term = terms[term_alias]
                logger_llm.debug(f'CONSTRUCTING PROMPT for {report:full} - {term_alias}. Saving to {report._prompt_path(terms, term, schema_version)}')
                construct_prompt(
                    report=report, term=term,
                    schema_version=schema_version,
                    overlap_strategy=overlap_strategy, overlap_amount=overlap_amount,
                    max_extracts=max_extracts, mask=mask,
                    save=True, overwrite=overwrite
                )
                pbar.update(1)

    if not progress_bar:
        print('prompts drafted')
        
def stage_submit(model: str = DEFAULT_MODEL, *,
                 schema_version: int = 1,
                 portion_strategy: Literal['max_tokens', 'max_prompts', 'split_into_n'] = 'max_tokens',
                 max_tokens: int = 900_000, max_prompts: int = 100, split_into_n: int = 10,
                 encoding: str = 'o200k_base',
                 cos: Company|CompanyCollection = None,
                 terms: TermCollection|None = None,
                 enqueued_token_limit: int = 900_000,
                 max_attempts: int = 3,
                 overwrite: bool = False,
                 progress_bar: bool = True,
                 wait_interval: int = 120) -> Job:
    '''
    Submit all prompts created in stage_draft for analysis. Returns a Job object, which can be
    used to keep track of the run's status.

    Parameters
    ----------
    model : str
        The name of the LLM model to use for analysis. Defaults to "gpt-5.1".
    schema_version : int
        The response schema version to use for the prompts. Must be one of the available schema versions. Defaults to 1.
    portion_strategy : str
        The strategy to use for splitting the prompts into batches. Defaults to 'max_tokens'.
    max_tokens : int
        The maximum number of tokens to include in each batch portion submitted to the LLM. Defaults
        to 900,000.
    max_prompts : int
        The maximum number of prompts to include in each batch portion submitted to the LLM. Defaults
        to 100.
    split_into_n : int
        The number of batches to split the prompts into. Defaults to 10.
    encoding : str
        The name of the token encoding to use for counting tokens in the prompts. Defaults to 'o200k_base'.
    cos : Company, CompanyCollection or None
        The company or companies to include in the analysis. If None, includes all companies for which prompts have been drafted. Defaults to None.
    terms : TermCollection or None
        The collection of terms to include in the analysis. If None, includes all terms for which prompts have been drafted. Defaults to None.
    enqueued_token_limit : int
        The account's enqueued token limit for the model. Submission waits for headroom under this
        limit before sending each portion. Defaults to 900,000.
    max_attempts : int
        How many times to resubmit a portion rejected for exceeding the enqueued token limit. Defaults to 3.
    overwrite : bool
        Whether to rewrite the input jsonl if it already exists. Defaults to False.
    progress_bar : bool
        Whether to display a progress bar during the submission process. Defaults to True.
    wait_interval : int
        The number of seconds to wait between checking the status of submitted batches. Defaults to 120 seconds.

    Returns
    -------
    Job
        The submitted job.

    Raises
    ------
    ValueError
        If `terms` is neither a TermCollection nor None.
    ValueError
        If `schema_version` is not one of the available schema versions.
    '''

    if schema_version not in RESPONSE_SCHEMAS:
        raise ValueError(f'Invalid schema_version: {schema_version}. Must be one of {list(RESPONSE_SCHEMAS.keys())}')
    
    # collect all prompt file paths. assuming all text files are to be analysed.
    file_paths = []
    for report in co_files(yield_level='Report', companies=cos):
        if isinstance(terms, TermCollection):
            term_coll_folder_names = [terms.folder_name]
        elif terms is None:
            term_coll_folder_names = report.term_coll_folder_names()
        else:
            raise ValueError(f'Invalid terms argument: {terms}. Must be a TermCollection or None.')

        for term_coll_folder in term_coll_folder_names:
            if report.prompts_dir(term_coll_folder) is None:
                continue
            for prompt_file in os.listdir(report.prompts_dir(term_coll_folder)):
                if prompt_file.endswith(f'_v{schema_version}.txt'):
                    file_paths.append(report.prompt_path(term_coll_folder, prompt_file, schema_version))
    logger_llm.info(f'COLLECTED PROMPTS: terms={terms}, schema_version={schema_version}, {len(file_paths)} prompt files found in .{len(list(co_files(yield_level="Report", companies=cos)))} reports.')

    job = Job.from_prompts(file_paths, model=model, overwrite=overwrite)
    job.submit(
        portion_strategy=portion_strategy,
        portion_max_tokens=max_tokens, portion_max_prompts=max_prompts, portion_split_into_n=split_into_n,
        portion_encoding=encoding,
        enqueued_token_limit=enqueued_token_limit, max_attempts=max_attempts,
        overwrite=overwrite, progress_bar=progress_bar, wait_interval=wait_interval
    )
    return job

def stage_retrieve(jobs: Iterable[Job]|Job|None = None, *,
                   wait: bool = False,
                   progress_bar: bool = True, verbose: bool = False) -> list[Job]:
    '''
    Retrieve the results of submitted batches, saving them to the output directory. Refreshes and
    returns the Job objects it retrieved for.

    Parameters
    ----------
    jobs : Iterable[Job] | Job | None
        The jobs to retrieve results for. If None (default), retrieves for every job found under TERMS_DIR.
    wait : bool
        Whether to wait for batches that are still in progress. Defaults to False.
    progress_bar : bool
        Whether to display a progress bar. Defaults to True.
    verbose : bool
        Whether to print detailed information about the batch retrieval process. Defaults to False.

    Returns
    -------
    list[Job]
        The jobs that were refreshed.
    '''
    if jobs is None:
        jobs = Job.list_all()
    elif isinstance(jobs, Job):
        jobs = [jobs]
    else:
        jobs = list(jobs)

    totals = {}
    for job in tqdm(jobs, disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}', desc='retrieving job results'):
        job.refresh(progress_bar=False)
        retrieved = job.retrieve(wait=wait, progress_bar=False)
        for status, count in job.n_batch_statuses.items():
            totals[status] = totals.get(status, 0) + count
        if verbose and not progress_bar:
            print(f'{job.id}: status={job.status}, batches={job.n_batch_statuses}, '
                  f'retrieved={len(retrieved)}, missing={len(job.missing_prompts)}/{job.n_prompts}')

    if verbose:
        print(f'retrieval complete for {len(jobs)} job(s). batch statuses: {totals}')
    return jobs

def pipeline(
    cos: Company|CompanyCollection = COMPANIES_ALL,
    report_types: Iterable[str]|str = REPORT_TYPES,
    terms: TermCollection=TERMS_ALL,
    model: str = DEFAULT_MODEL,
    *,
    progress_bars: bool = True,
    convert_kwargs: dict = {},
    chunk_kwargs: dict = {},
    search_kwargs: dict = {},
    draft_kwargs: dict = {},
    submit_kwargs: dict = {},
):
    '''
    Run the full LLM pipeline up to Job submission for the given companies, report types, and terms.
    Returns the submitted Job object.

    Parameters
    ----------
    cos : Company | CompanyCollection
        The company/companies to process. Defaults to BP, Chevron, Exxon, Shell, and Total.
    report_types : Iterable[str] | str
        The report type(s) to process, given as folder names to be found in each Company's data folder.
        Defaults to annual reports and sustainability reports.
    terms : TermCollection
        The collection of terms to search for and draft prompts for. Defaults to TERMS_ALL.
    model : str
        The name of the LLM model to use for analysis. Defaults to "gpt-5.1".
    progress_bars : bool
        Whether to display progress bars for each stage. If 'progress_bar' is specified in any of the kwargs, 
        it will override this value for that stage. Defaults to True.
    convert_kwargs : dict
        Keyword arguments to pass to the stage_convert function. Defaults to an empty dictionary.
    chunk_kwargs : dict
        Keyword arguments to pass to the stage_chunk function. Defaults to an empty dictionary.
    search_kwargs : dict
        Keyword arguments to pass to the stage_search function. Defaults to an empty dictionary.
    draft_kwargs : dict
        Keyword arguments to pass to the stage_draft function. Defaults to an empty dictionary.
    submit_kwargs : dict
        Keyword arguments to pass to the stage_submit function. Defaults to an empty dictionary.

    Returns
    -------
    Job
        The submitted job from stage_submit.
    '''

    for kwargs in [convert_kwargs, chunk_kwargs, search_kwargs, draft_kwargs, submit_kwargs]:
        if not isinstance(kwargs, dict):
            raise ValueError(f'kwargs must be a dict, got {type(kwargs)}')
        if 'progress_bar' not in kwargs:
            kwargs['progress_bar'] = progress_bars
        if 'terms' in kwargs:
            raise ValueError(f'kwargs should not contain "terms" key, - set this as a parameter to the pipeline function. Got: {list(kwargs.keys())}')
        if 'cos' in kwargs:
            raise ValueError(f'kwargs should not contain "cos" key, - set this as a parameter to the pipeline function. Got: {list(kwargs.keys())}')
        if 'report_types' in kwargs:
            raise ValueError(f'kwargs should not contain "report_types" key, - set this as a parameter to the pipeline function. Got: {list(kwargs.keys())}')
        if 'model' in kwargs:
            raise ValueError(f'kwargs should not contain "model" key, - set this as a parameter to the pipeline function. Got: {list(kwargs.keys())}')

    failed_submissions = stage_convert(cos=cos, report_types=report_types, **convert_kwargs)
    if failed_submissions and convert_kwargs.get('progress_bar', progress_bars):
        print(f'{len(failed_submissions)} reports failed to convert from pdf to md. Check logs for details.')
    stage_chunk(**chunk_kwargs)
    stage_search(terms=terms, **search_kwargs)
    stage_draft(terms=terms, **draft_kwargs)
    return stage_submit(model=model, **submit_kwargs)