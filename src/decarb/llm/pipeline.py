import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sys
from typing import Iterable, Any, Literal
import json
from tqdm import tqdm

from decarb.llm.rag import search_chunks, pdf_to_md, chunk_md
from decarb.llm.prompt import (
    construct_prompt, create_openai_jsonl, jsonl_portions, 
    submit_openai_batch, check_batch_status, retrieve_batch_results,
    DEFAULT_MODEL,
    STATUS_WORKING, STATUS_FAILING, 
    STATUS_COMPLETED, STATUS_FAILED,
    STATUS_TERMINAL,  STATUS_IN_QUEUE, STATUS_NOT_IN_QUEUE
)
from decarb.companies import Company, COS, CompanyCollection, co_files, file_metadata
from decarb.util import REPORT_TYPES, DATA_DIR, TERMS_DIR, OUT_DIR, logger_llm, COMPANIES_DIR
from decarb.terms import Term, TermCollection, TERMS_ALL

def stage_convert(cos: Company|CompanyCollection = COS, report_types: Iterable[str]|str = REPORT_TYPES,
                  num_workers: int = 2,
                  overwrite: bool = False,
                  progress_bar: bool = True):
    '''
    Convert all company pdfs to markdown.

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
    overwrite : bool
        Whether to overwrite existing markdown files. Defaults to False.
    progress_bar : bool
        Whether to show a progress bar during conversion. Defaults to True.
    '''
    if isinstance(cos, Company):
        cos = [cos]
    if isinstance(report_types, str):
        report_types = [report_types]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for report in co_files(yield_level='Report', companies=cos, report_types=report_types):
            if report.pdf_path is None:
                logger_llm.warning(f'PDF NOT FOUND: {report.report_name}. Skipping pdf->md conversion. Full path: {report._pdf_path}')
                continue
            futures.append(executor.submit(pdf_to_md, report=report, save=True, overwrite=overwrite))
        
        with tqdm(
            total=len(futures), desc="converting pdfs to md", disable=not progress_bar, 
            bar_format='{desc}: {bar} {n}/{total}', file=sys.stdout) as pbar:
            for future in as_completed(futures):
                future.result()  # discard result, but raise any exceptions
                sys.stdout.flush()  # ensure tqdm output is printed immediately
                sys.stderr.flush()  # mupdf loves to throw to stderr, which obscures tqdm output
                pbar.update(1)
    
    if not progress_bar:
        print('conversion complete')
    

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


def stage_search(terms: TermCollection=TERMS_ALL, *, overwrite: bool = False, progress_bar: bool = True):
    '''
    Search the markdown files created in stage_convert for the given terms, and save the results as csvs.

    Parameters
    ----------
    terms : TermCollection
        The collection of terms to search for.
    '''
    # get total by counting the number of md files to be searched, for progress bar

    total = sum([report.n_chunks for report in co_files(yield_level='Report')])

    with tqdm(total=total, desc="Processing report chunks", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for report in co_files(yield_level='Report'):
            if report.n_chunks == 0:
                logger_llm.warning(f'NO CHUNKS FOUND for {report.report_name}. Skipping search. Full path: {report._chunks_dir}')
                continue

            search_chunks(
                report=report, terms=terms,
                save=True,
                overwrite=overwrite
            )
            pbar.update(report.n_chunks)
    
    if not progress_bar:
        print('search complete')


def stage_draft(terms: TermCollection=TERMS_ALL, *, 
                overlap_strategy: Literal["before", "after", "both", ]|None = 'both',
                overlap_amount: float = 0.2,
                overwrite: bool = False,
                progress_bar: bool = True):
    '''
    Use the markdown files and search results from the previous stages to draft LLM prompts for each article.

    Parameters
    ----------
    terms : TermCollection
        The collection of terms to draft prompts for.
    overwrite : bool
        Whether to overwrite existing prompt files. Defaults to False.
    progress_bar : bool
        Whether to display a progress bar. Defaults to True.
    '''

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
                
                term = terms[term_alias]
                logger_llm.debug(f'CONSTRUCTING PROMPT for {report:full} - {term_alias}. Saving to {report.prompt_path(terms, term)}')
                construct_prompt(
                    report=report, term=term,
                    overlap_strategy=overlap_strategy, overlap_amount=overlap_amount,
                    save=True, overwrite=overwrite
                )
                pbar.update(1)

    if not progress_bar:
        print('prompts drafted')
        

def stage_submit(model: str = DEFAULT_MODEL, *,
                 portion_strategy: Literal['max_tokens', 'max_prompts', 'split_into_n'] = 'max_tokens',
                 max_tokens: int = 1_000_000, max_prompts: int = 100, split_into_n: int = 10,
                 encoding: str = 'o200k_base',
                 companies: Company|CompanyCollection = None, 
                 terms: TermCollection|None = None,
                 overwrite: bool = False,
                 progress_bar: bool = True,
                 wait_interval: int = 120):
    '''
    Submit all prompts created in stage_draft for analysis, and save the batch details as json files.

    Parameters
    ----------
    model : str
        The name of the LLM model to use for analysis. Defaults to "gpt-5.1".
    portion_strategy : str
        The strategy to use for splitting the prompts into batches. Defaults to 'max_tokens'.
    max_tokens : int
        The maximum number of tokens to include in each batch portion submitted to the LLM. Defaults
        to 1,000,000.
    max_prompts : int
        The maximum number of prompts to include in each batch portion submitted to the LLM. Defaults
        to 100.
    split_into_n : int
        The number of batches to split the prompts into. Defaults to 10.
    encoding : str
        The name of the token encoding to use for counting tokens in the prompts. Defaults to 'o200k_base'.
    companies : Company, CompanyCollection or None
        The company or companies to include in the analysis. If None, includes all companies for which prompts have been drafted. Defaults to None.
    terms : TermCollection or None
        The collection of terms to include in the analysis. If None, includes all terms for which prompts have been drafted. Defaults to None.
    overwrite : bool
        Whether to overwrite existing analysis results. Defaults to False.
    progress_bar : bool
        Whether to display a progress bar during the submission process. Defaults to True.
    wait_interval : int
        The number of seconds to wait between checking the status of submitted batches. Defaults to 120 seconds.
    '''
    # collect all prompt file paths. assuming all text files are to be analysed.
    file_paths = []
    for report in co_files(yield_level='Report', companies=companies):
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
                if prompt_file.endswith('.txt'):
                    file_paths.append(report.prompt_path(term_coll_folder, prompt_file))
    logger_llm.info(f'COLLECTED PROMPTS: terms={terms} {len(file_paths)} prompt files found in .{len(list(co_files(yield_level="Report", companies=companies)))} reports.')

    # create jsonl for openai batch
    jsonl_path = create_openai_jsonl(
        prompt_paths=file_paths,
        model=model,
        save=True,
        overwrite=overwrite,
        return_type='path'
    )

    # submit portions of jsonl to openai
    terms_coll =  TermCollection.from_folder(os.path.dirname(os.path.dirname(file_paths[0])))  # structure asserted in create_openai_jsonl
    save_dir = terms_coll.batches_folder
    info_path = os.path.join(terms_coll.prompts_folder, 
                             'portions-info-' + os.path.basename(jsonl_path).replace('.jsonl', '.json'))

    # check if we have any batches currently running
    batch_statuses = check_batch_status(batch_id = None, on_miss='create')
    running_batches = [batch['id'] for batch in batch_statuses if batch['status'] in STATUS_IN_QUEUE]
    if running_batches:
        if not progress_bar:
            print(f'Found {len(running_batches)} running batches. Retrieving results for these batches before submitting new ones.')
        for batch_id in tqdm(running_batches, desc="Retrieving results for running batches", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}'):
            retrieve_batch_results(
                batch_id=batch_id, 
                save=True, 
                if_incomplete='wait', wait_interval=wait_interval,
                if_failed='ignore'
            )

    # construct portion info dict to save to file, so we can resume if interrupted
    if portion_strategy == 'max_tokens':
        portion_strategy_param = max_tokens
    elif portion_strategy == 'max_prompts':
        portion_strategy_param = max_prompts
    elif portion_strategy == 'split_into_n':
        portion_strategy_param = split_into_n
    portion_info = {
        'metadata': {
            'parent': jsonl_path, 
            'strategy': portion_strategy, 'strategy_param': portion_strategy_param, 
            'encoding': encoding
        },
        'batches': {}
    }

    total_prompts = sum(1 for _ in open(jsonl_path, 'r'))
    if not progress_bar:
        print(f'Submitting {total_prompts} prompts to {model} in portions using strategy {portion_strategy} with parameter {portion_strategy_param}.')
    with tqdm(total=total_prompts, desc=f"Submitting prompts to {model} and awaiting results", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for portion in jsonl_portions(
            jsonl_path=jsonl_path, 
            strategy = portion_strategy, max_tokens=max_tokens, max_prompts=max_prompts, split_into_n=split_into_n, 
            encoding_name=encoding,
            #save = True, overwrite=overwrite
        ):
            batch_id = submit_openai_batch(
                jsonl=portion.portion,
                save_dir=save_dir,
            )
            portion_info['batches'][batch_id] = {
                'start_index': portion.start_index, 'end_index': portion.end_index,
                'token_count': portion.token_count, 'prompt_count': len(portion.portion),
                'completed': False, 'status': 'submitted'
            }
            # write the portion info to file after each batch submission, so we can resume if interrupted
            with open(info_path, 'w') as f:
                json.dump(portion_info, f, indent=4)

            # retrieve before submitting another batch - assuming we're maximising our enqueued token limits, so need to wait for the batch to complete before submitting another
            batch_res = retrieve_batch_results(
                batch_id=batch_id, 
                save=True, 
                if_incomplete='wait',  # wait for batch to complete before submitting another
                if_failed='ignore',
                wait_interval=wait_interval
            )
            if batch_res is None:
                # batch failed
                portion_info['batches'][batch_id]['completed'] = False
                portion_info['batches'][batch_id]['status'] = 'failed'
            else:
                portion_info['batches'][batch_id]['completed'] = True
                portion_info['batches'][batch_id]['status'] = 'completed'
            with open(info_path, 'w') as f:
                json.dump(portion_info, f, indent=4)

            pbar.update(len(portion.portion))

    with open(info_path, 'w') as f:
        json.dump(portion_info, f, indent=4)
    print(f'Batch results processed and retrieved for {len(portion_info["batches"])} batches.')

def stage_gather_ids(status: str = '', *, companies: Company|CompanyCollection = None, terms: TermCollection|None = None):
    '''
    Gather the batch IDs of all submitted batches for the given companies and terms, and return them as a list.

    Parameters
    ----------
    companies : Company, CompanyCollection or None
        The company or companies to include in the analysis. If None, includes all companies for which prompts have been drafted. Defaults to None.
    terms : TermCollection or None
        The collection of terms to include in the analysis. If None, includes all terms for which prompts have been drafted. Defaults to None.
    '''
    batch_ids = []
    for report in co_files(yield_level='Report', companies=companies):
        if isinstance(terms, TermCollection):
            term_coll_folder_names = [terms.folder_name]
        elif terms is None:
            term_coll_folder_names = report.term_coll_folder_names()
        else: 
            raise ValueError(f'Invalid terms argument: {terms}. Must be a TermCollection or None.')

        for term_coll_folder in term_coll_folder_names:
            info_path = os.path.join(report.prompts_dir(term_coll_folder), 'portions-info.json')
            if not os.path.exists(info_path):
                logger_llm.warning(f'Portions info file not found for {report:full} - {term_coll_folder}. Skipping.')
                continue
            with open(info_path, 'r') as f:
                portion_info = json.load(f)
                batch_ids.extend(list(portion_info['batches'].keys()))
    return batch_ids

def stage_retrieve(batch_ids: Iterable[str]|None = None, *, 
                   progress_bar: bool = True, verbose: bool = False):
    '''
    Retrieve the results of the submitted batches, saving the results as json files in the 
    output directory for the relevant term collection. If batches have not yet completed,
    the function will print the currrent status of the batch.

    Parameters
    ----------
    batch_ids : Iterable[str]
        The IDs of the batches to retrieve results for.
    progress_bar : bool
        Whether to display a progress bar. Defaults to True.
    verbose : bool
        Whether to print detailed information about the batch retrieval process. Defaults to False.
        If progress_bar is True, printing is limited to avoid cluttering the progress bar output.
    '''
    if batch_ids is None:
        # Retrieve all batch IDs from the portions info file
        batch_ids = []
        for root, dirs, files in os.walk(TERMS_DIR):
            for file in files:
                if file.startswith('portions-info-') and file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        info = json.load(f)
                        batch_ids.extend(list(info['batches'].keys()))
        if len(batch_ids) == 0:
            # try the batches folder
            for root, dirs, files in os.walk(TERMS_DIR):
                for dir in dirs:
                    if dir == 'batches':
                        for batch_file in os.listdir(os.path.join(root, dir)):
                            if batch_file.endswith('.json'):
                                batch_ids.append(batch_file.replace('.json', ''))

    n_working = 0
    n_stopped = 0
    n_stopping = 0
    n_completed = 0

    for batch_id in tqdm(batch_ids, disable = not progress_bar):
        while True:
            batch_info = check_batch_status(batch_id)
            status = batch_info.get('status')
            logger_llm.info(f'BATCH CHECKED {batch_id}: {status}. ')
            if status in STATUS_COMPLETED:
                n_completed += 1
                if not progress_bar and verbose:
                    print(f'batch completed - {batch_id} - status={status}. retrieving results.')
                retrieve_batch_results(batch_id, save=True)
                break
            elif status in STATUS_WORKING:
                n_working += 1
                if not progress_bar and verbose:
                    print(f'batch working - {batch_id} - status={status}.')
                break
            elif status in STATUS_FAILING:
                n_stopping += 1
                logger_llm.error(f'Batch {batch_id} {status}. skipping retrieval.')
                if not progress_bar and verbose:
                    print(f'batch stopping - {batch_id} - status={status}. skipping retrieval.')
                break
            elif status in STATUS_FAILED:
                n_stopped += 1
                logger_llm.error(f'Batch {batch_id} {status}. skipping retrieval.')
                if not progress_bar and verbose:
                    print(f'batch stopped - {batch_id} - status={status}. skipping retrieval.')
                break
    
    if verbose:
        print(f'retrieval complete. {n_completed} completed, {n_working} working, {n_stopped} stopped, {n_stopping} stopping / {len(batch_ids)} total.')

