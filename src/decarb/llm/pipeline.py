import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sys
from dotenv import load_dotenv
from openai import OpenAI
from typing import Iterable, Any, Literal
import json
from tqdm import tqdm

from .rag import search_chunks, pdf_to_md, chunk_md, chunks_list
from .prompt import construct_prompt
from ..companies import Company, COS, CompanyCollection, co_files, file_metadata
from ..util import REPORT_TYPES, DATA_DIR, OUT_DIR, logger_llm, COMPANIES_DIR
from ..regex.terms import Term, TermCollection, TERMS_ALL


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
        for pdf in co_files(yield_level='report_pdf', companies=cos, report_types=report_types):
            if pdf.endswith('.pdf'):
                futures.append(executor.submit(pdf_to_md, pdf_path=pdf, save=True, save_path=pdf.replace('.pdf', '.md'), overwrite=overwrite))
        
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
        for file in co_files(yield_level='report_md'):
            dir_to_save = file.replace(os.path.basename(file), 'chunks')
            futures.append(executor.submit(chunk_md, md=file, dir_to_save=dir_to_save, overwrite_chunks=overwrite, remove_original_md=True, **chunk_kwargs))
        
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

    total = sum([1 for _ in co_files(yield_level='chunk')])

    with tqdm(total=total, desc="Processing companies", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for chunks_dir in co_files(yield_level='chunk_folder'):

            save_dir = os.path.join(os.path.dirname(chunks_dir), terms.folder_name)
            os.makedirs(save_dir, exist_ok=True)

            search_chunks(
                chunks_dir, terms, 
                save_path=os.path.join(save_dir, f'search-results.json'), 
                overwrite=overwrite
            )
            pbar.update(sum(1 for f in os.listdir(chunks_dir) if f.endswith('.md')))
    
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
        for chunks_dir in co_files(yield_level='chunk_folder'):

            # extract company, report type, report name, and year from file path
            report_mdata = file_metadata(chunks_dir)
            co = report_mdata['company']
            report_type = report_mdata['report_type']
            report = report_mdata['report_name']
            year = report_mdata['year']

            # check year is a valid 4 digit no.
            if year is None or not (year.isdigit() and len(year) == 4):
                logger_llm.warning(f'INVALID PROMPT YEAR: "{year}" from -> "{report}". Skipping prompt construction for this report.')
                pbar.update(1)
                continue

            report_folder = os.path.dirname(chunks_dir)
            search_results_filepath = os.path.join(report_folder, terms.folder_name, 'search-results.json')
                    
            if not os.path.exists(search_results_filepath):
                logger_llm.warning(f'SEARCH RESULTS NOT FOUND for {co} - {report_type} - {report}. Skipping prompt construction.')
                pbar.update(len(terms))
                skip_counter += len(terms)
                if skip_counter > 10:
                    raise ValueError(f'More than 10 prompts skipped due to missing search results - try running stage_search. Last missing file: {search_results_filepath}')
                continue

            logger_llm.debug(f'PROCESSING: {co} - {report_type} - {search_results_filepath}')
            
            search_results = json.load(open(search_results_filepath))
            for term_alias, chunk_inds in search_results.items():
                if chunk_inds is None or len(chunk_inds) == 0:
                    logger_llm.debug(f'NO CHUNKS for {term_alias} in {co} - {report_type} - {report}. Skipping prompt construction for this term.')
                    pbar.update(1)
                    continue
                
                term = terms[term_alias]
                
                prompt_path = os.path.join(report_folder, terms.folder_name, 'prompts', term.prompt_fname)

                logger_llm.debug(f'CONSTRUCTING PROMPT for {co} - {report_type} - {report} - {term_alias}. Saving to {prompt_path}')
                if not overwrite and os.path.isfile(prompt_path):
                    logger_llm.debug(f'PROMPT NOT CREATED: {prompt_path} already exists and overwrite=False')
                    pbar.update(1)
                    continue
                
                construct_prompt(
                    company=co, report_type=report_type, year=year, term=term,
                    chunks=chunks_list(chunks_dir), chunk_inds=chunk_inds,
                    overlap_strategy=overlap_strategy, overlap_amount=overlap_amount,
                    save_path=prompt_path, overwrite=overwrite
                )
                pbar.update(1)

    if not progress_bar:
        print('prompts drafted')
        

def stage_analyse(model: str = "gpt-5.1", max_workers: int = 5):
    '''
    Run LLM analysis on all prompts created in stage_draft, and save the results as json files.

    Parameters
    ----------
    model : str
        The name of the LLM model to use for analysis. Defaults to "gpt-5.1".
    max_workers : int
        The maximum number of worker threads to use for parallel analysis. Defaults to 5.
    '''
    pass


