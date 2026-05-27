import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from typing import Iterable, Any, Literal
import json
from tqdm import tqdm

from .rag import search_chunks, pdf_to_md, chunk_md, md_dir_contents, file_co_rp_name, chunks_list
from .prompt import construct_prompt
from ..companies import Company, COS, company_aliases
from ..util import REPORT_TYPES_SHORT, DATA_DIR, OUT_DIR, logger_llm
from ..regex.terms import Term, TermCollection, TERMS_ALL


def stage_convert(cos: Company|Iterable[Company] = COS, report_types: Iterable[str]|str = REPORT_TYPES_SHORT,
                  num_workers: int = 2,
                  overwrite: bool = False,
                  progress_bar: bool = True):
    '''
    Convert all company pdfs to markdown.

    Parameters
    ----------
    cos : Company | Iterable[Company]
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
        for co in cos:
            for report_type in report_types:
                for pdf in os.listdir(os.path.join(co.raw_folder, report_type)):
                    if pdf.endswith('.pdf'):
                        futures.append(executor.submit(pdf_to_md, os.path.join(co.raw_folder, report_type, pdf), save=True, overwrite=overwrite))
        
        with tqdm(total=len(futures), desc="converting pdfs to md", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
            for future in as_completed(futures):
                future.result()  # discard result, but raise any exceptions
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
        for file in md_dir_contents(yield_level='report'):
            dir_to_save = file.replace('.md', '')
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
    total = 0
    for _, _, files in os.walk(os.path.join(DATA_DIR, 'md')):
        total += sum([1 for f in files if f.endswith('.md')])

    with tqdm(total=total, desc="Processing companies", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for report_dir in md_dir_contents(yield_level='chunk_folder'):
            co, report_type, report = file_co_rp_name(report_dir)
            
            save_dir = os.path.join(terms.folder, 'chunk-search-results', co, report_type)
            os.makedirs(save_dir, exist_ok=True)

            search_chunks(
                report_dir, terms, 
                save_path=os.path.join(save_dir, f'{report}.json'), 
                overwrite=overwrite
            )
            pbar.update(sum(1 for f in os.listdir(report_dir) if f.endswith('.md')))
    
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

    if isinstance(terms, str):
        res_folder = os.path.join(DATA_DIR, 'terms', terms, 'chunk-search-results')
    elif isinstance(terms, TermCollection):
        res_folder = os.path.join(terms.folder, 'chunk-search-results')
    else:
        raise ValueError(f'Invalid type for terms: {type(terms)}. Please provide either a TermCollection object, or the folder name of a TermCollection (as a str).')
    
    if not os.path.exists(res_folder):
        raise ValueError(f'TermCollection folder not found at {res_folder}. Please provide a valid TermCollection folder name, or a TermCollection object.')
        
    with tqdm(total=sum(1 for _ in md_dir_contents(yield_level='chunk_folder'))*len(terms), desc="Drafting prompts", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
        for co in os.listdir(res_folder):
            for report_type in os.listdir(os.path.join(res_folder, co)):
                for res_file in os.listdir(os.path.join(res_folder, co, report_type)):
                    
                    if res_file.endswith('.json'):
                        
                        report = res_file.replace('.json', '')
                        report_path = os.path.join(terms.folder, 'prompts', co, report_type, f'{report}')
                        chunks_path = os.path.join(DATA_DIR, 'md', co, report_type, report)
                        os.makedirs(report_path, exist_ok=True)
                        logger_llm.debug(f'PROCESSING: {co} - {report_type} - {res_file}')
                        
                        
                        with open(os.path.join(res_folder, co, report_type, res_file)) as f:
                            search_results = json.load(f)
                        
                        
                        for term_alias, chunk_inds in search_results.items():
                            if chunk_inds is None or len(chunk_inds) == 0:
                                logger_llm.debug(f'NO CHUNKS for {term_alias} in {co} - {report_type} - {report}. Skipping prompt construction for this term.')
                                pbar.update(1)
                                continue
                            
                            term = terms[term_alias]
                            
                            prompt_path = os.path.join(report_path, term.prompt_fname)

                            logger_llm.debug(f'Constructing prompt for {co} - {report_type} - {report} - {term_alias}. Saving to {prompt_path}')
                            if not overwrite and os.path.isfile(prompt_path):
                                logger_llm.debug(f'PROMPT NOT CREATED: {prompt_path} already exists and overwrite=False')
                                pbar.update(1)
                                continue

                            # check year is a valid 4 digit no
                            year = report.split('-')[-1]
                            if not (year.isdigit() and len(year) == 4):
                                logger_llm.warning(f'INVALID PROMPT YEAR: "{year}" from -> "{report}". Skipping prompt construction for this report.')
                                pbar.update(1)
                                continue
                            
                            chunks = chunks_list(chunks_path)
                            
                            construct_prompt(
                                company=co, report_type=report_type, year=year, term=term,
                                chunks=chunks, chunk_inds=chunk_inds,
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


