import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from typing import Iterable, Any, Literal
import json
from tqdm import tqdm

from .rag import search_chunks, co_pdfs_to_md, pdf_to_md, chunk_md
from ..companies import Company, COS, company_aliases
from ..util import REPORT_TYPES_SHORT, DATA_DIR
from ..regex.terms import Term, TermCollection, TERMS_ALL

def md_dir_contents(yield_level: Literal['report', 'chunk_folder', 'chunk']) -> Iterable[str]:
    '''
    Generator that yields the paths of all markdown files/directories for all companies and report types.
    '''
    for co in os.listdir(os.path.join(DATA_DIR, 'md')):
        co_dir = os.path.join(DATA_DIR, 'md', co)
        if not os.path.isdir(co_dir):
            continue
        for report_type in os.listdir(co_dir):
            report_type_dir = os.path.join(co_dir, report_type)
            if not os.path.isdir(report_type_dir):
                continue
            for report in os.listdir(report_type_dir):
                report = os.path.join(report_type_dir, report)

                if yield_level == 'report':
                    if report.endswith('.md'):
                        yield report

                elif yield_level == 'chunk_folder':
                    if os.path.isdir(report):
                        yield report

                elif yield_level == 'chunk':
                    if os.path.isdir(report):
                        for md_file in os.listdir(report):
                            if md_file.endswith('.md'):
                                yield os.path.join(report, md_file)
                                
                else:
                    raise ValueError(f"yield_level must be one of 'report', 'chunk_folder', or 'chunk'. Got {yield_level}")

def file_co_rp_name(file_path) -> tuple[str|None, str|None, str|None]:
    '''
    Get the company, report type, and report name from a given filepath in the data directory.
    
    Parameters
    ----------
    file_path : str
        The path of the file to extract the company, report type, and report name from.
    
    Returns
    -------
    tuple[str|None, str|None, str|None]
        A tuple containing the company, report type, and report name, in that order. 
        If any of these cannot be extracted from the file path, the corresponding value in the tuple will be None.
    '''
    flist = file_path.replace(DATA_DIR + os.path.sep, '').split(os.path.sep)
    if len(flist) < 4:
        flist += [None] * (4 - len(flist))  # Pad with None if not enough elements
    return flist[1], flist[2], flist[3]

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
            out_dir = os.path.join(terms.folder, 'chunk-search-results', co, report_type)
            if os.path.exists(os.path.join(out_dir, f'{report}.json')) and not overwrite:
                continue
            os.makedirs(out_dir, exist_ok=True)

            chunks = [open(os.path.join(report_dir, f)).read() for f in sorted(os.listdir(report_dir)) if f.endswith('.md')]
            search_results = search_chunks(chunks, terms)
            pbar.update(len(chunks))

            # save search results as csv
            with open(os.path.join(out_dir, f'{report}.json'), 'w') as f:
                json.dump(search_results, f, indent=4)
    
    if not progress_bar:
        print('search complete')

def stage_draft():
    '''
    Use the markdown files and search results from the previous stages to draft LLM prompts for each article.
    '''
    pass

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


