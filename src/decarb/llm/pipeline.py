import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from typing import Iterable, Any

from .rag import chunk_md, search_chunks, co_pdfs_to_md
from ..companies import Company, COS
from ..util import REPORT_TYPES, DATA_DIR
from ..regex.terms import Term, TermCollection, TERMS_ALL

def stage_convert(cos: Company|Iterable[Company] = COS, report_types: Iterable[str]|str = REPORT_TYPES,
                  num_workers: int = 5,
                  **chunk_kwargs: Any):
    '''
    Convert all company pdfs to markdown, and chunk them for later analysis.

    Parameters
    ----------
    cos : Company | Iterable[Company]
        The company/companies to convert. Defaults to BP, Chevron, Exxon, Shell, and Total.
    report_types : Iterable[str]
        The report type(s) to convert, given as folder names to be found in each Company's data folder.
        Defaults to annual reports and sustainability reports.
    '''
    if isinstance(cos, Company):
        cos = [cos]
    if isinstance(report_types, str):
        report_types = [report_types]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for co in cos:
            for report_type in report_types:
                futures.append(executor.submit(co_pdfs_to_md, co, report_type, overwrite=False, **chunk_kwargs))

        for future in as_completed(futures):
            future.result()

    print('Conversion and chunking complete.')


def stage_search(terms: Term|TermCollection):
    '''
    Search the markdown files created in stage_convert for the given terms, and save the results as csvs.

    Parameters
    ----------
    terms : Term | TermCollection
        The term or collection of terms to search for.
    '''
    pass
        

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


