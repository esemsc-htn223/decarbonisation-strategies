import pymupdf4llm
from ..companies import Company, Report
from ..regex.terms import Term, TermCollection, TERMS_ALL
from ..util import logger_llm, DATA_DIR, REPORT_TYPES_SHORT, COMPANIES_DIR


import os
import pymupdf4llm
import pymupdf
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Iterable, Literal
import json


load_dotenv()  # in case TESSDATA_PREFIX is in .env - required for ocr of troublesome docs

MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

def tidy_md(md:str):
    tidy_md = md.replace('$', r'\$')  # no maths in these reports
    tidy_md = re.sub(r'(-)\1{3,}', r'\1\1\1', tidy_md) # limit repeated - to 3
    tidy_md = re.sub(r'\n{3,}', r'\n\n', tidy_md) # limit repeated \n to 2
    return tidy_md

def pdf_to_md(report: Report, *,
              save:bool = True,
              save_as:str|None = None, 
              overwrite:bool=False) -> str:
    '''
    Convert the given pdf to markdown.

    Parameters
    ----------
    report : Report 
        The report to convert. The pdf path will be inferred from report.pdf_path.
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
    if save and save_as is None:
        save_as = report._md_path

    if not overwrite and os.path.isfile(save_as):
        logger_llm.debug(f'NOT CONVERTED: "{os.path.basename(pdf_path)}". overwrite=False, md file exists. pdf path: {pdf_path}')
        if save:
            return save_as
        else:
            return open(save_as).read()
    try:
        md = tidy_md(pymupdf4llm.to_markdown(pdf_path))
    except pymupdf.mupdf.FzErrorLimit as e:
        logger_llm.error(f'mupdf error - attempting without OCR. Full error: {type(e).__name__}: {e}')
        try:
            md = tidy_md(pymupdf4llm.to_markdown(pdf_path, use_ocr=False))
        except Exception as e2:
            logger_llm.error(f'ERROR mupdf error - even without OCR. Full error: {type(e2).__name__}: {e2}')
            raise
    logger_llm.debug(f'CONVERTED "{os.path.basename(pdf_path)}". len: {len(md)}. pdf path: {pdf_path}.')
    
    if save:
        with open(save_as, 'w') as f:
            f.write(md)
        logger_llm.debug(f'MD SAVED "{os.path.basename(save_as)}". full path: {save_as}. pdf path: {pdf_path}.')
        return save_as
    else:
        return md

def chunk_md(
        report:Report, 
        *, 
        max_chunk_size:int=MAX_CHUNK_SIZE, 
        min_chunk_size:int=MIN_CHUNK_SIZE, 
        save:bool=True,
        save_as:str|None=None,
        overwrite_chunks:bool=False,
        remove_original_md:bool=False,
        encoder_name:str='cl100k_base',
    ) -> list[str]:
    '''
    Split the given markdown into chunks via content-aware splitting.
    Content is first split on markdown headers (##, and ###). Each chunk is then split to
    enforce a maximum chunk size (in the token space). Finally, small chunks are added to the
    next chunk to enforce a minimum chunk size.

    Parameters
    ----------
    md : str
        The markdown content to split, or the path to a markdown file to read and split.
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
            logger_llm.debug(f'NOT CHUNKED: {os.path.basename(dir_to_save)}. overwrite_chunks=False, chunk files exist. Full path: {dir_to_save}')
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
            logger_llm.debug(f'CHUNKED to {os.path.basename(dir_to_save)}. {len(chunks)} chunks created. Full path: {dir_to_save}')
        
        out = [chunk.page_content for chunk in chunks]
        
    if remove_original_md and md_path is not None and os.path.isfile(md_path):
        os.remove(md_path)
        logger_llm.debug(f'REMOVED md file {os.path.basename(md_path)}. Full path: {md_path}')
    
    return out

def search_chunks(
        report: Report, 
        terms: Term|TermCollection = TERMS_ALL, 
        *,
        save:bool=True,
        save_as: str | None = None, 
        overwrite: bool = False
    ) -> dict[str, list[int]]:
    '''
    Search the given markdown chunks for the given terms. Returns a list of indices, where each index
    indicates the position of the chunk where the term was found.

    Parameters
    ----------
    report : Report
        The report object containing the chunks to search.
    terms : Term | TermCollection
        The term or collection of terms to search for.
    save : bool
        Whether to save the search results as a json file. If False, the search results will be returned as a dictionary.
    save_as : str | None
        The path to save the search results to, as a json file. If None, results will be saved to the
        `report._search_results_path` for the given term collection.
    overwrite : bool
        Whether to overwrite existing search results the save location. If False and a file already exists at `save_as` or the default location, 
        the function will skip searching and return the existing results. Only relevant if `save` is True.

    Returns
    -------
    dict[str, list[int]]
        A dictionary where the keys are term aliases and the values are lists of chunk indices where the term was found.
    '''


    chunks = report.chunks
    if isinstance(terms, Term):
        terms = [terms]
    results = {}

    for term in terms:
        results[term.alias] = [i for i, chunk in enumerate(chunks) if re.search(term.regex, chunk, re.I)]
        logger_llm.debug(f'SEARCHED for {term.alias}. Found in chunks: {results[term.alias]}')

    if save:
        save_path = report._search_results_path(terms) if save_as is None else save_as
        if not overwrite and os.path.isfile(save_path):
            logger_llm.debug(f'SEARCH NOT SAVED. overwrite=False, file exists: {save_path}')
        else:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger_llm.debug(f'SEARCH SAVED to {save_path}.')

    return results