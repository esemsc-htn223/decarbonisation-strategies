import pymupdf4llm
from ..companies import Company
from ..regex.terms import Term, TermCollection, TERMS_ALL
from ..util import logger_llm, DATA_DIR
import os
import pymupdf4llm
import pymupdf
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()  # in case TESSDATA_PREFIX is in .env - required for ocr of troublesome docs

MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

def chunk_files_exist(chunk_dir:str) -> bool:
    '''
    Check if chunk files already exist in the given directory.

    Parameters
    ----------
    chunk_dir : str
        The directory to check for chunk files.

    Returns
    -------
    bool
        True if chunk files exist, False otherwise.
    '''
    if os.path.isdir(chunk_dir):
        list_dir = os.listdir(chunk_dir)
        list_dir.pop('.DS_Store') if '.DS_Store' in list_dir else None
        return len(list_dir) > 0 and any(f.endswith('.md') for f in list_dir)
    return False

def tidy_md(md:str):
    tidy_md = md.replace('$', r'\$')  # no maths in these reports
    tidy_md = re.sub(r'(-)\1{3,}', r'\1\1\1', tidy_md) # limit repeated - to 3
    tidy_md = re.sub(r'\n{3,}', r'\n\n', tidy_md) # limit repeated \n to 2
    return tidy_md

def chunk_md(md:str, *, 
             max_chunk_size:int=MAX_CHUNK_SIZE, 
             min_chunk_size:int=MIN_CHUNK_SIZE, 
             dir_to_save:str|None=None,
             overwrite_chunks:bool=False,
             remove_original_md:bool=False,
             #chunk_overlap:int=CHUNK_OVERLAP  # not implemented - could do manually?
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
    dir_to_save : str | None
        The directory to save the chunked markdown files to. If None, the chunked files will not be saved.
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
    if os.path.isfile(md):
        md_path = md
    else:
        md_path = None
    
    if os.path.isdir(dir_to_save) and not overwrite_chunks:
        if chunk_files_exist(dir_to_save):
            logger_llm.debug(f'NOT CHUNKED: {os.path.basename(dir_to_save)}. overwrite_chunks=False, chunk files exist. Full path: {dir_to_save}')
            
            out = [open(os.path.join(dir_to_save, f)).read() for f in sorted(os.listdir(dir_to_save)) if f.endswith('.md')]
    else:
        # read md if required
        if md_path:
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

def pdf_to_md(pdf_path:str, *,
              save:bool = True,
              save_path:str|None = None, 
              overwrite:bool=False) -> str:
    '''
    Convert the given pdf to markdown.

    Parameters
    ----------
    pdf_path : str
        The path to the pdf to convert.
    save : bool
        Whether to save the converted markdown as a file. If False, the markdown content will be returned as a string.
    save_path : str | None
        The path to save the converted markdown file to. 
        If None, markdown will be saved in the md directory, with the same name but with a .md extension.
    overwrite : bool
        Whether to overwrite existing markdown files. If False and a markdown file already exists at `save_path`, the function will skip conversion and return the existing markdown content.

    Returns
    -------
    str
        If save=True, the directory where the chunked markdown files are saved. Otherwise, the markdown content as a string.
    '''
    # get default save dir
    if save_path is None and save:
        save_path = pdf_path.replace(os.path.join('data', 'raw'), os.path.join('data', 'md')).replace('.pdf', '.md')

    if not overwrite and os.path.isfile(save_path):
        logger_llm.debug(f'NOT CONVERTED: "{os.path.basename(pdf_path)}". overwrite=False, chunk files exist. pdf path: {pdf_path}')
        if save:
            return save_path
        else:
            return open(save_path).read()
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
        with open(save_path, 'w') as f:
            f.write(md)
        logger_llm.debug(f'MD SAVED "{os.path.basename(save_path)}". full path: {save_path}. pdf path: {pdf_path}.')
        return save_path
    else:
        return md

def co_pdfs_to_md(co: Company, report_type: str, *, overwrite: bool = False, **chunk_kwargs) -> list[str]:
    '''
    Convert the pdfs for the given company and report type to chunked markdown, 
    and save them in the company's md folder.

    Parameters
    ----------
    co : Company
        The company whose pdfs to convert.
    report_type : str
        The report type to convert, given as a folder name to be found in the Company's data folder.

    Returns
    -------
    list[str]
        A list of the locations of the chunked markdown folders created.
    '''

    md_locs = []
    reports_dir = os.path.join(co.raw_folder, report_type)
    md_dir = os.path.join(co.md_folder, report_type)
    if not os.path.isdir(reports_dir):
        raise ValueError(f'{reports_dir} is not a directory')
    if not os.path.isdir(md_dir):
        os.makedirs(md_dir)

    for pdf in os.listdir(reports_dir):
        if pdf.endswith('.pdf'):
            pdf_path = os.path.join(reports_dir, pdf)
            md_chunks_dir = os.path.join(md_dir, pdf.replace('.pdf', ''))

            md = pdf_to_md(pdf_path, save=False, overwrite=False)
            chunk_md(md, dir_to_save=md_chunks_dir, overwrite=overwrite, **chunk_kwargs)
            md_locs.append(md_chunks_dir)

    return md_locs

def search_chunks(chunks: list[str]|str, terms: Term|TermCollection = TERMS_ALL) -> dict[str, list[int]]:
    '''
    Search the given markdown chunks for the given terms. Returns a list of indices, where each index
    indicates the position of the chunk where the term was found.

    Parameters
    ----------
    chunks : list[str]|str
        The markdown chunks to search, either as a list of md content, or the path to a folder containing md files.
    terms : Term | TermCollection
        The term or collection of terms to search for.

    Returns
    -------
    dict[str, list[int]]
        A dictionary where the keys are term aliases and the values are lists of chunk indices where the term was found.
    '''
    if isinstance(chunks, str):
        if os.path.isfile(chunks):
            chunks = [open(chunks).read()]
        elif os.path.isdir(chunks):
            chunks = [open(f).read() for f in os.listdir(chunks) if f.endswith('.md')]
        else:
            raise ValueError(f'chunks is not a file or directory: {chunks}')

    if isinstance(terms, Term):
        terms = [terms]
    results = {}

    for term in terms:
        results[term.alias] = [i for i, chunk in enumerate(chunks) if re.search(term.regex, chunk, re.I)]
        logger_llm.debug(f'SEARCHED for {term.alias}. Found in chunks: {results[term.alias]}')

    return results
