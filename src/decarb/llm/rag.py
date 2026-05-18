import pymupdf4llm
from ..companies import Company
from ..regex.terms import Term, TermCollection, TERMS_ALL
import os
import pymupdf4llm
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

def tidy_md(md:str):
    tidy_md = md.replace('$', r'\$')  # no maths in these reports
    tidy_md = re.sub(r'(-)\1{3,}', r'\1\1\1', tidy_md) # limit repeated - to 3
    tidy_md = re.sub(r'\n{3,}', r'\n\n', tidy_md) # limit repeated \n to 2
    return tidy_md

def chunk_md(md:str, *, 
             max_chunk_size:int=MAX_CHUNK_SIZE, 
             min_chunk_size:int=MIN_CHUNK_SIZE, 
             dir_to_save:str|None=None,
             overwrite:bool=False,
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
        The markdown content to split.
    max_chunk_size : int
        Maximum chunk size, in the token space. Note due to token character length being variable,
        chunks may have a __len__ that is longer than `max_chunk_size`
    min_chunk_size : int
        Minimum chunk size. Note this is calculated by characters (not tokens).
    dir_to_save : str | None
        The directory to save the chunked markdown files to. If None, the chunked files will not be saved.
    overwrite : bool
        Whether to overwrite existing chunk files in `dir_to_save`. If False and chunk files already exist,
        the function will skip conversion.

    Returns
    -------
    list[str]
        List of markdown chunks.
    '''
    if os.path.isdir(dir_to_save) and not overwrite:
        list_dir = os.listdir(dir_to_save)
        list_dir.pop('.DS_Store') if '.DS_Store' in list_dir else None
        if len(list_dir) > 0 and all(f.endswith('.md') for f in list_dir):
            print(f'Chunk files already exist in {dir_to_save}, and overwrite is False. Skipping conversion.')
            return [open(os.path.join(dir_to_save, f)).read() for f in sorted(os.listdir(dir_to_save)) if f.endswith('.md')]

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

    return [chunk.page_content for chunk in chunks]

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
            
            md = tidy_md(pymupdf4llm.to_markdown(pdf_path))
            chunk_md(md, dir_to_save=md_chunks_dir.replace('.md', ''), overwrite=overwrite, **chunk_kwargs)
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

    return results
