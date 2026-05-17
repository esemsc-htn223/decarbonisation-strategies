import pymupdf4llm
from decarb.companies import Company
from decarb.regex.terms import Term, TermCollection, TERMS_ALL
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
    tidy_md = md.replace('$', '\$')  # no maths in these reports
    tidy_md = re.sub(r'(-)\1{3,}', r'\1\1\1', tidy_md) # limit repeated - to 3
    tidy_md = re.sub(r'\n{3,}', r'\n\n', tidy_md) # limit repeated \n to 2
    return tidy_md

def pdf_to_md(co: Company, report_type: str):
    for pdf in os.listdir(os.path.join(co.raw_folder, report_type)):
        if pdf.endswith('.pdf'):
            pdf_path = os.path.join(co.raw_folder, report_type, pdf)
            md_path = os.path.join(co.md_folder, report_type, pdf.replace('.pdf', '.md'))
            with open(md_path, 'w') as f:
                md = pymupdf4llm.to_markdown(pdf_path)
                f.write(tidy_md(md))
                

def chunk_md(md:str, *, 
             max_chunk_size:int=MAX_CHUNK_SIZE, 
             min_chunk_size:int=MIN_CHUNK_SIZE, 
             #chunk_overlap:int=CHUNK_OVERLAP
             ) -> list[str]:
    '''
    Split the given markdown into chunks via content-aware splitting.
    Content is first split on markdown headers (##, and ###). Each chunk is then split to
    enforce a maximum chunk size (in the token space). Finally, small chunks are added to the
    next chunk to enforce a minimum chunk size.

    Parameters
    ----------
    md : str
        The markdown text to split.
    max_chunk_size : int
        Maximum chunk size, in the token space. Note due to token character length being variable,
        chunks may have a __len__ that is longer than `max_chunk_size`
    min_chunk_size : int
        Minimum chunk size. Note this is calculated by characters (not tokens).
    '''
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

    return [chunk.page_content for chunk in chunks]


def search_md_chunks(chunks: list[str], term: Term) -> list[bool]:
    '''
    Search the given markdown chunks for the given terms. Returns a list of booleans, where each boolean
    indicates whether the term was found in the corresponding chunk.

    Parameters
    ----------
    chunks : list[str]
        The markdown chunks to search.
    term : Term
        The term to search for.
    '''
    results = []

    for chunk in chunks:
        results.append(bool(re.search(term.regex, chunk, re.I)))
    return results

def search_md_chunks_parallel(chunks: list[str], term: Term, max_workers: int = 4) -> list[bool]:
    '''
    Search the given markdown chunks for the given term in parallel. Returns a list of booleans, where each boolean
    indicates whether the term was found in the corresponding chunk.

    Parameters
    ----------
    chunks : list[str]
        The markdown chunks to search.
    term : Term
        The term to search for.
    max_workers : int
        The maximum number of worker threads to use for parallel processing.
    '''
    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(re.search, term.regex, chunk, re.I): i for i, chunk in enumerate(chunks)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = bool(result)
            except Exception as exc:
                print(f'Chunk {index} generated an exception: {exc}')
                results[index] = False

    return results

def search_md_chunks_parallel_multiple_terms(chunks: list[str], terms: list[Term], max_workers: int = 4) -> list[dict[str, bool]]:
    '''
    Search the given markdown chunks for the given terms in parallel. Returns a list of dictionaries, where each dictionary
    contains the search results for each term in the corresponding chunk.

    Parameters
    ----------
    chunks : list[str]
        The markdown chunks to search.
    terms : list[Term]
        The terms to search for.
    max_workers : int
        The maximum number of worker threads to use for parallel processing.
    '''
    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(search_md_chunks_parallel, [chunk], term, max_workers): (i, term.alias) for i, chunk in enumerate(chunks) for term in terms}
        for future in as_completed(future_to_index):
            index, term_alias = future_to_index[future]
            try:
                result = future.result()
                if results[index] is None:
                    results[index] = {}
                results[index][term_alias] = result[0]
            except Exception as exc:
                print(f'Chunk {index} generated an exception: {exc}')
                if results[index] is None:
                    results[index] = {}
                results[index][term_alias] = False

    return results