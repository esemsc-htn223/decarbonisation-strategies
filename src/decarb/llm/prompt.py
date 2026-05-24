from typing import Iterable, Literal

from ..companies import Company
from ..regex.terms import Term
from ..util import REPORT_TYPES_LONG

def construct_extracts(
        chunks:Iterable[str], chunk_inds:Iterable[int], 
        overlap_strategy:Literal["before", "after", "both", ]|None = None,
        overlap_amount:float = 0.5
) -> str:
    '''
    Construct the extracts to include in the prompt, based on the given chunks and chunk indices, and the specified overlap strategy and amount.
    
    Parameters
    ----------
    chunks : Iterable[str]
        The chunks to construct the extracts from, as an iterable of strings.
    chunk_inds : Iterable[int]
        The indices of the chunks to construct the extracts from, as an iterable of ints. 
    overlap_strategy : Literal["before", "after", "both", ]|None, optional
        Whether to include text from the preceeding or following chunk for each extract, or both.
        This is to provide additional context to the LLM, as sometimes relevant information may be split across chunk boundaries.
        If None (the default), no additional context will be included.
    overlap_amount : float, optional
        The amount of text to include from the adjacent chunk(s), as a proportion of the chunk.
        Defaults to 0.5 (i.e., half the adjacent chunk(s)). Only has an effect if overlap_strategy is not None.
    
    Returns
    -------
    str
        The constructed extracts, as a string.
    '''
    extracts = []
    n_chunks = len(chunks)
    for i, chunk_ind in enumerate(chunk_inds):
        extract = chunks[chunk_ind]
        
        if overlap_strategy in ['before', 'both'] and chunk_ind > 0:
            prev_chunk = chunks[chunk_ind - 1]
            overlap_len = int(len(prev_chunk.split()) * overlap_amount)
            extract = ' '.join(prev_chunk.split()[-overlap_len:]) + "\n\n" + extract
        
        if overlap_strategy in ['after', 'both'] and chunk_ind < len(chunks) - 1:
            next_chunk = chunks[chunk_ind + 1]
            overlap_len = int(len(next_chunk.split()) * overlap_amount)
            extract = extract + "\n\n" + ' '.join(next_chunk.split()[:overlap_len])

        header = f"# Extract {i+1} (section {chunk_ind} / {n_chunks}):\n\n"
        
        extracts.append(f"{header + extract}")
    
    return "\n\n------------------\n\n".join(extracts)

def construct_prompt(
        company:str|Company, report_type:str, year:int|str, term:str|Term, 
        chunks:Iterable[str], chunk_inds:Iterable[int], 
        overlap_strategy:Literal["before", "after", "both", ]|None = None,
        overlap_amount:float = 0.5
):
    '''
    Construct a prompt to pass to the LLM, including task description, output format
    and the relevant extracts.
    
    Parameters
    ----------
    company : str or Company
        The company the report relates to, either as a string (e.g., "Shell") or as a Company object.
        If a Company object is passed, the company's alias will be used in the prompt.
    report_type : str
        The type of report the extracts are taken from (e.g., "annual-report" or "sustainability-report"). 
        This will be converted to a more reader-friendly format in the prompt using REPORT_TYPES_LONG (e.g., "annual financial report" or "sustainability report").
    year : int or str
        The year the report was published.
    term : str or Term
        The technology/term the extracts relate to, either as a string (e.g., "carbon capture") or as a Term object. 
        If a Term object is passed, the term's prompt_alias will be used in the prompt.
    chunks : Iterable[str]
        The extracts to include in the prompt, as an iterable of strings.
    chunk_inds : Iterable[int]
        The indices of the extracts to include, as an iterable of ints. 
    overlap_strategy : Literal["before", "after", "both", ]|None, optional
        Whether to include text from the preceeding or following chunk for each extract, or both.
        This is to provide additional context to the LLM, as sometimes relevant information may be split across chunk boundaries.
        If None (the default), no additional context will be included.
    overlap_amount : float, optional
        The amount of text to include from the adjacent chunk(s), as a proportion of the chunk.
        Defaults to 0.5 (i.e., half the adjacent chunk(s)). Only has an effect if overlap_strategy is not None.
    
    Returns
    -------
    str
        The constructed prompt, as a string.
    '''

    if isinstance(company, str):
        co_alias = company
    elif isinstance(company, Company):
        co_alias = company.alias
    else:
        raise ValueError(f"company should be either the name of a company (as a str), or a Company object. Got {type(company)}")
    
    report_alias = REPORT_TYPES_LONG.get(report_type, report_type.replace('-', ' '))
    
    chunks_str = construct_extracts(chunks, chunk_inds, overlap_strategy, overlap_amount)
    
    overlap_info = ""
    if overlap_strategy is not None:
        overlap_info = f"Note, extracts include {overlap_amount:.0%} of"
        if overlap_strategy in ['before', 'both']:
            overlap_info += " the previous "
        if overlap_strategy in ['after', 'both']:
            if overlap_strategy in ['both']:
                overlap_info += " and"
            overlap_info += " following section"
            if overlap_strategy in ['both']:
                overlap_info += "s"
        overlap_info += f" of the report to provide additional context. As such, some information may be repeated. Additionally, OCR errors may result in some incorrect text or formatting in the extracts.\n\n"


    prompt = f'''The following extracts have been taken from {co_alias}'s {year} {report_alias} and all relate to {term.prompt_alias}. 

Use the contained information to answer the following questions:
1. "How positive or negative is the overall sentiment towards {term.prompt_alias}?"
2. "What key commitments has {co_alias} made to {term.prompt_alias}?"
3. "At what stage of implementation is {term.prompt_alias}?"
4. "How mature does {co_alias} consider {term.prompt_alias} to be?"

{overlap_info}

Answer in the following JSON format, with only the JSON and no additional text:
{{
    "sentiment": float,  # sentiment towards {term.prompt_alias}, quantified on a numerical scale -1 to 1, where -1 is totally negative, 0 is neutral, and 1 is totally positive.
    "key-commitments": list[str],  # key extracts/quotes showing commitments towards {term.prompt_alias}
    "stage": Literal["commercial-deployment", "demonstration", "pilot", "research and development", "unclear"],  # stage of {term.prompt_alias}
    "maturity": Literal["emerging", "established", "phasing-out", "unclear"]  # maturity of {term.prompt_alias}
}}


# Extracts:
---------

{chunks_str}
'''
    
    return prompt

