from dataclasses import dataclass
import io
from typing import Iterable, Literal, Iterator
import os
from dotenv import load_dotenv
from openai import OpenAI, NotFoundError
from openai.types import Batch
from pydantic import BaseModel
import instructor
import json
import time
import uuid
import tiktoken
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

from decarb.companies import Company, CompanyCollection, Report, file_metadata, co_files, REPORT_TYPES_LONG
from decarb.terms import Term, TermCollection
from decarb.util import OUT_DIR, logger_llm, TERMS_DIR, BASE_DIR, resolve_path
from decarb.llm.schemas import RESPONSE_SCHEMAS

DEFAULT_MODEL = 'gpt-5.1'

# waiting required
STATUS_WORKING = ('validating', 'in_progress', 'finalizing')  # may still produce a result file
STATUS_FAILING = ('cancelling',)  # will not produce a result file

# no further waiting required
STATUS_FAILED = ('failed', 'cancelled', 'expired')  # will not produce a result file
STATUS_COMPLETED = ('completed',)  # result file available

STATUS_TERMINAL = STATUS_FAILED + STATUS_FAILING  # statuses for which we will not get a result file
STATUS_IN_QUEUE = STATUS_WORKING + STATUS_FAILING  # all statuses that consume enqueued limit
STATUS_NOT_IN_QUEUE = STATUS_FAILED + STATUS_COMPLETED  # all statuses that do not consume enqueued limit


# Company and brand names to remove when masking. Matching is case-insensitive, because report
# headings are frequently upper case (e.g. "SHELL STORY: WHAT WE DO").
COMPANY_PATTERNS = {
    'bp': r'\bBP\b|\bB\.P\.|British\s+Petroleum|\bAmoco\b|\bCastrol\b|\bAral\b|\bArco\b',
    'chevron': r'\bChevron\b|\bTexaco\b|\bCaltex\b|\bUnocal\b',
    'exxon': r'\bExxonMobil\b|\bExxon\b|\bMobil\b|\bEsso\b',
    'shell': r'\bShell\b|Royal\s+Dutch',
    'eni': r'\bEni\b|\bAgip\b',
    'total': r'\bTotalEnergies\b|\bTotal\s+E&P\b|\bTotal\s+S\.A\.|\bElf\b|\bFina\b',  # can't do just 'Total'
}
MASK_TOKEN = 'the company'


def mask_entities(text: str) -> str:
    '''
    Replace company and brand names in a piece of text with a neutral placeholder.

    Every major's patterns are applied, not just the report's own owner, so a competitor named in the
    text is masked too. Matching is case-insensitive. Years are not masked, and neither is the bare
    word "Total", which is far more often an accounting term than the company.

    Parameters
    ----------
    text : str
        The text to mask.

    Returns
    -------
    str
        The text with every recognised company or brand name replaced by "the company".
    '''
    for pattern in COMPANY_PATTERNS.values():
        text = re.sub(pattern, MASK_TOKEN, text, flags=re.I)
    # adjacent names ("Royal Dutch Shell") collapse to a single placeholder
    return re.sub(rf'(?:{MASK_TOKEN}\s+)+{MASK_TOKEN}', MASK_TOKEN, text, flags=re.I)


def construct_extracts(
        report:Report, term:Term, *,
        overlap_strategy:Literal["before", "after", "both", ]|None = None,
        overlap_amount:float = 0.2,
        max_extracts:int|None = None,
        mask:bool = False
) -> tuple[str, int, int]:
    '''
    Construct the extracts to include in the prompt, based on the given chunks and chunk indices, and the specified overlap strategy and amount.
    
    Parameters
    ----------
    report : Report
        The Report object containing the chunks to construct the extracts from, and the search results for the given term.
    term : Term
        The Term object containing the alias to use to retrieve the search results for the given term from the report.
    overlap_strategy : Literal["before", "after", "both", ]|None, optional
        Whether to include text from the preceeding or following chunk for each extract, or both.
        This is to provide additional context to the LLM, as sometimes relevant information may be split across chunk boundaries.
        If None (the default), no additional context will be included.
    overlap_amount : float, optional
        The amount of text to include from the adjacent chunk(s), as a proportion of the chunk.
        Defaults to 0.2 (i.e., a fifth of the adjacent chunk(s)). Only has an effect if overlap_strategy is not None.
    max_extracts : int or None, optional
        The maximum number of extracts to include. If the term matched more chunks than this, the
        extracts are taken evenly spaced across the document rather than from the start, so the
        sample covers the whole report. If None (the default), every matching chunk is included.
    mask : bool, optional
        Whether to replace the company's name and brands with a neutral placeholder, via
        `mask_entities`. Defaults to False.
    
    Returns
    -------
    tuple[str, int, int]
        The constructed extracts, as a string.

    Raises
    ------
    FileNotFoundError
        If no search results are found for the given term in the report, or if the `report.n_chunks` is 0.
    '''
    search_results = report.search_results(term.collection)
    if search_results is None:
        raise FileNotFoundError(f'No search results found for term {term.prompt_alias} in report {report.report_name}. Cannot construct extracts.')
    n_chunks = report.n_chunks
    if n_chunks == 0:
        raise FileNotFoundError(f'No chunks found for report {report.report_name}. Cannot construct extracts.')

    chunk_inds = search_results[term.alias]
    chunks = report.chunks

    n_total = len(chunk_inds)
    if max_extracts is not None and n_total > max_extracts:
        # evenly spaced, so a capped prompt still covers the whole document
        keep = np.linspace(0, n_total - 1, max_extracts).round().astype(int)
        chunk_inds = [chunk_inds[i] for i in sorted(set(keep.tolist()))]

    extracts = []
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

        if mask:
            extract = mask_entities(extract)

        header = f"# Extract {i+1} (section {chunk_ind} / {n_chunks}):\n\n"
        
        extracts.append(f"{header + extract}")
    
    return "\n\n------------------\n\n".join(extracts), len(extracts), n_total


def _prompt_task_v1(term:Term, *, co_alias:str, co_ref:str, year:int, mask:bool) -> str:
    '''Task text for schema version 1 (PromptResponseV1): sentiment, TRL, commercial stage and commitments.'''
    if mask:
        basis = (f"Use these extracts to answer the following questions. Base your answers only on the extracts, "
                 f"not on any other information you may have about this company or about "
                 f"{term.prompt_alias}. Answers must be evidenced by direct quotes from the extracts, along\n"
                 f"with a justification of why each quote is relevant.")
    else:
        basis = (f"Use these extracts to answer the following questions. Base your answers only on the extracts, not on any other information you may \n"
                 f"have about {co_alias}'s use of {term.prompt_alias} in {year}. Answers must be evidenced by direct quotes from the extracts, along\n"
                 f"with a justification of why each quote is relevant.")

    return f'''{basis}

# Questions:
1. "How positive or negative is the overall sentiment towards {term.prompt_alias}? 
Score on a scale from -1 (totally negative) to 1 (totally positive)."

2. "How mature is {term.prompt_alias} framed to be in the text in terms of Technology Readiness Levels (TRL)? 
Assign a TRL from 1-9 based on the following definitions:
- TRL 1. Initial idea reported: Basic principles have been defined
- TRL 2. Application formulated: Concept and application have been formulated
- TRL 3. Components tested: Critical functions proven experimentally without all components integrated
- TRL 4. Early prototype tested: First prototype successfully tested in non-representative conditions
- TRL 5. Large prototype tested: Integrated components proven in representative operational conditions
- TRL 6. Prototype operated in real conditions: Prototype proven at a representative scale and in conditions to be deployed
- TRL 7. Validated at pre-commercial scale: System demonstrated sufficiently to move to full-scale
- TRL 8. First-of-a-kind commercial scale commissioned: Commercial scale realised in real-world environment
- TRL 9. Technically ready for sale or licensing: Successfully operated and commercially available with technology guarantees. Ready for further learning-by-doing.
If insufficient information is available to assign a TRL, leave the answer null."

3. "At what stage of commercialisation is {term.prompt_alias} within the company?
Choose from: 
- 'research and development'
- 'pilot'
- 'demonstration'
- 'commercial-deployment'
- 'phasing-out'
- 'unclear'"

4. "What commitments has {co_ref} made towards {term.prompt_alias}?
Provide a list of commitments, flagging commitments that include specific measurable targets as verifiable."
'''


def _prompt_task_v2(term:Term, *, co_ref:str, year:int, mask:bool) -> str:
    '''
    Task text for schema version 2 (PromptResponseV2, built per term with build_extraction_model).
    Deliberately short: field definitions live in the response schema, so this only covers what the
    schema cannot: the scope of an action, de-duplication, evidence rules and how to resolve dates.
    '''
    tech = term.prompt_alias
    if mask:
        timing = ("The year of the report is not given. Where timing is only relative (e.g. 'next year'), "
                  "leave the year fields null and record the phrase in timing_text.")
    else:
        timing = (f"The report is for {year}. Resolve relative timing (e.g. 'next year', 'over the next five "
                  f"years') against {year}.")

    return f'''Extract every action {co_ref} reports or announces concerning {tech}, and record its overall stance towards {tech}. Follow the field descriptions in the response schema.

- An action is something the company itself (including its subsidiaries and joint ventures) has done, is doing, or has committed or planned to do. Include past actions the report mentions as well as new ones, and actions that cover {tech} in conjunction with other technologies.
- Do not record general statements about {tech}, or the actions of governments, other companies or the industry as a whole, as actions. Statements about {tech}'s role or prospects, or about the company's strategy towards it, belong in the stance.
- Record each distinct action once, combining every extract that refers to it.
- Use only the extracts, not anything else you know about the company or {tech}. Copy quotes exactly as they appear in the extracts.
- {timing}
- The extracts were found by keyword search, so some may not discuss {tech}. Take nothing from them and list them in off_topic_extracts.
- If there are no actions, return an empty list.'''


def construct_prompt(
        report:Report, term:Term,
        *,
        schema_version:int = 1,
        overlap_strategy:Literal["before", "after", "both", ]|None = None,
        overlap_amount:float = 0.2,
        max_extracts:int|None = None,
        mask:bool = False,
        save:bool = True,
        save_as:str|None = None,
        overwrite:bool = False
):
    '''
    Construct a prompt to pass to the LLM, including task description and the relevant extracts.

    Parameters
    ----------
    report : Report
        The Report object containing the chunks to construct the extracts from, and the search results for the given term.
    term : Term
        The Term object containing the alias to use to retrieve the search results for the given term from the report.
    schema_version : int, optional
        Which response schema the prompt is written for. 1 (the default) gives the original questions,
        answered with PromptResponseV1. 2 gives the action-extraction task, answered with the
        PromptResponseV2 model built for this term by build_extraction_model(term.alias, term.variants).
        The v2 prompt relies on the schema's field descriptions, so it must be sent with that schema.
    overlap_strategy : Literal["before", "after", "both", ]|None, optional
        Whether to include text from the preceeding or following chunk for each extract, or both.
        This is to provide additional context to the LLM, as sometimes relevant information may be split across chunk boundaries.
        If None (the default), no additional context will be included.
    overlap_amount : float, optional
        The amount of text to include from the adjacent chunk(s), as a proportion of the chunk.
        Defaults to 0.2 (i.e., a fifth of the adjacent chunk(s)). Only has an effect if overlap_strategy is not None.
    max_extracts : int or None, optional
        The maximum number of extracts to include, passed to `construct_extracts`. If None (the
        default), every matching chunk is included.
    mask : bool, optional
        Whether to replace company names with a neutral placeholder and leave the company and year
        out of the task text. Defaults to False.
    save : bool, optional
        Whether to save the constructed prompt as a text file in the report's term directory (or at the path specified by `save_as` if provided). Defaults to True.
    save_as : str or None, optional
        The path at which to save the constructed prompt. If None (the default), the prompt will be saved in the report's term directory.
        For schema versions after 1, the default filename gets a '.v{schema_version}' suffix before '.txt', so prompts for
        different versions do not overwrite (or, with overwrite=False, silently stand in for) each other. Only has an effect if `save` is True.
    overwrite : bool, optional
        Whether to overwrite an existing file at the save path. Defaults to False.

    Returns
    -------
    str
        The constructed prompt, as a string.

    Raises
    ------
    ValueError
        If schema_version is not one of RESPONSE_SCHEMAS, or the save path does not end with .txt.
    '''
    if schema_version not in RESPONSE_SCHEMAS:
        raise ValueError(f'schema_version should be one of {list(RESPONSE_SCHEMAS.keys())}. Got {schema_version}')

    if save_as is None:
        save_path = report._prompt_path(term.collection, term, schema_version)
    else:
        save_path = save_as
    if not save_path.endswith('.txt'):
        raise ValueError(f'save_as should be a path ending with .txt. Got {save_path}')

    co_alias = report.company.full_name
    co_ref = MASK_TOKEN if mask else co_alias
    year = report.year
    report_alias = REPORT_TYPES_LONG.get(report.report_type, report.report_type.replace('-', ' '))

    chunks_str, n_used, n_available = construct_extracts(
        report=report, term=term,
        overlap_strategy=overlap_strategy, overlap_amount=overlap_amount,
        max_extracts=max_extracts, mask=mask
    )

    if mask:
        # no company or year in the preamble: naming them anchors the answer on the model's priors
        preamble = (f"The following extracts have been taken from the annual reporting of an energy "
                    f"company and all relate to {term.prompt_alias}.")
    else:
        preamble = (f"The following extracts have been taken from {co_alias}'s {year} {report_alias} "
                    f"and all relate to {term.prompt_alias}.")

    if schema_version == 1:
        task = _prompt_task_v1(term, co_alias=co_alias, co_ref=co_ref, year=year, mask=mask)
    else:
        task = _prompt_task_v2(term, co_ref=co_ref, year=year, mask=mask)

    prompt = f'''{preamble}

{task}

# Extracts:
-----------
{chunks_str}'''

    logger_llm.debug(f'PROMPT CONSTRUCTED: {co_alias}_{report_alias}_{year}_{term.prompt_fname.replace(".txt", "")} - schema_version={schema_version}, overlap_strategy={overlap_strategy}, overlap_amount={overlap_amount}, mask={mask}, extracts={n_used}/{n_available}')
    if save:
        fname = os.path.basename(save_path)
        if os.path.exists(save_path) and not overwrite:
            logger_llm.debug(f'PROMPT NOT SAVED: {fname} overwrite=False, txt present at {save_path}')
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(prompt)
            logger_llm.debug(f'PROMPT SAVED: {fname} at {save_path}')
    else:
        logger_llm.debug(f'PROMPT NOT SAVED: save=False.')

    return prompt



class AnalysisRecord(BaseModel):
    company: str
    report_type: Literal["annual-report", "sustainability-report"]
    year: int
    term: str
    schema_version: int
    response: BaseModel  # PromptResponseVx

    @property
    def custom_id(self) -> str:
        return f"{self.company}_{self.report_type}_{self.year}_{self.term}".lower().replace(' ', '-').replace('(', '').replace(')', '')

def prompt_id(report:Report, term:str|Term) -> str:
    '''
    Create a unique ID for a prompt based on the company, report type, year, and term.
    
    Parameters
    ----------
    report : Report
        The Report object containing the metadata to use to create the prompt ID.
    term : str or Term
        The technology/term the extracts relate to, either as a string (e.g., "carbon capture") or as a Term object. 
        If a Term object is passed, the term's prompt_alias will be used in the ID.
    
    Returns
    -------
    str
        The generated prompt ID, in the format "{company}_{report_type}_{year}_{term}", with all values in lowercase and spaces replaced by hyphens.
    '''
    if isinstance(term, str):
        term_fname = term
    elif isinstance(term, Term):
        term_fname = term.prompt_fname.replace('.txt', '')
    else:
        raise ValueError(f"term should be either the name of a term (as a str), or a Term object. Got {type(term)}")
    
    return f"{report.company.folder_name}_{report.report_type}_{report.year}_{term_fname}".lower().replace(' ', '-').replace('(', '').replace(')', '')

def make_strict_schema(schema: dict) -> dict:
    """Recursively patch a Pydantic JSON schema for OpenAI strict mode."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            # Every property must appear in required
            schema["required"] = list(schema["properties"].keys())
            for value in schema["properties"].values():
                make_strict_schema(value)
    if "items" in schema:
        make_strict_schema(schema["items"])
    if "$defs" in schema:
        for value in schema["$defs"].values():
            make_strict_schema(value)
    return schema

def schema_depends_on_term(schema_version:int) -> bool:
    '''
    Check if the given schema version depends on a specific term (i.e., requires a Term object to build the schema).
    
    Parameters
    ----------
    schema_version : int
        The schema version to check. Must be one of RESPONSE_SCHEMAS.

    Returns
    -------
    bool
        True if the schema version depends on a specific term, False otherwise.

    Raises
    ------
    ValueError
        If the schema_version is not one of RESPONSE_SCHEMAS.
    '''
    if schema_version not in RESPONSE_SCHEMAS:
        raise ValueError(f'schema_version should be one of {list(RESPONSE_SCHEMAS.keys())}. Got {schema_version}')
    
    schema = RESPONSE_SCHEMAS[schema_version]
    return not issubclass(schema, BaseModel)  # if not a BaseModel, it must be built per term

def get_schema(schema_version:int, term:Term|None = None, prompt_path:str|None = None) -> type[BaseModel]:
    '''
    Get the Pydantic model for the given schema version, optionally built for a specific term.
    
    Parameters
    ----------
    schema_version : int
        The schema version to retrieve. Must be one of RESPONSE_SCHEMAS.
    term : Term | None, optional
        The term to build the schema for, if required by the schema version. If None (the default),
        the schema will be returned without being built for a specific term.
    prompt_path : str | None, optional
        The path to the prompt file, used to infer the term if it is not provided. Only required if 
        the schema version depends on the term (e.g., PromptResponseV2) and term is None. 
        If provided, the term will be inferred from the prompt file name.
        Must be in the format: `.../{term_collection}/prompts/{term_prompt_fname}` - for
        custom file locations, pass the Term object directly to `term` instead of relying on inference.

    Returns
    -------
    type[BaseModel]
        The Pydantic model corresponding to the specified schema version and term.

    Raises
    ------
    ValueError
        If the schema_version is not one of RESPONSE_SCHEMAS, or if a term is required but not provided.
    '''
    if schema_version not in RESPONSE_SCHEMAS:
        raise ValueError(f'schema_version should be one of {list(RESPONSE_SCHEMAS.keys())}. Got {schema_version}')
    
    schema = RESPONSE_SCHEMAS[schema_version]
    if not issubclass(schema, BaseModel):
        if term is None:
            # assume given file is in the expected location
            if prompt_path is None:
                raise ValueError('prompt_path is required when term is None')
            term_prompt_fname = os.path.basename(prompt_path)
            term_collection_name = os.path.join(term_prompt_fname, '..', '..')  # format is .../{term_collection}/prompts/{term_prompt_fname}
            try:
                term = TermCollection.from_folder(term_collection_name)[term_prompt_fname]
            except Exception as e:
                raise ValueError(f'Term required but could not be inferred from prompt file {prompt_path}. Pass a valid Term object to term. Error: {e}')
        try:
            schema = schema(term)  # v2 requires building per term
        except Exception as e:
            raise ValueError(f'Failed to build schema for term {term}. Ensure term is a Term object. Error: {e}')
    
    return schema


def analyse_one(
        prompt_file:str, 
        *, 
        schema_version:int = 1,
        term:Term|None = None,
        model:str = DEFAULT_MODEL, max_retries:int = 3,
        save_path:str|None = None, overwrite:bool = False
) -> AnalysisRecord:
    '''
    Analyse the given prompt using the specified model, and return the response.
    
    Parameters
    ----------
    prompt_file : str
        The path of the prompt file.
    schema_version : int, optional
        The schema version to use for the response. Defaults to 1 (PromptResponseV1). Must be one of RESPONSE_SCHEMAS.
    term : Term | None, optional
        The term the extracts relate to, as a Term object. 
        Only required if the schema version depends on the term (e.g., PromptResponseV2).
        If None (the default), the term will be inferred from the prompt file name.
    model : str, optional
        The model to use for analysis. Defaults to 'gpt-5.1'.
    max_retries : int, optional
        The maximum number of times to retry the analysis. Defaults to 3.
    save_path : str | None, optional
        The path where the analysis results json should be saved. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite an existing file at save_path. Defaults to False.

    Returns
    -------
    AnalysisRecord
        The analysis results, as an AnalysisRecord object.
    '''
    if schema_version not in RESPONSE_SCHEMAS:
        raise ValueError(f'schema_version should be one of {list(RESPONSE_SCHEMAS.keys())}. Got {schema_version}')

    load_dotenv()
    client = instructor.from_openai(OpenAI())

    if os.path.isfile(prompt_file):
        with open(prompt_file, 'r') as f:
            prompt_content = f.read()
    else:
        raise ValueError(f'prompt_file should be a path to a text file containing the prompt. Got {prompt_file}')

    schema = get_schema(schema_version=schema_version, term=term, prompt_path=prompt_file)

    response = client.chat.completions.create(  # chat completions only allows single prompts - fine for this application
        model=model,
        response_model=schema,
        messages=[{"role":"user", "content": prompt_content}],
        max_retries=max_retries
    )
    logger_llm.debug(f'ANALYSIS COMPLETE: {prompt_file} - model={model}, max_retries={max_retries}')
    # get info for response
    file_mdata = file_metadata(prompt_file)
    year = int(file_mdata['year'])
    term = os.path.basename(prompt_file).split('.')[0]  # get term from filename
    
    out = AnalysisRecord(
        company=file_mdata['company'],
        report_type=file_mdata['report_type'],
        year=year,
        term=term,
        schema_version=schema_version,
        response=response
    )

    if save_path is not None:
        fname = os.path.basename(save_path)
        if os.path.exists(save_path) and not overwrite:
            logger_llm.debug(f'ANALYSIS NOT SAVED: {fname} overwrite=False, file present at {save_path}')
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(out.model_dump_json(indent=4))
            logger_llm.debug(f'ANALYSIS SAVED: {fname} at {save_path}')
    
    return out
    
def create_openai_jsonl(
        prompt_paths:list[str],
        *,
        model:str = DEFAULT_MODEL,
        schema_version:int = 1,
        save:bool = True, save_as:str|None = None, 
        overwrite:bool = False,
        return_type:Literal['fname', 'path', 'data', None] = 'path'
    )-> list[dict]:
    '''
    Create a jsonl file containing the prompts and metadata for analysis with OpenAI's API, based on the given file paths or directories.

    Parameters
    ----------
    prompt_paths : list[str]
        A list of paths to prompt text files to include in the jsonl. Each file should be a text file containing a prompt.
    model : str, optional
        The model to use for analysis. Defaults to 'gpt-5.1'.
    schema_version : int, optional
        The schema version to use for the response. Defaults to 1 (PromptResponseV1). Must be one of RESPONSE_SCHEMAS.
    save : bool, optional
        Whether to save the jsonl file. Defaults to True.
    save_as : str | None, optional
        The path where the jsonl file should be saved. If None, a unique filename will be generated based on the prompt contents and saved in the prompts folder of the term collection. Only has
    save_path : str | None, optional
        The path where the jsonl file should be saved. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite an existing file at save_path. Defaults to False.
    
    Returns
    -------
    list[dict]
        A list of dictionaries, each containing the metadata and prompt content for a single prompt, ready to be passed to OpenAI's API. If 
        save_path is provided and overwrite is True, the list will also be saved as a jsonl file at the specified path.

    Raises
    ------
    ValueError
        If prompts from multiple term collections are provided, or if save_as is not a path ending with .jsonl.
    '''
    # preliminary checks to ensure we're not doing work for nothing
    if return_type not in ['fname', 'path', 'data', None]:
        raise ValueError(f'return_type should be one of "fname", "data", or None. Got {return_type}')
    if schema_version not in RESPONSE_SCHEMAS:
        raise ValueError(f'schema_version should be one of {list(RESPONSE_SCHEMAS.keys())}. Got {schema_version}')

    if save:
        if save_as is None:
            term_coll = TermCollection.from_folder(os.path.dirname(os.path.dirname(prompt_paths[0])))  # use first prompt path for collection. if we have multiple later, we raise an error so we can assume this is correct.
            # make a unique filename based on the prompt contents
            fname = uuid.uuid5(uuid.NAMESPACE_OID, ''.join([open(p, 'r').read() for p in prompt_paths])).hex
            save_as = os.path.join(term_coll.prompts_folder, f'{fname}.jsonl')
        else:
            if not save_as.endswith('.jsonl'):
                raise ValueError(f'save_as should be a path ending with .jsonl. Got {save_as}')
            fname = os.path.basename(save_as)
        
        if os.path.exists(save_as) and not overwrite:
            logger_llm.debug(f'JSONL NOT SAVED: {fname} overwrite=False, file present at {save_as}')
            save = False
            if return_type is None:
                return None
            elif return_type == 'fname':
                return fname
            elif return_type == 'path':
                return save_as
            # only continue processing if we need to return data
    
    # validate file paths
    valid_file_paths = []
    for path in prompt_paths:
        if os.path.isfile(path) and path.endswith('.txt'):
            valid_file_paths.append(path)
        else:
            logger_llm.warning(f'JSONL - INVALID PROMPT PATH: {path}. Skipping.')
    prompt_paths = valid_file_paths

    # gather jsons in jsonl format
    data = []
    need_term = schema_depends_on_term(schema_version)
    if not need_term:
        schema = make_strict_schema(get_schema(schema_version=schema_version).model_json_schema())
    term_collections = set()
    for file_path in prompt_paths:
        with open(file_path, 'r') as f:
            prompt_content = f.read()
        term = os.path.basename(file_path).split('.')[0]  # get term from filename
        term_coll = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # get term collection from parent folder name
        term_collections.add(term_coll)
        if len(term_collections) > 1:
            raise ValueError(f'Multiple TermCollections found in prompt_paths: {term_collections}. All prompts in a single batch must relate to the same TermCollection.')
        
        if need_term:
            schema = make_strict_schema(
                get_schema(
                    schema_version=schema_version, 
                    term=TermCollection.from_folder(term_coll)[term], 
                    prompt_path=file_path
                ).model_json_schema()
            )
        data.append({
            'custom_id': prompt_id(Report(file_path), term),  # custom ID for OpenAI API
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': model,
                'messages': [{"role":"user", "content": prompt_content}],
                'response_format': {
                    'type': 'json_schema',
                    'json_schema': {
                        'name': schema.__name__,
                        'schema': schema,
                        'strict': True
                    }
                }
            }
        })
    logger_llm.debug(f'JSONL DATA CREATED: {len(data)} records created from {len(prompt_paths)} file paths. Schema version: {schema_version}.')

    # write jsonl file
    if save:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        with open(save_as, 'w') as f:
            for record in data:
                f.write(json.dumps(record) + '\n')
        logger_llm.debug(f'JSONL SAVED: {fname} at {save_as}')
    
    if return_type == 'data':
        return data
    elif return_type == 'fname':
        return fname
    elif return_type == 'path':
        return save_as
    else:
        return None

@dataclass
class JSONLPortion:
    # portion control is essential for a healthy diet.
    portion: list[dict]
    token_count: int
    start_index: int
    end_index: int  # portion = records[start_index:end_index] (exclusive of end_index)

    def __len__(self):
        return len(self.portion)

    def __getitem__(self, key):
        return self.portion[key]

    @property
    def custom_ids(self) -> list[str]:
        '''Ordered list of custom_ids of the prompts in this portion.'''
        return [record['custom_id'] for record in self.portion]

    def __repr__(self):
        return f'JSONLPortion(start_index={self.start_index}, end_index={self.end_index}, token_count={self.token_count}, portion_len={len(self.portion)})'

JOB_SCHEMA_VERSION = 2
JOB_STATUSES = ('created', 'working', 'completed', 'failed', 'part-failed-completed', 'part-failed-working')

# how many of the most recent batches to examine when checking for enqueued-token headroom
HEADROOM_CHECK_LIMIT = 100


def _resolve_prompt_jsonl(prompt_jsonl: str) -> str:
    '''Resolve a job ID, filename, or path to the path of an existing prompt jsonl file.'''
    for candidate in (prompt_jsonl, os.path.join(BASE_DIR, prompt_jsonl)):
        if candidate.endswith('.jsonl') and os.path.isfile(candidate):
            return candidate

    stem = prompt_jsonl
    if stem.endswith('.jsonl'):
        stem = stem[:-len('.jsonl')]
    if os.sep in stem:
        stem = os.path.basename(stem)

    for root, dirs, files in os.walk(TERMS_DIR):
        if f'{stem}.jsonl' in files:
            return os.path.join(root, f'{stem}.jsonl')
    raise FileNotFoundError(f'Prompt jsonl {prompt_jsonl} not found in {TERMS_DIR}. Cannot create Job.')


def _read_json(path: str) -> dict|None:
    '''Read a json file, returning None if it does not exist or cannot be parsed.'''
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger_llm.warning(f'JSON UNREADABLE: {path}. Error: {e}')
        return None


def _normalise_batch_id(batch_id: str) -> str:
    '''Strip path, extension and -info/-results decoration from a batch id.'''
    if os.sep in batch_id:
        batch_id = os.path.basename(batch_id)
    batch_id = batch_id.split('.')[0]
    for suffix in ('-info', '-results'):
        if batch_id.endswith(suffix):
            batch_id = batch_id[:-len(suffix)]
    return batch_id


def _iter_job_infos(terms: TermCollection|None = None) -> Iterator[tuple[str, dict]]:
    '''Yield (path, contents) for every job info file under TERMS_DIR.'''
    if terms is None:
        dir_to_search = TERMS_DIR
    else:
        if isinstance(terms, str):
            terms = TermCollection.from_folder(terms)
        dir_to_search = terms.folder
    for root, dirs, files in os.walk(dir_to_search):
        for file in sorted(files):
            if not file.endswith('-info.json'):
                continue
            if file.startswith('portions-info-') or file.endswith('.bak.json'):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, 'r') as f:
                    info = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger_llm.warning(f'JOB INFO UNREADABLE: {path}. Error: {e}. Skipping.')
                continue
            if 'parent' in info:
                yield path, info


def _claimed_batch_ids() -> set[str]:
    '''Every batch ID recorded against a job.'''
    claimed = set()
    for path, info in _iter_job_infos():
        for batch in info.get('batches') or []:
            if isinstance(batch, dict) and 'id' in batch:
                claimed.add(batch['id'])

    # legacy portions-info files - will be deprecated in future versions
    for root, dirs, files in os.walk(TERMS_DIR):
        for file in files:
            if file.startswith('portions-info-') and file.endswith('.json'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        legacy = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                legacy_batches = legacy.get('batches') or {}
                if isinstance(legacy_batches, dict):
                    claimed.update(legacy_batches.keys())
                else:
                    claimed.update(b['id'] for b in legacy_batches if isinstance(b, dict) and 'id' in b)
    return claimed


def _batch_id_from_filename(fname: str) -> str:
    stem = fname.split('.')[0]
    if stem.endswith('-results'):
        stem = stem[:-len('-results')]
    return stem


def _iter_unclaimed_batches(claimed: set[str]) -> Iterator[tuple[str, str|None, str|None]]:
    '''
    Yield (batch_id, absolute result path or None, absolute batch detail path or None) for every
    batch that hasnt been claimed by a job.
    '''
    found = {}
    for root, dirs, files in os.walk(TERMS_DIR):
        if os.path.basename(root) != 'batches':
            continue
        for file in files:
            if file.startswith('batch_') and file.endswith('.json'):
                batch_id = _batch_id_from_filename(file)
                result_path, _ = found.get(batch_id, (None, None))
                found[batch_id] = (result_path, os.path.join(root, file))

    if os.path.isdir(OUT_DIR):
        for root, dirs, files in os.walk(OUT_DIR):
            if os.path.basename(root) != 'batch-results':
                continue
            for file in files:
                if file.startswith('batch_') and file.endswith('.jsonl'):
                    batch_id = _batch_id_from_filename(file)
                    _, detail_path = found.get(batch_id, (None, None))
                    found[batch_id] = (os.path.join(root, file), detail_path)

    for batch_id, (result_path, detail_path) in sorted(found.items()):
        if batch_id not in claimed:
            yield batch_id, result_path, detail_path


def _batch_created_at(batch: dict) -> float|None:
    '''When a batch was created. Either taken from the job's own record or the info from the API.'''
    if batch.get('submitted_at'):
        return batch['submitted_at']
    info = batch.get('info') or {}
    return info.get('created_at')


def _find_result_file(batch_id: str, terms_folder: str) -> str|None:
    '''
    Locate a batch's result file under out/<terms_folder>/batch-results/.
    Returns a path relative to the repository root, or None if there is no result file.
    '''
    results_folder = os.path.join(OUT_DIR, terms_folder, 'batch-results')
    if not os.path.isdir(results_folder):
        return None
    for fname in sorted(os.listdir(results_folder)):
        if fname.startswith(batch_id) and fname.endswith('.jsonl'):
            return os.path.relpath(os.path.join(results_folder, fname), BASE_DIR)
    return None


def _custom_ids_in_result_file(path: str) -> set[str]:
    '''The custom_ids present in a batch result file.'''
    custom_ids = set()
    try:
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if 'custom_id' in record:
                    custom_ids.add(record['custom_id'])
    except (OSError, json.JSONDecodeError) as e:
        logger_llm.warning(f'BATCH RESULT FILE UNREADABLE: {path}. Error: {e}')
    return custom_ids


def _custom_ids_from_api_input(batch_id: str) -> set[str]:
    '''Download a batch's input file from the API and collect the custom_ids it requested.'''
    load_dotenv()
    client = OpenAI()
    try:
        batch = client.batches.retrieve(batch_id)
        if batch.input_file_id is None:
            return set()
        content = client.files.content(batch.input_file_id)
    except NotFoundError as e:
        logger_llm.warning(f'BATCH INPUT UNAVAILABLE: {batch_id}. Error: {e}')
        return set()

    custom_ids = set()
    for line in content.iter_lines():
        if line.strip():
            custom_ids.add(json.loads(line)['custom_id'])
    return custom_ids


def _batch_error_code(batch: dict) -> str|None:
    '''The first error code the API reported for a batch, if any.'''
    info = batch.get('info') or {}
    errors = info.get('errors') or {}
    data = errors.get('data') or []
    if data and isinstance(data[0], dict):
        return data[0].get('code')
    return None


class Job:
    '''
    A job is a collection of prompts to be processed by a LLM, with the prompts stored in jsonl format.

    A job can contain prompts from multiple companies and report types, but all prompts must relate to the same term collection.
    The job is identified by the filename of its jsonl file, which is stored in `data/terms/<Job term collection>/prompts/<Job id>.jsonl`.
    The job's info file is stored in the same folder (in `<Job id>-info.json`), and contains metadata about the job, including the number of prompts,
    the term collection ID, the companies involved, the report types, and info on any created batches.

    Coverage of a Job is defined by the prompt `custom_id`s present in the job's jsonl file. Note this means
    Job coverage is only sensitive to the *companies*, *terms*, *years*, and *report types* of its prompts rather than the prompts contents.
    This is in contrast to the Job ID, which is derived from the prompt contents.
    '''

    def __init__(self, prompt_jsonl: str):
        '''
        A job is a collection of prompts to be processed by a LLM, with the prompts stored in jsonl format.

        Parameters
        ----------
        prompt_jsonl : str
            Either the job ID (filename of the jsonl file without the .jsonl extension), or the full path to the jsonl file.

        Raises
        ------
        FileNotFoundError
            If the jsonl file does not exist at the specified path.
        '''
        prompt_jsonl_path = _resolve_prompt_jsonl(prompt_jsonl)

        self.prompt_jsonl_path = os.path.relpath(prompt_jsonl_path, BASE_DIR)
        self.input_path = self.prompt_jsonl_path  # alias for consistency with output_paths property
        self.id = os.path.basename(prompt_jsonl_path).replace('.jsonl', '')
        self.info_path = os.path.join(os.path.dirname(prompt_jsonl_path), f'{self.id}-info.json')

        # read schema version from first entry in jsonl - all prompts have the same version
        with open(prompt_jsonl_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError(f'Prompt jsonl {prompt_jsonl_path} is empty. Cannot create Job.')
            try:
                first_record = json.loads(first_line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Prompt jsonl {prompt_jsonl_path} is not valid JSON. Cannot create Job. Error: {e}')
            body = first_record.get('body') or {}
            response_format = body.get('response_format') or {}
            schema_info = response_format.get('json_schema') or {}
            self.response_schema_version = int(schema_info.get('name', '').replace('PromptResponseV', '')[0]) if 'PromptResponseV' in schema_info.get('name', '') else 1
        

        # caches, invalidated by _write_info / refresh
        self._info_cache = None
        self._info_mtime = None
        self._prompt_jsonl_cache = None
        self._completed_cache = None
        self._expected_cache = None

        self.n_prompts = len(self.prompt_jsonl)

        raw_info = _read_json(self.info_path)
        if raw_info is None:
            # handle legacy portions-info file if one exists - will be deprecated in future versions
            raw_info = _read_json(self._legacy_info_path)

        # metadata must be resolved before migration, which rebuilds the info file around it
        self.terms = self._resolve_terms((raw_info or {}).get('terms_id'), prompt_jsonl_path)
        recorded_companies = (raw_info or {}).get('companies')
        self.companies = CompanyCollection(sorted(recorded_companies)) if recorded_companies else self._find_companies()
        recorded_report_types = (raw_info or {}).get('report_types')
        self.report_types = set(recorded_report_types) if recorded_report_types else self._find_report_types()

        info = self._new_info() if raw_info is None else self._migrate_info(raw_info)
        if info is not raw_info:  # _migrate_info returns the same object when nothing needed doing
            self._write_info(info)
        self.created_at = info.get('created_at')  # for checking job creation time without loading info into cache
        self._update_job_status('auto')

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_prompts(cls, prompt_paths: Iterable[str], *,
                     model: str = DEFAULT_MODEL,
                     overwrite: bool = False) -> 'Job':
        '''
        Create a Job from a collection of prompt .txt files, building the input jsonl if it does not
        already exist. The new Job ID is derived from the prompt contents, so jobs containing the same
        prompts will have the same ID.

        Parameters
        ----------
        prompt_paths : Iterable[str]
            Paths to the prompt .txt files to include in the job. All must belong to the same TermCollection.
        model : str, optional
            The model to request in each prompt body. Defaults to DEFAULT_MODEL.
        overwrite : bool, optional
            Whether to rewrite the jsonl if it already exists. Defaults to False.

        Returns
        -------
        Job
        '''
        prompt_paths = list(prompt_paths)
        if not prompt_paths:
            raise ValueError('prompt_paths is empty. Cannot create a Job with no prompts.')

        # infer schema version, and ensure all prompts are for the same schema
        schema_versions = set(int(os.path.basename(fpath).split('_v')[1]) for fpath in prompt_paths)
        if len(schema_versions) != 1:
            raise ValueError('All prompt files must be for the same schema version. Got versions: ' + ', '.join(schema_versions))

        jsonl_path = create_openai_jsonl(
            prompt_paths=prompt_paths,
            model=model,
            schema_version=schema_versions.pop(),  # only one val
            save=True, overwrite=overwrite,
            return_type='path'
        )
        job = cls(jsonl_path)
        if job.info.get('model') != model:
            info = job.info
            info['model'] = model
            job._write_info(info)
        return job

    @classmethod
    def from_batch_id(cls, batch_id: str) -> 'Job':
        '''
        Create a Job object from a batch ID, by searching the TERMS_DIR for the
        Job info file containing the batch ID.

        Parameters
        ----------
        batch_id : str
            The batch ID to search for. Accepts path/filename of the batch info/results json/jsonl file, so long
            as the filename is in the format {batch_id}-info.json or {batch_id}-results.jsonl.

        Returns
        -------
        Job
        '''
        batch_id = _normalise_batch_id(batch_id)

        for info_path, info in _iter_job_infos():
            for batch in info.get('batches', []):
                if batch['id'] == batch_id:
                    return cls(info['parent'])

        # handle legacy file if exists - will be deprecated in future versions
        for root, dirs, files in os.walk(TERMS_DIR):
            for file in files:
                if file.startswith('portions-info-') and file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        legacy_info = json.load(f)
                    batches = legacy_info.get('batches') or {}
                    if batch_id in batches:
                        parent = legacy_info.get('parent') or legacy_info.get('metadata', {}).get('parent')
                        if parent:
                            return cls(parent)
        raise ValueError(f"Batch with ID {batch_id} not found.")

    @classmethod
    def list_all(cls, terms: TermCollection|None = None) -> list['Job']:
        '''Return a Job for every job info file found under TERMS_DIR or the specified term collection.'''
        jobs = []
        for info_path, info in _iter_job_infos(terms=terms):
            try:
                jobs.append(cls(info['parent']))
            except FileNotFoundError:
                logger_llm.warning(f'JOB INPUT MISSING: {info.get("parent")} referenced by {info_path}. Skipping.')
        return jobs

    def merge(self, *others: 'Job', save_as: str|None = None, overwrite: bool = False) -> 'Job':
        '''
        Combine this job with one or more others into a single new job.

        The merged job's input jsonl is the union of the sources' prompts, de-duplicated by
        custom_id and keeping the order they are encountered (this job's prompts first). Its ID is
        derived from the prompt contents in the same way `from_prompts` derives one, so merging the
        same jobs twice returns the same merged job.

        Source jobs are left unmodified.

        Parameters
        ----------
        *others : Job
            The job(s) to merge into this one. All must share this job's term collection and response schema version.
        save_as : str or None, optional
            Where to write the merged jsonl. Defaults to a content-derived name in the term
            collection's prompts folder.
        overwrite : bool, optional
            Whether to rewrite the merged jsonl if it already exists. Defaults to False.

        Returns
        -------
        Job
            The merged job.

        Raises
        ------
        ValueError
            If the jobs do not all share the same term collection and response schema version.
        ValueError
            If no other jobs are provided.
        ValueError
            If save_as is provided but does not end with .jsonl.
        '''
        jobs = [self, *others]
        if len(jobs) < 2:
            raise ValueError('merge needs at least one other job.')

        term_folders = {job.terms.folder_name for job in jobs}
        if len(term_folders) > 1:
            raise ValueError(
                f'Cannot merge jobs from different term collections: {sorted(term_folders)}. '
                'All prompts in a job must relate to the same TermCollection.'
            )
        response_schema_versions = {job.response_schema_version for job in jobs}
        if len(response_schema_versions) > 1:
            raise ValueError(
                f'Cannot merge jobs with different response schema versions: {sorted(response_schema_versions)}. '
                'All prompts in a job must relate to the same response schema version.'
            )

        records = []
        seen = set()
        for job in jobs:
            for record in job.prompt_jsonl:
                if record['custom_id'] in seen:
                    continue
                seen.add(record['custom_id'])
                records.append(record)

        if save_as is None:
            # same scheme as create_openai_jsonl: hash the prompt contents, in order
            fname = uuid.uuid5(
                uuid.NAMESPACE_OID,
                ''.join(r.get('body', {}).get('messages', [{}])[0].get('content', '') for r in records)
            ).hex
            save_as = os.path.join(self.terms.prompts_folder, f'{fname}.jsonl')
        elif not save_as.endswith('.jsonl'):
            raise ValueError(f'save_as should be a path ending with .jsonl. Got {save_as}')

        if os.path.exists(save_as) and not overwrite:
            logger_llm.debug(f'MERGED JSONL NOT WRITTEN: {os.path.basename(save_as)} overwrite=False, file present at {save_as}')
        else:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
            logger_llm.debug(f'MERGED JSONL SAVED: {os.path.basename(save_as)} at {save_as}, {len(records)} prompts from {len(jobs)} jobs.')

        merged = Job(save_as)

        # carry over every source batch, so the merged job already knows where the results are
        for job in jobs:
            for batch in job.batches:
                merged._update_batch({**batch, 'merged_from': job.id})

        info = merged.info
        info['merged_from'] = sorted({job.id for job in jobs})
        merged._write_info(info)

        merged._resolve_duplicate_coverage()
        merged._find_batch_results()
        merged._update_job_status('auto')

        logger_llm.info(
            f'JOBS MERGED: {", ".join(job.id for job in jobs)} -> {merged.id}. '
            f'{len(records)} prompts, {len(merged.batches)} batches, {len(merged.completed_prompts)} with results.'
        )
        return merged

    def __add__(self, other: 'Job') -> 'Job':
        if not isinstance(other, Job):
            return NotImplemented
        return self.merge(other)

    def __len__(self):
        return self.n_prompts

    def __repr__(self):
        return f'Job(id={self.id}, terms={self.terms}, n_prompts={self.n_prompts}, created_at={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at))}, status={self.status})'

    # ------------------------------------------------------------------ #
    # info file
    # ------------------------------------------------------------------ #

    @property
    def _legacy_info_path(self) -> str:
        return os.path.join(os.path.dirname(resolve_path(self.prompt_jsonl_path)), f'portions-info-{self.id}.json')

    def _new_info(self) -> dict:
        now = time.time()
        return {
            'schema_version': JOB_SCHEMA_VERSION,
            'response_schema_version': self.response_schema_version,
            'id': self.id,
            'parent': self.prompt_jsonl_path,
            'n_prompts': self.n_prompts,
            'model': self._model_from_prompts(),
            'terms_id': self.terms.uuid,
            'terms_folder': self.terms.folder_name,
            'companies_id': self.companies.uuid,
            'companies': sorted(company.folder_name for company in self.companies.companies),
            'report_types': sorted(self.report_types),
            'status': 'created',  # one of 'created', 'working', 'completed', 'failed', 'part-failed-completed', 'part-failed-working'
            'created_at': now,
            'updated_at': now,
            'batches': [],
        }

    @property
    def info(self) -> dict:
        '''The job info file. Only re-read from disk when changed.'''
        mtime = os.path.getmtime(self.info_path) if os.path.exists(self.info_path) else None
        if self._info_cache is None or mtime != self._info_mtime:
            with open(self.info_path, 'r') as f:
                self._info_cache = json.load(f)
            self._info_mtime = mtime
        return self._info_cache

    def _write_info(self, info: dict):
        info['updated_at'] = time.time()
        with open(self.info_path, 'w') as f:
            json.dump(info, f, indent=4)
        self._info_cache = info
        self._info_mtime = os.path.getmtime(self.info_path)
        self._completed_cache = None

    def _resolve_terms(self, terms_id: str|None, prompt_jsonl_path: str) -> TermCollection:
        '''Resolve the job's TermCollection, preferring the recorded uuid over the folder layout.'''
        if terms_id:
            try:
                return TermCollection.from_uuid(terms_id)
            except (ValueError, FileNotFoundError):
                logger_llm.warning(f'TERM COLLECTION NOT FOUND BY UUID: {terms_id}. Falling back to folder structure.')
        return TermCollection.from_folder(os.path.dirname(os.path.dirname(prompt_jsonl_path)))

    def _model_from_prompts(self) -> str|None:
        for record in self.prompt_jsonl:
            model = record.get('body', {}).get('model')
            if model:
                return model
        return None

    def _migrate_info(self, info: dict) -> dict:
        '''
        Bring a job info dict up to JOB_SCHEMA_VERSION.

        Handles three inputs:
        - schema v2: returned unchanged.
        - schema v1: batches carry `indices` but no `custom_ids`. Indices are resolved against the
          input jsonl. Where `prompt_count` disagrees with `len(indices)`, `prompt_count` wins.
        - legacy `portions-info-<id>.json`: `batches` is a dict keyed by batch id, with inclusive
          `end_index`. Coverage is rebuilt from `start_index` and `prompt_count`.
        '''
        if info.get('schema_version') == JOB_SCHEMA_VERSION:
            return info

        custom_ids = self.custom_ids

        def indices_to_custom_ids(indices: Iterable[int], batch_id: str) -> list[str]:
            out = []
            for i in indices:
                if 0 <= i < len(custom_ids):
                    out.append(custom_ids[i])
                else:
                    logger_llm.warning(f'MIGRATION - INDEX OUT OF RANGE: index {i} for batch {batch_id} in job {self.id}. Skipping.')
            return out

        def coverage(entry: dict, batch_id: str, start: int|None) -> list[str]:
            '''Rebuild a batch's custom_id coverage from whatever the old entry recorded.'''
            if entry.get('custom_ids'):
                return list(entry['custom_ids'])
            indices = entry.get('indices')
            count = entry.get('prompt_count')
            if indices and count and len(indices) != count:
                # v1 off-by-one: trust prompt_count and re-expand from the first index
                indices = range(min(indices), min(indices) + count)
            elif indices is None and start is not None and count:
                indices = range(start, start + count)
            elif indices is None:
                logger_llm.warning(f'MIGRATION - NO COVERAGE INFO for batch {batch_id} in job {self.id}.')
                return []
            return indices_to_custom_ids(indices, batch_id)

        legacy_batches = info.get('batches')
        migrated_batches = []

        if isinstance(legacy_batches, dict):
            # legacy portions-info shape
            metadata = info.get('metadata', {})
            strategy = metadata.get('strategy', info.get('portioning_strategy'))
            strategy_param = metadata.get('strategy_param', info.get('portioning_param'))
            for batch_id, entry in legacy_batches.items():
                migrated_batches.append({
                    'id': batch_id,
                    'custom_ids': coverage(entry, batch_id, entry.get('start_index')),
                    'portioning_strategy': strategy,
                    'portioning_strategy_param': strategy_param,
                    'est_token_count': entry.get('token_count'),
                    'prompt_count': entry.get('prompt_count'),
                    'status': entry.get('status'),
                    'result_path': entry.get('result_path'),
                    'submitted_at': entry.get('submitted_at') or (entry.get('info') or {}).get('created_at'),
                    'attempt': 1,
                    'supersedes': None,
                    'superseded': False,
                    'adopted': False,
                    'info': entry.get('info'),
                })
        elif isinstance(legacy_batches, list):
            for entry in legacy_batches:
                migrated_batches.append({
                    'id': entry['id'],
                    'custom_ids': coverage(entry, entry['id'], None),
                    'portioning_strategy': entry.get('portioning_strategy'),
                    'portioning_strategy_param': entry.get('portioning_strategy_param'),
                    'est_token_count': entry.get('est_token_count'),
                    'prompt_count': entry.get('prompt_count'),
                    'status': entry.get('status'),
                    'result_path': entry.get('result_path'),
                    'submitted_at': entry.get('submitted_at') or (entry.get('info') or {}).get('created_at'),
                    'attempt': entry.get('attempt', 1),
                    'supersedes': entry.get('supersedes'),
                    'superseded': entry.get('superseded', False),
                    'adopted': entry.get('adopted', False),
                    'info': entry.get('info'),
                })

        migrated = self._new_info()
        # keep whatever the old file already knew
        for key in ('created_at', 'status'):
            if info.get(key) is not None:
                migrated[key] = info[key]
        migrated['batches'] = migrated_batches

        # keep a copy of what we replaced, in case the rebuilt coverage needs checking
        if os.path.exists(self.info_path):
            backup_path = self.info_path.replace('-info.json', f'-info.v{info.get("schema_version", 1)}.bak.json')
            if not os.path.exists(backup_path):
                with open(backup_path, 'w') as f:
                    json.dump(info, f, indent=4)
                logger_llm.debug(f'JOB INFO BACKED UP: {self.id} at {backup_path}')

        logger_llm.info(
            f'JOB INFO MIGRATED: {self.id} to schema v{JOB_SCHEMA_VERSION}. '
            f'{len(migrated_batches)} batches, covering {len(set().union(*[set(b["custom_ids"]) for b in migrated_batches]) if migrated_batches else set())}/{self.n_prompts} prompts.'
        )
        return migrated

    # ------------------------------------------------------------------ #
    # prompts
    # ------------------------------------------------------------------ #

    @property
    def prompt_jsonl(self) -> list[dict]:
        '''Sorted input jsonl contents. Cached after first read.'''
        if self._prompt_jsonl_cache is None:
            with open(resolve_path(self.prompt_jsonl_path), 'r') as f:
                self._prompt_jsonl_cache = [json.loads(line) for line in f if line.strip()]
        return self._prompt_jsonl_cache

    @property
    def custom_ids(self) -> list[str]:
        '''The custom_ids of every prompt in the job, in file order.'''
        return [record['custom_id'] for record in self.prompt_jsonl]

    def _find_companies(self) -> CompanyCollection:
        '''Return a CompanyCollection of every company referenced in the job's prompts.'''
        companies = {}
        for prompt in self.prompt_jsonl:
            company_name = prompt['custom_id'].split('_')[0]
            if company_name not in companies:
                companies[company_name] = Company.from_folder_name(company_name)
        return CompanyCollection([companies[name] for name in sorted(companies)])

    def _find_report_types(self) -> set[str]:
        return {prompt['custom_id'].split('_')[1] for prompt in self.prompt_jsonl}

    @property
    def stats(self) -> dict:
        '''Prompt and token statistics for the job's input jsonl.'''
        return jsonl_stats(resolve_path(self.prompt_jsonl_path))

    # ------------------------------------------------------------------ #
    # batches
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> str:
        '''Last recorded job status, as stored in the info file. Use `refresh()` to get up-to-date info from the API.'''
        return self.info['status']

    @property
    def batches(self) -> list[dict]:
        return self.info.get('batches', [])

    @property
    def active_batches(self) -> list[dict]:
        '''Batches that have not been superseded by a retry.'''
        return [b for b in self.batches if not b.get('superseded')]

    def batch_by_id(self, batch_id: str) -> dict|None:
        '''
        Get a batch by its ID, or None if not found.
        
        Parameters
        ----------
        batch_id : str
            The batch ID to look up.

        Returns
        -------
        dict | None
        '''
        for batch in self.batches:
            if batch['id'] == batch_id:
                return batch
        return None

    def batches_by_ids(self, batch_ids: Iterable[str]) -> list[dict]:
        '''
        Get a list of batches by their IDs. Batches not found are omitted.
        
        Parameters
        ----------
        batch_ids : Iterable[str]
            The batch IDs to look up.
        
        Returns
        -------
        list[dict]
        '''
        batch_ids = set(batch_ids)
        return [batch for batch in self.batches if batch['id'] in batch_ids]

    @property
    def batch_statuses_local(self) -> dict:
        '''{batch_id: status} as recorded in the info file.'''
        return {b['id']: b['status'] for b in self.batches}

    @property
    def batch_statuses_api(self) -> dict:
        '''{batch_id: status} as currently reported by the API.'''
        if not self.batches:
            return {}
        batch_ids = [b['id'] for b in self.batches]
        batch_info = check_batch_status(batch_ids=batch_ids, save=False, progress_bar=False)
        return {b['id']: b['status'] for b in batch_info}

    @property
    def n_batch_statuses(self) -> dict:
        '''
        {status: count} over every batch, superseded ones included. See `active_batches` for those
        that still count towards the job's status.
        '''
        statuses = [b['status'] for b in self.batches]
        return {s: statuses.count(s) for s in set(statuses)}

    @property
    def n_active_batch_statuses(self) -> dict:
        '''
        {status: count} over every batch that has not been superseded by a retry. See `batches` for
        the full list of batches.
        '''
        statuses = [b['status'] for b in self.active_batches]
        return {s: statuses.count(s) for s in set(statuses)}

    def batches_by_status(self, status: str|tuple) -> list[dict]:
        '''
        Return a list of batches with the given status.
        If a tuple of statuses is provided, return all batches with any of those statuses.

        Parameters
        ----------
        status : str | tuple
            The status or tuple of statuses to filter batches by. Any valid OpenAI batch status can be used.

        Returns
        -------
        list[dict]
            A list of batches with the given status(es).
        '''
        if isinstance(status, tuple):
            return [b for b in self.batches if b['status'] in status]
        return [b for b in self.batches if b['status'] == status]

    def _update_job_status(self, status: Literal['created', 'working', 'completed', 'failed', 'part-failed-completed', 'part-failed-working', 'auto'] = 'auto'):
        '''
        Update the job status in the info file.

        Parameters
        ----------
        status : Literal['created', 'working', 'completed', 'failed', 'part-failed-completed', 'part-failed-working', 'auto']
            The new status of the job. Should be one of 'created', 'working', 'completed', 'failed', 'part-failed-completed', or 'part-failed-working'.
            If 'auto' is provided, the status will be determined based on the statuses of the batches:
            - 'created': if no batches have been submitted yet
            - 'completed': if all batches are completed
            - 'failed': if all batches have failed
            - 'working': if any batch is working, and no batches have failed
            - 'part-failed-completed': if any batch has failed and all other batches are completed
            - 'part-failed-working': if any batch has failed and any batch is working
        '''
        status = status.lower()
        if status not in JOB_STATUSES + ('auto',):
            raise ValueError(f'status should be one of {", ".join(JOB_STATUSES)}, or "auto". Got {status}')
        if status == 'auto':
            statuses = [b['status'] for b in self.active_batches]
            if not statuses: 
                status = 'created'  # no batches submitted yet
            elif all(s in STATUS_COMPLETED for s in statuses):
                status = 'completed'  # all batches completed
            elif all(s in STATUS_FAILED for s in statuses):
                status = 'failed'  # all batches failed
            elif any(s in STATUS_WORKING for s in statuses) and not any(s in STATUS_FAILED for s in statuses):
                status = 'working'  # some batches working, none failed
            elif any(s in STATUS_FAILED for s in statuses) and any(s in STATUS_WORKING for s in statuses):
                status = 'part-failed-working'  # some batches failed, some working
            elif any(s in STATUS_FAILED for s in statuses):
                status = 'part-failed-completed'  # some batches failed, all others completed
            else:
                status = 'created'  # fallback, should not happen
        info = self.info
        if info.get('status') != status:
            info['status'] = status
            self._write_info(info)

    def _update_batch(self, batch_info: dict):
        '''
        Update the job info with the given batch information. If a batch with the same ID already
        exists, the given keys are merged into it, leaving keys it does not mention untouched.
        Otherwise the new batch is appended.

        Parameters
        ----------
        batch_info : dict
            A dictionary containing information about the batch, including at least an 'id' key.
        '''
        if not isinstance(batch_info, dict):
            raise ValueError(f'batch_info should be a dict. Got {type(batch_info)}')
        if 'id' not in batch_info:
            raise ValueError('batch_info must contain an "id" key.')

        info = self.info
        batches = info.setdefault('batches', [])
        for i, b in enumerate(batches):
            if b['id'] == batch_info['id']:
                batches[i] = {**b, **batch_info}
                break
        else:
            batches.append(batch_info)
        self._write_info(info)

    def _refresh_batch_statuses(self, batch_ids: Iterable[str]|None = None, *, progress_bar: bool = False):
        '''Pull live statuses from the API and merge them into the info file.'''
        if batch_ids is None:
            batch_ids = [b['id'] for b in self.batches]
        batch_ids = [batch_ids] if isinstance(batch_ids, str) else list(batch_ids)
        if not batch_ids:
            return
        for api_batch in check_batch_status(batch_ids=batch_ids, save=True, progress_bar=progress_bar):
            self._update_batch({'id': api_batch['id'], 'status': api_batch['status'], 'info': api_batch})

    # ------------------------------------------------------------------ #
    # coverage
    # ------------------------------------------------------------------ #

    @property
    def covered_prompts(self) -> set[str]:
        '''custom_ids claimed by a batch that has not failed or been superseded.'''
        covered = set()
        for batch in self.active_batches:
            if batch['status'] in STATUS_FAILED:
                continue
            covered.update(batch.get('custom_ids') or [])
        return covered

    @property
    def completed_prompts(self) -> set[str]:
        '''custom_ids for which a result has actually been written to disk, without an error.'''
        if self._completed_cache is None:
            completed = set()
            for path in self.output_paths:
                abs_path = resolve_path(path)
                if not os.path.isfile(abs_path):
                    continue
                with open(abs_path, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if record.get('error') is not None:
                            continue
                        status_code = (record.get('response') or {}).get('status_code')
                        if status_code is not None and status_code != 200:
                            continue
                        completed.add(record['custom_id'])
            self._completed_cache = completed
        return self._completed_cache

    @property
    def missing_prompts(self) -> list[str]:
        '''custom_ids with no successful result on disk, in input-file order.'''
        completed = self.completed_prompts
        return [cid for cid in self.custom_ids if cid not in completed]

    @property
    def expected_prompts(self) -> set[str]:
        '''
        Every custom_id drafted on disk for this job's term collection, across all companies,
        report types and years. Terms that matched no chunk in a report have no prompt file, and
        so are absent.
        '''
        if self._expected_cache is None:
            expected = set()
            for report in co_files(yield_level='Report', companies=self.companies, report_types=self.report_types):
                try:
                    paths = report.prompt_paths(self.terms)
                except ValueError:
                    continue  # no prompts drafted for this report and term collection
                for path in paths:
                    term_fname = os.path.basename(path).split('.')[0]
                    expected.add(prompt_id(report, term_fname))
            self._expected_cache = expected
        return self._expected_cache

    @property
    def prompts_comprehensive(self) -> bool:
        '''
        Whether the job's input jsonl contains every prompt in `expected_prompts`, whether or not
        those prompts have been submitted.
        '''
        return not (self.expected_prompts - set(self.custom_ids))

    @property
    def results_comprehensive(self) -> bool:
        '''
        Whether the job holds a successful result for every prompt in `expected_prompts`. False
        while any expected prompt is unsubmitted, still working, or failed.
        '''
        return not (self.expected_prompts - self.completed_prompts)

    def coverage(self) -> dict:
        '''
        Break down the job's completeness against `expected_prompts`.

        Returns
        -------
        dict
            With keys:
            - `prompts_comprehensive` : bool
                Whether every expected prompt is present in the input jsonl.
            - `results_comprehensive` : bool
                Whether every expected prompt has a successful result.
            - `n_expected`, `n_prompts_present`, `n_completed` : int
                Counts of expected prompts, of those present in the input, and of those with results.
            - `missing_from_input` : list[str]
                Sorted list of expected custom_ids not found in the input jsonl.
            - `missing_results` : list[str]
                Sorted list of expected custom_ids with no successful result. A superset of `missing_from_input`.
            - `missing_by_company`, `missing_by_report_type`, `missing_by_year`, `missing_by_term` : dict[str, int]
                Tallies of `missing_results`, in descending count order.
            - `unexpected` : list[str]
                custom_ids in the job with no corresponding prompt file on disk, sorted.
        '''
        expected = self.expected_prompts
        present = set(self.custom_ids)
        completed = self.completed_prompts

        missing_from_input = sorted(expected - present)
        missing_results = sorted(expected - completed)

        def tally(index: int) -> dict:
            '''Count missing results by the given field of the custom_id.'''
            counts = {}
            for custom_id in missing_results:
                parts = custom_id.split('_')
                key = parts[index] if len(parts) > index else 'unknown'
                counts[key] = counts.get(key, 0) + 1
            return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

        # custom_id is {company}_{report_type}_{year}_{term}
        return {
            'prompts_comprehensive': not missing_from_input,
            'results_comprehensive': not missing_results,
            'n_expected': len(expected),
            'n_prompts_present': len(expected & present),
            'n_completed': len(expected & completed),
            'missing_from_input': missing_from_input,
            'missing_results': missing_results,
            'missing_by_company': tally(0),
            'missing_by_report_type': tally(1),
            'missing_by_year': tally(2),
            'missing_by_term': tally(3),
            'unexpected': sorted(present - expected),
        }

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #

    @property
    def has_results(self) -> Literal['all', 'partial', 'none']:
        self._find_batch_results()  # ensure result paths are up to date
        batches = self.active_batches
        if not batches:
            return 'none'
        if all(b.get('result_path') is not None for b in batches):
            return 'all'
        if any(b.get('result_path') is not None for b in batches):
            return 'partial'
        return 'none'

    def batch_results_present(self, batch_ids: str|Iterable[str]|None = None) -> dict[str, bool]|bool:
        '''
        Check if the given batch IDs have results. If batch_ids is None, checks all batches in the job.

        Parameters
        ----------
        batch_ids : str | Iterable[str] | None
            The batch ID(s) to check. If None, checks all batches in the job.

        Returns
        -------
        dict[str, bool] | bool
            A dictionary mapping batch IDs to a boolean indicating whether they have results, or a single boolean if a single batch ID is provided.
        '''
        single = isinstance(batch_ids, str)
        self._find_batch_results(batch_ids=batch_ids)  # ensure result paths are up to date

        if batch_ids is None:
            batch_ids = [b['id'] for b in self.batches]
        elif single:
            batch_ids = [batch_ids]

        result = {}
        for batch_id in batch_ids:
            batch = self.batch_by_id(batch_id)
            if batch is None:
                raise ValueError(f'Batch with id {batch_id} not found in job {self.id}.')
            result[batch_id] = batch.get('result_path') is not None
        if single:
            return result[batch_ids[0]]
        return result

    def _find_batch_results(self, batch_ids: str|Iterable[str]|None = None) -> str|list[str|None]|None:
        '''
        Find the result file paths for the given batch IDs. By default returns the result_path
        stored in the info file, or if this is None (or the property is missing), attempts to locate the
        results file in the 'out/<self.terms.folder_name>/batch-results' folder, and records what it finds.

        If batch_ids is None, returns the result paths for all batches in the job.

        Parameters
        ----------
        batch_ids : str | Iterable[str] | None
            The batch ID(s) to find results for. If a single string is provided, returns a single result path.
            If an iterable of strings is provided, returns a list of result paths. If None (default), returns
            result paths for all batches in the job.

        Returns
        -------
        str | list[str | None] | None
            The result file paths for the given batch IDs. If a batch has no results, returns None for that batch.

        Raises
        ------
        ValueError
            If a batch ID is provided that does not exist in the job.
        '''
        single = isinstance(batch_ids, str)
        if single:
            batch_ids = [batch_ids]
        elif batch_ids is None:
            batch_ids = [b['id'] for b in self.batches]
        else:
            batch_ids = list(batch_ids)

        if not batch_ids:
            return None if single else []

        out = []
        for batch_id in batch_ids:
            batch = self.batch_by_id(batch_id)
            if batch is None:
                raise ValueError(f'Batch with id {batch_id} not found in job {self.id}.')
            result_path = batch.get('result_path')
            if result_path is None:
                result_path = _find_result_file(batch_id, self.terms.folder_name)
                if result_path is not None or 'result_path' not in batch:
                    # record what we found (or just the fact that we looked) to limit future searches
                    self._update_batch({'id': batch_id, 'result_path': result_path})
            out.append(result_path)

        if single:
            return out[0]
        return out

    @property
    def output_paths(self) -> list[str]:
        '''Paths to the result files of every batch in the job that has one.'''
        self._find_batch_results()
        return [b['result_path'] for b in self.active_batches if b.get('result_path') is not None]

    @property
    def outputs(self) -> list[dict]:
        '''Every result record across all of the job's result files.'''
        records = []
        for path in self.output_paths:
            with open(resolve_path(path), 'r') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        return records

    def to_df(self, **kwargs) -> pd.DataFrame:
        '''
        Build the analysis dataframe for this job. See `decarb.llm.results.to_df` for arguments.
        '''
        from decarb.llm import results  # imported here to avoid circular import
        return results.to_df(job=self, **kwargs)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def refresh(self, *, adopt_orphans: bool = True, progress_bar: bool = False) -> 'Job':
        '''
        Bring the job info file up to date by: 
        1. Getting latest batch statuses from the API
        2. Adopting any batches from this job that have become orphaned (if requested)
        3. Locating new/missing result files
        4. Recomputing the job status

        Parameters
        ----------
        adopt_orphans : bool, optional
            Whether to look for unclaimed batches belonging to this job. Defaults to True.
        progress_bar : bool, optional
            Whether to show a progress bar while checking batch statuses. Defaults to False.

        Returns
        -------
        Job
            self, to allow calls to be chained (updates made inplace).
        '''
        self._completed_cache = None
        self._expected_cache = None
        if self.batches:
            self._refresh_batch_statuses(progress_bar=progress_bar)
        if adopt_orphans:
            self.adopt_orphan_batches()
        self._find_batch_results()
        self._resolve_duplicate_coverage()
        self._update_job_status('auto')
        return self

    def adopt_orphan_batches(self, *, only_report: bool = False, use_api: bool = False,
                             since: float|None = None) -> list[str]:
        '''
        Find batches that belong to this job but are recorded in no job info file, and attach them.
        Batches created by older versions of the pipeline, or by a run interrupted before its info
        file was written, end up orphaned this way.

        For a batch to be adopted, it must meet three conditions:
        1. Every custom_id in the batch belongs to this job.
        2. It was created no earlier than the job's own first batch (ignored for new jobs - see `since`).
        3. It covers at least one prompt the job is currently missing.

        Parameters
        ----------
        only_report : bool, optional
            If True, report what would be adopted without modifying the info file. Defaults to False.
        use_api : bool, optional
            If True, batches with no result file on disk are matched by downloading their input
            file from the API. Defaults to False.
        since : float or None, optional
            Only consider batches created at or after this unix timestamp. If None (default), the
            creation time of the job's own earliest batch is used. Jobs with no batches of their own
            have no reference point, so condition 2 is ignored and this parameter has no effect.

        Returns
        -------
        list[str]
            The batch IDs that were (or would be) adopted.
        '''
        own_ids = set(self.custom_ids)
        still_missing = set(self.missing_prompts)
        claimed = _claimed_batch_ids()
        adopted = []

        if since is None:
            # Have to use creation time to differentiate as custom_id only identifies 
            # (company, report type, year, term), rather than the job that created it.
            own_times = [t for t in (_batch_created_at(b) for b in self.batches) if t]
            since = min(own_times) if own_times else None

        for batch_id, result_path, detail_path in _iter_unclaimed_batches(claimed):
            batch_info = _read_json(detail_path) if detail_path else None
            created_at = (batch_info or {}).get('created_at')
            if since is not None and created_at is not None and created_at < since:
                logger_llm.debug(f'ORPHAN BATCH SKIPPED: {batch_id} predates the first batch of job {self.id}.')
                continue

            if result_path is not None:
                batch_custom_ids = _custom_ids_in_result_file(result_path)
            elif use_api:
                batch_custom_ids = _custom_ids_from_api_input(batch_id)
            else:
                continue

            if not batch_custom_ids or not batch_custom_ids <= own_ids:
                continue
            fills = batch_custom_ids & still_missing
            if not fills:
                logger_llm.debug(f'ORPHAN BATCH SKIPPED: {batch_id} covers {len(batch_custom_ids)} prompts of job {self.id}, none of them missing.')
                continue

            adopted.append(batch_id)
            if only_report:
                still_missing -= fills
                continue

            if batch_info is None:
                batch_info = check_batch_status(batch_ids=batch_id, save=False, progress_bar=False)[0]
            ordered = [cid for cid in self.custom_ids if cid in batch_custom_ids]
            self._update_batch({
                'id': batch_id,
                'custom_ids': ordered,
                'portioning_strategy': None,
                'portioning_strategy_param': None,
                'est_token_count': None,
                'prompt_count': len(ordered),
                'status': batch_info.get('status'),
                'result_path': os.path.relpath(result_path, BASE_DIR) if result_path else None,
                'submitted_at': batch_info.get('created_at'),
                'attempt': 1,
                'supersedes': None,
                'superseded': False,
                'adopted': True,
                'info': batch_info,
            })
            still_missing -= fills
            logger_llm.info(f'ORPHAN BATCH ADOPTED: {batch_id} into job {self.id}, covering {len(ordered)} prompts ({len(fills)} previously missing).')

        if adopted and not only_report:
            self._completed_cache = None
            self._resolve_duplicate_coverage()
        return adopted

    def _resolve_duplicate_coverage(self):
        '''
        Mark any batch whose prompts are all accounted for elsewhere as superseded. Applies to 
        non-failed batches duplicated by newer ones, or failed batch whose prompts have been
        resubmitted in another batch. 
        
        Superseded batches are excluded from `output_paths` and from the job status.
        '''
        info = self.info
        changed = False
        completed = self.completed_prompts

        # newest first, so the first batch to claim a custom_id is the one we keep
        candidates = [b for b in info.get('batches', []) if not b.get('superseded')]
        live = sorted(
            [b for b in candidates if b['status'] not in STATUS_FAILED],
            key=lambda b: b.get('submitted_at') or 0,
            reverse=True
        )

        seen = set()
        for batch in live:
            custom_ids = set(batch.get('custom_ids') or [])
            if custom_ids and custom_ids <= seen:
                batch['superseded'] = True
                changed = True
                logger_llm.info(f'BATCH SUPERSEDED: {batch["id"]} in job {self.id} - all {len(custom_ids)} prompts already covered by a newer batch.')
                continue
            seen |= custom_ids

        for batch in candidates:
            if batch.get('superseded') or batch['status'] not in STATUS_FAILED:
                continue
            custom_ids = set(batch.get('custom_ids') or [])
            if custom_ids and custom_ids <= completed:
                batch['superseded'] = True
                changed = True
                logger_llm.info(f'FAILED BATCH RETIRED: {batch["id"]} in job {self.id} - all {len(custom_ids)} prompts have results from elsewhere.')

        if changed:
            self._write_info(info)

    def submit(self, *,
               model: str|None = None,
               portion_strategy: Literal['max_tokens', 'max_prompts', 'split_into_n'] = 'max_tokens',
               portion_max_tokens: int = 900_000, portion_max_prompts: int = 100, portion_split_into_n: int = 10,
               portion_encoding: str = 'o200k_base',
               portion_save: bool = False, portion_save_dir: str|None = None,
               only: Iterable[str]|None = None,
               enqueued_token_limit: int = 900_000,
               max_attempts: int = 3,
               overwrite: bool = False,
               progress_bar: bool = True,
               wait_interval: int = 120):
        '''
        Submit the job to the OpenAI API, splitting the prompts into portions according to the
        specified strategy and staying within the account's enqueued token limit.

        By default only prompts with no successful result on disk are submitted, so repeated calls
        will not duplicate work.

        Parameters
        ----------
        model : str or None, optional
            Recorded on the job for reference. The model actually used is the one baked into each
            prompt body when the jsonl was created. Defaults to None (leave as recorded).
        portion_strategy : Literal['max_tokens', 'max_prompts', 'split_into_n'], optional
            The strategy to use for portioning the prompts. Defaults to 'max_tokens'.
        portion_max_tokens : int, optional
            The maximum number of tokens that each portion should contain. Defaults to 900,000.
            Only has an effect if `portion_strategy` is 'max_tokens'.
        portion_max_prompts : int, optional
            The maximum number of prompts that each portion should contain. Defaults to 100.
            Only has an effect if `portion_strategy` is 'max_prompts'.
        portion_split_into_n : int, optional
            The number of portions to split the prompts into. Defaults to 10.
            Only has an effect if `portion_strategy` is 'split_into_n'.
        portion_encoding : str, optional
            The name of the encoding to use for token counting. Defaults to 'o200k_base'.
        portion_save : bool, optional
            Whether to save the portions to files. Defaults to False.
        portion_save_dir : str or None, optional
            The directory to save the portions to. If None, will save in the same directory as the
            jsonl file, in a folder named 'portions-<jsonl_filename>'. Defaults to None.
        only : Iterable[str] or None, optional
            The custom_ids to submit. If None (default), submits `missing_prompts`.
        enqueued_token_limit : int, optional
            The account's enqueued token limit for the model. Submission waits for headroom under
            this limit before sending each portion. Defaults to 900,000.
        max_attempts : int, optional
            How many times to resubmit a portion rejected with `token_limit_exceeded`. Defaults to 3.
        progress_bar : bool, optional
            Whether to display a progress bar. Defaults to True.
        wait_interval : int, optional
            Seconds between status checks while waiting. Defaults to 120.
        '''
        if portion_strategy == 'max_tokens' and portion_max_tokens > enqueued_token_limit:
            raise ValueError(
                f'portion_max_tokens ({portion_max_tokens:,}) exceeds enqueued_token_limit ({enqueued_token_limit:,}). '
                'Portions this large can never be accepted.'
            )

        self.refresh(progress_bar=False)

        if self.status in ('working', 'part-failed-working'):
            print(f'Job {self.id} still has batches in progress. Current status: {self.status}.')
            print(f'Batch statuses: {self.n_batch_statuses}')
            print('Wait for these to finish (or call retrieve(wait=True)) before submitting more.')
            return

        targets = list(only) if only is not None else self.missing_prompts
        if not targets:
            print(f'Job {self.id} has results for all {self.n_prompts} prompts. Nothing to submit.')
            self._update_job_status('auto')
            return

        if model is not None and self.info.get('model') != model:
            info = self.info
            info['model'] = model
            self._write_info(info)

        target_set = set(targets)
        records = [record for record in self.prompt_jsonl if record['custom_id'] in target_set]

        portion_strategy_param = {
            'max_tokens': portion_max_tokens,
            'max_prompts': portion_max_prompts,
            'split_into_n': portion_split_into_n,
        }[portion_strategy]

        if not progress_bar:
            print(f'Submitting {len(records)} of {self.n_prompts} prompts in portions using strategy {portion_strategy} with parameter {portion_strategy_param}.')

        with tqdm(total=len(records), desc="Submitting prompts", disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}') as pbar:
            for portion in jsonl_portions(
                jsonl=records,
                strategy=portion_strategy, max_tokens=portion_max_tokens, max_prompts=portion_max_prompts, split_into_n=portion_split_into_n,
                encoding_name=portion_encoding, save=portion_save, save_dir=portion_save_dir, overwrite=overwrite,
                parent=self.prompt_jsonl_path
            ):
                self._submit_portion(
                    portion,
                    portion_strategy=portion_strategy, portion_strategy_param=portion_strategy_param,
                    enqueued_token_limit=enqueued_token_limit, max_attempts=max_attempts,
                    progress_bar=progress_bar, wait_interval=wait_interval
                )
                pbar.update(len(portion))

        self._update_job_status('auto')
        print(f'Job {self.id} submitted. Current status: {self.status}. Batch statuses: {self.n_batch_statuses}.')

    def _submit_portion(self, portion: JSONLPortion, *,
                        portion_strategy: str, portion_strategy_param: int,
                        enqueued_token_limit: int, max_attempts: int,
                        progress_bar: bool, wait_interval: int,
                        supersedes: str|None = None) -> str|None:
        '''
        Submit a single portion, waiting for enqueued-token headroom first and resubmitting if the
        batch is rejected for exceeding the enqueued token limit.

        Returns the ID of the batch that ultimately carried the portion, or None if submission attempts failed.
        '''
        previous_id = supersedes
        for attempt in range(1, max_attempts + 1):
            self._await_headroom(
                portion.token_count,
                limit=enqueued_token_limit, wait_interval=wait_interval, progress_bar=progress_bar
            )

            batch = submit_openai_batch(
                jsonl=portion.portion,
                save=True, save_dir=self.terms.batches_folder,
                return_type='batch'
            )
            self._update_batch({
                'id': batch.id,
                'custom_ids': portion.custom_ids,
                'portioning_strategy': portion_strategy,
                'portioning_strategy_param': portion_strategy_param,
                'est_token_count': portion.token_count,
                'prompt_count': len(portion),
                'status': batch.status,
                'result_path': None,  # to be updated after retrieval
                'submitted_at': time.time(),
                'attempt': attempt,
                'supersedes': previous_id,
                'superseded': False,
                'adopted': False,
                'info': batch.model_dump(),
            })
            if previous_id is not None:
                self._update_batch({'id': previous_id, 'superseded': True})

            # wait for this batch before submitting the next portion
            out_path = retrieve_batch_results(
                batch_id=batch.id,
                save=True,
                if_incomplete='wait', wait_interval=wait_interval,
                if_failed='ignore', if_output_missing='ignore',
                return_type='path'
            )
            self._refresh_batch_statuses([batch.id])
            if out_path is not None:
                self._update_batch({'id': batch.id, 'result_path': os.path.relpath(out_path, BASE_DIR)})
            self._completed_cache = None

            current = self.batch_by_id(batch.id)
            if current['status'] not in STATUS_FAILED:
                return batch.id

            error_code = _batch_error_code(current)
            if error_code != 'token_limit_exceeded' or attempt == max_attempts:
                logger_llm.error(f'BATCH FAILED: {batch.id} in job {self.id} - status={current["status"]}, error={error_code}. Not retrying.')
                if not progress_bar:
                    print(f'Batch {batch.id} failed ({error_code or current["status"]}). Retry with retry_failed_batches().')
                return None

            logger_llm.warning(f'BATCH REJECTED (token_limit_exceeded): {batch.id} in job {self.id}. Attempt {attempt}/{max_attempts}, retrying.')
            if not progress_bar:
                print(f'Batch {batch.id} rejected for exceeding the enqueued token limit. Attempt {attempt}/{max_attempts}, retrying.')
            previous_id = batch.id
        return None

    def _await_headroom(self, need_tokens: int, *, limit: int, wait_interval: int, progress_bar: bool):
        '''
        Block until there is room under the enqueued token limit for a portion of `need_tokens`.

        Batches belonging to this job contribute their recorded `est_token_count` to the enqueued
        total, allowing the job to submit multiple portions in sequence without waiting for each one to finish.
        Estimates cannot be obtained for batches belonging to other jobs, so these jobs are allowed to finish
        before more submissions are made.

        # TODO: access other jobs token estimates via info file, to allow for more efficient submission.
        '''
        if need_tokens > limit:
            raise ValueError(f'Portion of {need_tokens:,} tokens can never fit under the enqueued token limit of {limit:,}.')

        own_estimates = {b['id']: (b.get('est_token_count') or 0) for b in self.batches}
        pbar = None
        try:
            while True:
                # only the most recent batches can still be in the queue, and this runs on a poll
                # loop, so cap the listing and do not rewrite every batch file on each check
                recent = check_batch_status(batch_ids=None, on_miss='pass', save=False, limit=HEADROOM_CHECK_LIMIT)
                in_queue = [b for b in recent if b['status'] in STATUS_IN_QUEUE]
                foreign = [b for b in in_queue if b['id'] not in own_estimates]
                own_tokens = sum(own_estimates.get(b['id'], 0) for b in in_queue if b['id'] in own_estimates)

                if not foreign and own_tokens + need_tokens <= limit:
                    return

                reason = (f'{len(foreign)} batch(es) from elsewhere still enqueued' if foreign
                          else f'{own_tokens:,} + {need_tokens:,} tokens exceeds the {limit:,} limit')
                logger_llm.info(f'WAITING FOR ENQUEUED HEADROOM: {reason}. Checking again in {wait_interval}s.')
                if pbar is None:
                    if progress_bar:
                        pbar = tqdm(total=None, desc='Waiting for enqueued token headroom', unit='check')
                    else:
                        print(f'Waiting for enqueued token headroom: {reason}.')
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({'in_queue': len(in_queue), 'own_tokens': f'{own_tokens:,}'})
                time.sleep(wait_interval)
        finally:
            if pbar is not None:
                pbar.close()

    def retrieve(self, batch_ids: Iterable[str]|str|None = None, *,
                 wait: bool = False, wait_interval: int = 120,
                 progress_bar: bool = True) -> list[str]:
        '''
        Retrieve results for the job's batches, saving them under out/<self.terms.folder_name>/batch-results/.

        Parameters
        ----------
        batch_ids : Iterable[str] | str | None, optional
            The batches to retrieve. If None (default), retrieves every batch that has no result file yet.
        wait : bool, optional
            Whether to wait for batches that are still in progress. Defaults to False.
        wait_interval : int, optional
            Seconds between status checks while waiting. Defaults to 120.
        progress_bar : bool, optional
            Whether to display a progress bar. Defaults to True.

        Returns
        -------
        list[str]
            The result file paths retrieved during this call.
        '''
        self._refresh_batch_statuses(batch_ids, progress_bar=progress_bar)

        if batch_ids is None:
            candidates = [b for b in self.active_batches if b.get('result_path') is None]
        else:
            batch_ids = [batch_ids] if isinstance(batch_ids, str) else list(batch_ids)
            candidates = self.batches_by_ids(batch_ids)

        retrieved = []
        for batch in tqdm(candidates, desc='Retrieving batch results', disable=not progress_bar, bar_format='{desc}: {bar} {n}/{total}'):
            if batch['status'] in STATUS_FAILED:
                continue
            if batch['status'] in STATUS_IN_QUEUE and not wait:
                logger_llm.info(f'BATCH STILL IN QUEUE: {batch["id"]} - status={batch["status"]}. Skipping (wait=False).')
                continue
            out_path = retrieve_batch_results(
                batch_id=batch['id'],
                save=True,
                if_incomplete='wait' if wait else 'ignore', wait_interval=wait_interval,
                if_failed='ignore', if_output_missing='ignore',
                return_type='path'
            )
            if out_path is not None:
                rel = os.path.relpath(out_path, BASE_DIR)
                self._update_batch({'id': batch['id'], 'result_path': rel})
                retrieved.append(rel)

        self._completed_cache = None
        self._update_job_status('auto')
        return retrieved

    def retry_failed_batches(self, batch_ids: Iterable[str]|None = None, **submit_kwargs):
        '''
        Retry the prompts covered by failed batches. The prompts from the selected batches are
        pooled and re-portioned, so a batch that failed because it was too large is broken up
        rather than resubmitted at the same size.

        Parameters
        ----------
        batch_ids : Iterable[str] | None, optional
            The batch IDs to retry. If None, retries all failed, cancelled or expired batches in the job.
        **submit_kwargs
            Passed through to `submit` (portion_strategy, portion_max_tokens, enqueued_token_limit, ...).
        '''
        if batch_ids is None:
            failed = self.batches_by_status(STATUS_FAILED)
        else:
            batch_ids = [batch_ids] if isinstance(batch_ids, str) else list(batch_ids)
            failed = []
            for batch_id in batch_ids:
                batch = self.batch_by_id(batch_id)
                if batch is None:
                    raise ValueError(f'Batch with id {batch_id} not found in job {self.id}.')
                if batch['status'] not in STATUS_FAILED:
                    raise ValueError(f'Batch with id {batch_id} is not failed. Current status: {batch["status"]}. Only failed, cancelled or expired batches can be retried.')
                failed.append(batch)

        if not failed:
            print(f'Job {self.id} has no failed batches to retry.')
            return

        completed = self.completed_prompts
        targets = set()
        for batch in failed:
            targets.update(cid for cid in (batch.get('custom_ids') or []) if cid not in completed)

        if not targets:
            print(f'Job {self.id}: the {len(failed)} failed batch(es) already have results from elsewhere. Nothing to retry.')
            for batch in failed:
                self._update_batch({'id': batch['id'], 'superseded': True})
            self._update_job_status('auto')
            return

        print(f'Retrying {len(targets)} prompts from {len(failed)} failed batch(es) in job {self.id}.')
        for batch in failed:
            self._update_batch({'id': batch['id'], 'superseded': True})

        ordered = [cid for cid in self.custom_ids if cid in targets]
        self.submit(only=ordered, **submit_kwargs)


def jsonl_portions(
        jsonl:str|list[dict], *,
        strategy:Literal['max_tokens', 'max_prompts', 'split_into_n'] = 'max_tokens',
        max_tokens:int = 1_000_000,
        max_prompts:int = 100,
        split_into_n:int = 10,
        encoding_name:str = 'o200k_base',
        save:bool = False, save_dir:str|None = None, overwrite:bool = False,
        parent:str|None = None
) -> Iterator[JSONLPortion]:
    '''
    Take a jsonl file and yield portions of it as lists of dictionaries, where each portion contains at 
    most `max_tokens` tokens in total.

    Parameters
    ----------
    jsonl : str
        The path to the jsonl file to portion, or a list of dictionaries representing the jsonl data.
    strategy : Literal['max_tokens', 'max_prompts', 'split_into_n'], optional
        The strategy to use for portioning the jsonl file. Defaults to 'max_tokens'.
    max_tokens : int, optional
        The maximum number of tokens that each portion should contain. Defaults to 1,000,000.
        Only has an effect if `strategy` is 'max_tokens'.
    max_prompts : int, optional
        The maximum number of prompts that each portion should contain. Defaults to 100.
        Only has an effect if `strategy` is 'max_prompts'.
    split_into_n : int, optional
        The number of portions to split the jsonl file into. Defaults to 10.
        Only has an effect if `strategy` is 'split_into_n'.
    encoding_name : str, optional
        The name of the encoding to use for token counting. Defaults to 'o200k_base'.
    save : bool, optional
        Whether to save the portions to files. Defaults to False.
    save_dir : str or None, optional
        The directory to save the portions to. If None, will save in the same directory as the jsonl file, 
        in a folder named 'portions-<jsonl_filename>'. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    parent : str or None, optional
        The parent job ID or path to the jsonl file. If provided, will be used to determine the save directory if `save_dir` is None. Defaults to None.

    Yields
    ------
    JSONLPortion
    '''
    if strategy not in ['max_tokens', 'max_prompts', 'split_into_n']:
        raise ValueError(f'strategy should be one of "max_tokens", "max_prompts", or "split_into_n". Got {strategy}')

    # normalise the input to a list of dicts up front, so every branch below can index and
    # re-iterate it freely (a file handle can only be consumed once, and a list is not a
    # context manager).
    if isinstance(jsonl, str):
        jsonl_path = jsonl
        with open(jsonl_path, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]
    elif isinstance(jsonl, list):
        records = list(jsonl)
    else:
        raise ValueError(f'jsonl should be a path to a jsonl file, or a list of dictionaries. Got {type(jsonl)}')

    if save:
        if save_dir is None:
            if isinstance(jsonl, str):
                pass
            elif parent is not None:
                if os.sep in parent:
                    parent = os.path.basename(parent)
                if not parent.endswith('.jsonl'):
                    parent = f'{parent}.jsonl'
                for root, dirs, files in os.walk(TERMS_DIR):
                    if parent in files:
                        jsonl_path = os.path.join(root, parent)
                        break
                else:
                    raise FileNotFoundError(f'Parent jsonl {parent} not found in {TERMS_DIR}. Cannot save portions.')
            else:
                raise ValueError('save_dir is None and parent is None. Cannot determine save directory for portions.')
            base_dir = os.path.dirname(jsonl_path)
            jsonl_fname = os.path.basename(jsonl_path).replace('.jsonl', '')
            save_dir = os.path.join(base_dir, f'portions-{jsonl_fname}')
        os.makedirs(save_dir, exist_ok=True)

    encoding = tiktoken.get_encoding(encoding_name)
    current_portion = []
    current_token_count = 0
    start_index = 0

    def count_tokens(record:dict) -> int:
        return len(encoding.encode(record.get('body', {}).get('messages', [{}])[0].get('content', '')))

    def save_portion_if_requested(portion:list[dict], i:int, requested = save):
        if requested:
            portion_fname = f'portion-{i}.jsonl'
            portion_path = os.path.join(save_dir, portion_fname)
            if os.path.exists(portion_path) and not overwrite:
                logger_llm.debug(f'PORTION NOT SAVED: {portion_fname} overwrite=False, file present at {portion_path}')
            else:
                with open(portion_path, 'w') as f:
                    for record in portion:
                        f.write(json.dumps(record) + '\n')
                logger_llm.debug(f'PORTION SAVED: {portion_fname} at {portion_path}')

    portion_counter = 0

    if strategy == 'split_into_n' or strategy == 'max_prompts':
        if strategy == 'split_into_n':
            total_lines = len(records)
            max_prompts = total_lines // split_into_n + (1 if total_lines % split_into_n > 0 else 0)

        for i, prompt_json in enumerate(records):
            current_portion.append(prompt_json)
            if len(current_portion) >= max_prompts:
                yield JSONLPortion(
                    portion=current_portion,
                    token_count=sum(count_tokens(p) for p in current_portion),
                    start_index=start_index,
                    end_index=i + 1  # exclusive
                )
                save_portion_if_requested(current_portion, portion_counter, requested = save)
                portion_counter += 1

                current_portion = []
                start_index = i + 1

        # yield any remaining portion
        if current_portion:
            yield JSONLPortion(
                portion=current_portion,
                token_count=sum(count_tokens(p) for p in current_portion),
                start_index=start_index,
                end_index=len(records)  # exclusive
            )
            save_portion_if_requested(current_portion, portion_counter, requested = save)
    else:
        for i, prompt_json in enumerate(records):
            token_count = count_tokens(prompt_json)

            if current_token_count + token_count > max_tokens and current_portion:
                yield JSONLPortion(
                    portion=current_portion,
                    token_count=current_token_count,
                    start_index=start_index,
                    end_index=i  # exclusive - this record starts the next portion
                )
                save_portion_if_requested(current_portion, portion_counter, requested = save)
                portion_counter += 1

                # (re)set for next portion
                current_token_count = 0
                start_index = i
                current_portion = []
            current_portion.append(prompt_json)
            current_token_count += token_count

        # yield any remaining portion
        if current_portion:
            yield JSONLPortion(
                portion=current_portion,
                token_count=current_token_count,
                start_index=start_index,
                end_index=len(records)  # exclusive
            )
            save_portion_if_requested(current_portion, portion_counter, requested = save)

def jsonl_stats(jsonl_path:str, *, encoding_name:str = 'o200k_base') -> dict:
    '''
    Get statistics about the given jsonl file, including the number of prompts and the total number of tokens.

    Parameters
    ----------
    jsonl_path : str
        The path to the jsonl file to get statistics for.
    encoding_name : str, optional
        The name of the encoding to use for token counting. Defaults to 'o200k_base'.

    Returns
    -------
    dict
        A dictionary containing the number of prompts and the total number of tokens in the jsonl file.
    '''
    encoding = tiktoken.get_encoding(encoding_name)
    n_prompts = 0
    n_tokens = 0
    token_counts = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            prompt_json = json.loads(line)
            n_prompts += 1
            n_tokens += len(encoding.encode(prompt_json.get('body', {}).get('messages', [{}])[0].get('content', '')))
            token_counts.append(len(encoding.encode(prompt_json.get('body', {}).get('messages', [{}])[0].get('content', ''))))

    if token_counts:
        return {
            'n_prompts': n_prompts,
            'n_tokens': n_tokens,
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'mean_tokens': sum(token_counts) / len(token_counts),
            'median_tokens': sorted(token_counts)[len(token_counts) // 2]
        }
    else:
        return {
            'n_prompts': n_prompts,
            'n_tokens': n_tokens,
            'min_tokens': 0,
            'max_tokens': 0,
            'mean_tokens': 0,
            'median_tokens': 0
        }

def submit_openai_batch(
        jsonl:str|list[dict], *,
        save_dir:str|None = None,
        save: bool = True, return_type: Literal['id', 'batch', 'none'] = 'id'
) -> str | Batch | None:
    '''
    Submit a batch of prompts for analysis by OpenAI models. Returns the batch ID (`batch_id`), and saves the batch
    details to `batch_id`.json in `save_dir` (if provided, or in the 'batches' folder in the relevant TermCollection 
    directory if not).
    
    Parameters
    ----------
    jsonl_path : str or list[dict]
        The path to the JSONL data to submit to OpenAI's API, or a list of dictionaries containing the data to submit. 
        If a list of dicts is provided, `save_dir` must be provided.
    save_dir : str or None
        The directory to save the batch details json to. Will be created if it does not exist. If None (the default), 
        will be inferred from the path provided to `jsonl`. If `jsonl` is provided as a list of 
        dicts, this argument must be provided.
    save : bool, optional
        Whether to save the batch details json to `save_dir`. Defaults to True.
    return_type : Literal['id', 'batch', 'none'], optional
        What to return from the function. Defaults to 'id'.
        - `'id'`: return the ID of the submitted batch (default)
        - `'batch'`: return the batch object itself
        - `'none'`: return None
    
    Returns
    -------
    str | Batch | None
        The ID of the submitted batch, the batch object itself, or None based on the `return_type` parameter.
    '''
    if isinstance(jsonl, str):
        if not os.path.isfile(jsonl):
            raise ValueError(f'jsonl should be a path to a jsonl file containing the prompts. Got {jsonl}')
    elif isinstance(jsonl, list):
        # validate list of dicts
        for record in jsonl:
            if not isinstance(record, dict):
                raise ValueError(f'jsonl list should contain dictionaries. Got {type(record)}')
    else:
        raise ValueError(f'jsonl should be either a path to a jsonl file, or a list of dictionaries containing the data. Got {type(jsonl)}')

    return_type = return_type.lower()
    if return_type not in ['id', 'batch', 'none']:
        raise ValueError(f'return_type should be one of "id", "batch", or "none". Got {return_type}')
    # determine term collection for save location
    if save:
        if save_dir is None:
            if isinstance(jsonl, str):
                terms = os.path.basename(os.path.dirname(os.path.dirname(jsonl)))  # get term collection from parent folder name
                if not os.path.exists(os.path.join(TERMS_DIR, terms)):
                    raise ValueError(f'TermCollection "{terms}" inferred from jsonl path {jsonl} does not exist in TERMS_DIR. Check the path or specify save_dir explicitly.')
                save_dir = os.path.join(TERMS_DIR, terms, 'batches')
            else:
                raise ValueError(f'If jsonl is provided as a list of dicts, save_dir must be specified to determine where to save batch details. Got jsonl as list and save_dir={save_dir}')
        elif not os.path.isdir(save_dir):
                raise ValueError(f'save_dir should be a path to a directory where batch details can be saved. Got {save_dir}')

    # set up client
    load_dotenv()
    client = OpenAI()

    if isinstance(jsonl, list):
        jsonl_bytes = io.BytesIO("\n".join(json.dumps(record) for record in jsonl).encode())

        batch_file = client.files.create(
            file=('jsonl_data.jsonl', jsonl_bytes, "application/json"),
            purpose="batch"
        )
        logger_llm.debug(f'JSONL PORTION UPLOADED -> OpenAI: with ID {batch_file.id}. From list, with len: {len(jsonl)}')
    else:
        batch_file = client.files.create(
            file=open(jsonl, "rb"),
            purpose="batch"
        )
        logger_llm.debug(f'JSONL UPLOADED -> OpenAI: {os.path.basename(jsonl)} with ID {batch_file.id}. From file, with path: {jsonl}')

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={}
    )
    logger_llm.debug(f'OPENAI BATCH CREATED: {batch.id} for file {batch_file.id}.')

    # save
    if save:
        os.makedirs(save_dir, exist_ok=True)
        batch_save_path = os.path.join(save_dir, f'{batch.id}.json')
        with open(batch_save_path, 'w') as f:
            f.write(json.dumps(batch.model_dump(), indent=4))
        logger_llm.debug(f'BATCH DETAILS SAVED: {batch.id} at {batch_save_path}')

    if return_type == 'id':
        return batch.id
    elif return_type == 'batch':
        return batch
    elif return_type == 'none':
        return None


def check_batch_status(batch_ids:Iterable[str] | str | None = None, *, 
                       on_miss: Literal['raise', 'print', 'pass', 'create'] = 'raise',
                       progress_bar: bool = False,
                       save: bool = True,
                       limit: int|None = None) -> list[dict]:
    '''
    Check the status of all batches, or the requested batch, and update the saved batch details file.

    Parameters
    ----------
    batch_ids : Iterable[str] | str | None
        The ID(s) of the batch(es) to check (also the name of the batch json file), as returned by submit_openai_batch.
        If None (default), will check the status of every batch on the account.
    on_miss : Literal['raise', 'print', 'pass', 'create'], optional
        What to do if the batch details file is not found.
        - `'raise'`: raise a FileNotFoundError (default)
        - `'print'`: print a warning message and continue
        - `'pass'`: silently continue
        - `'create'`: create a new batch details file with the current batch status
    progress_bar : bool, optional
        Whether to show a progress bar when checking the status of multiple batches. Defaults to False.
    save : bool, optional
        Whether to save the updated batch details to the batch json file. Defaults to True.
    limit : int | None, optional
        Only used when `batch_ids` is None. Caps the listing at the `limit` most recent batches,
        fetched as a single page. Defaults to None, which pages through the account's entire
        batch history.

    Returns
    -------
    list[dict]
        The updated batch details, as a list of dictionaries.
    '''
    load_dotenv()
    client = OpenAI()

    def update_batch_file(batch):
        batch_file_path = None
        for root, dirs, files in os.walk(TERMS_DIR):
            for file in files:
                if file == f'{batch.id}.json':
                    batch_file_path = os.path.join(root, file)
                    break
            if batch_file_path:
                break
        
        if batch_file_path:
            with open(batch_file_path, 'w') as f:
                f.write(json.dumps(batch.model_dump(), indent=4))
            logger_llm.debug(f'BATCH DETAILS UPDATED: {batch.id} at {batch_file_path}')
        else:
            logger_llm.warning(f'BATCH DETAILS FILE NOT FOUND: {batch.id}.json in directory {TERMS_DIR}. on_miss={on_miss}')
            if on_miss == 'raise':
                raise FileNotFoundError(f'Searched for batch file "{batch.id}.json" in directory {TERMS_DIR}, but found no match.')
            elif on_miss == 'print':
                print(f'Searched for batch file "{batch.id}.json" in directory {TERMS_DIR}, but found no match.')
            elif on_miss == 'pass':
                pass
            elif on_miss == 'create':
                batch_file_path = os.path.join(TERMS_DIR, 'unknown', 'batches', f'{batch.id}.json')
                os.makedirs(os.path.dirname(batch_file_path), exist_ok=True)
                with open(batch_file_path, 'w') as f:
                    f.write(json.dumps(batch.model_dump(), indent=4))
                logger_llm.debug(f'BATCH DETAILS CREATED: {batch.id} at {batch_file_path}')

    if batch_ids is None:
        if limit is not None:
            batches = list(client.batches.list(limit=limit).data)  # most recent n batches in account
        else:
            batches = list(client.batches.list())  # all batches in account
        for batch in batches:
            logger_llm.debug(f'BATCH STATUS CHECKED: {batch.id} - status={batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}, created at {batch.created_at}, updated at {time.time()}')
            if save:
                update_batch_file(batch)
        return [batch.model_dump() for batch in batches]
    else:
        if isinstance(batch_ids, str):
            batch_ids = [batch_ids]
        batches = []
        for batch_id in tqdm(batch_ids, desc="Checking batch statuses", disable = not progress_bar, bar_format='{desc}: {bar} {n}/{total}'):
            batch = client.batches.retrieve(batch_id)
            logger_llm.debug(f'BATCH STATUS CHECKED: {batch.id} - status={batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}, created at {batch.created_at}, updated at {time.time()}')
            if save:
                update_batch_file(batch)
            batches.append(batch)
        return [batch.model_dump() for batch in batches]

    
def retrieve_batch_results(batch_id:str, *, 
                       if_incomplete:Literal['error', 'ignore', 'wait'] = 'error',
                       if_failed:Literal['error', 'ignore'] = 'error',
                       if_output_missing:Literal['error', 'ignore'] = 'error',
                       save:bool = True, save_as:str = None,
                       wait_interval: int = 120,
                       return_type:Literal['contents', 'path'] = 'contents') -> list[dict]|None:
    '''
    Retrieve the results of the requested batch.

    Parameters
    ----------
    batch_id : str
        The ID of the batch to retrieve results for, as returned by submit_openai_batch.
    if_incomplete : Literal['error', 'ignore', 'wait'] = 'error'
        What to do if the batch is not completed yet. 
        - `'error'`: a ValueError will be raised. 
        - `'ignore'`: the function will return None and log a warning. 
        - `'wait'`: wait for the batch to complete before retrieving results. May take a long time (e.g. up to an hour)
    if_failed : Literal['error', 'ignore'] = 'error'
        What to do if the batch has failed.
        - `'error'`: a ValueError will be raised.
        - `'ignore'`: the function will return None and log a warning.
    if_output_missing : Literal['error', 'ignore'] = 'error'
        What to do if the output file for the batch is missing.
        - `'error'`: a ValueError will be raised.
        - `'ignore'`: the function will return None and log a warning.
    save : bool, optional
        Whether to save the retrieved results as a json file in the report's term directory (or at the path specified by `save_as` if provided). Defaults to True.
    save_as : str or None, optional
        The filepath to a jsonl file to save the retrieved results to. If None (the default), the results will be saved in the report's term directory. Only has an effect if `save` is True.
    wait_interval : int, optional
        The number of seconds to wait between checks for batch completion if `if_incomplete` is set to 'wait'. Defaults to 120 seconds (2 minutes).

    Returns
    -------
    list[dict]|None
        The batch results, as a list of dictionaries, or None if the batch is not completed.
    '''

    if return_type not in ['contents', 'path']:
        raise ValueError(f'return_type should be one of "contents" or "path". Got {return_type}')
    elif return_type == 'path' and not save:
        raise ValueError(f'return_type="path" requires save=True. Got save={save}')

    load_dotenv()
    client = OpenAI()
    
    batch = client.batches.retrieve(batch_id)
    
    if batch.status in STATUS_IN_QUEUE:  # i.e., still working, cancelling, finalizing, etc (requires waiting before next batch can start)
        if if_incomplete == 'error':
            raise ValueError(f'Batch {batch_id} is not yet complete. Current status: {batch.status}')
        elif if_incomplete == 'ignore':
            logger_llm.warning(f'ATTEMPTED RETRIEVAL OF INCOMPLETE BATCH {batch_id}. Current status: {batch.status}. Returning None.')
            return None
        elif if_incomplete == 'wait':
            logger_llm.info(f'WAITING FOR BATCH {batch_id} TO COMPLETE.')
            with tqdm(total = None, desc=f'Waiting for batch {batch_id} to complete', unit='check') as pbar:
                while batch.status in STATUS_IN_QUEUE:
                    time.sleep(wait_interval)
                    batch = client.batches.retrieve(batch_id)
                    if batch.status in STATUS_FAILED:
                        logger_llm.error(f'ENDING WAIT - BATCH FAILED: Batch {batch_id} has status {batch.status}.')
                        if if_failed == 'error':
                            raise ValueError(f'Batch {batch_id} has failed. Current status: {batch.status}')
                        else:
                            return None
                    logger_llm.info(f'BATCH {batch_id} STATUS: {batch.status}. Waiting {wait_interval} seconds before checking again.')
                    pbar.update(1)
                    pbar.set_postfix({'status': batch.status, 'completed': f'{batch.request_counts.completed}/{batch.request_counts.total}', 'failed': f'{batch.request_counts.failed}/{batch.request_counts.total}'})
        else:
            raise ValueError(f'Invalid value for if_incomplete: {if_incomplete}. Must be "error", "ignore", or "wait".')
    elif batch.status in STATUS_FAILED:
        logger_llm.error(f'BATCH {batch_id} FAILED: status={batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}, failed={batch.request_counts.failed}/{batch.request_counts.total}')
        if if_failed == 'error':
            raise ValueError(f'Batch {batch_id} has failed. Current status: {batch.status}')
        else:
            return None

    if batch.output_file_id is None:
        logger_llm.error(f'BATCH {batch_id} RESULTS RETRIEVAL FAILED: Batch has no output file ID. Current status: {batch.status}.')
        if if_output_missing == 'error':
            raise ValueError(f'Batch {batch_id} has no output file ID. Current status: {batch.status}.')
        else:
            return None
    
    try:
        out_file = client.files.content(batch.output_file_id)
    except NotFoundError as e:
        logger_llm.error(f'BATCH {batch_id} RESULTS RETRIEVAL FAILED: Could not retrieve output file {batch.output_file_id}. Current status: {batch.status}. Error: {e}')
        if if_output_missing == 'error':
            raise ValueError(f'Output file for batch {batch_id} is missing. Current status: {batch.status}. Error: {e}')
        else:
            return None
    logger_llm.debug(f'BATCH RESULTS RETRIEVED: {batch.id} - {sum(1 for _ in out_file.iter_lines())} results retrieved.')

    save_path = save_as
    if save:
        if save_path is None:
            # batch details are normally saved alongside the term collection, so we can use the
            # location of {batch_id}.json to work out which collection this batch belongs to.
            batch_file_path = None
            for root, dirs, files in os.walk(TERMS_DIR):
                for file in files:
                    if file == f'{batch_id}.json':
                        batch_file_path = os.path.join(root, file)
                        break
                if batch_file_path:
                    break
            if batch_file_path is None:
                logger_llm.warning(f'BATCH DETAILS FILE NOT FOUND: {batch_id}.json in {TERMS_DIR}. Saving results under "unknown".')
                term_coll = 'unknown'
            else:
                term_coll = os.path.basename(os.path.dirname(os.path.dirname(batch_file_path)))
            save_path = os.path.join(OUT_DIR, term_coll, 'batch-results', f'{batch_id}.jsonl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_file.write_to_file(save_path)
        logger_llm.debug(f'BATCH RESULTS SAVED: {batch.id} at {save_path}')

    if return_type.lower() == 'contents':
        return [json.loads(line) for line in out_file.iter_lines()]
    elif save_path is None:
        raise ValueError(f'return_type="path" requires save=True (or an explicit save_as). Nothing was written for batch {batch_id}.')
    else:
        return save_path