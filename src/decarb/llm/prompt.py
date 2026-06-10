from dataclasses import dataclass
import io
from typing import Iterable, Literal, Iterator
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
import json
import time
import uuid
import warnings
import tiktoken

from ..companies import Company, COMPANY_ALIASES, Report, file_metadata
from ..regex.terms import Term, TermCollection
from ..util import OUT_DIR, REPORT_TYPES_LONG, logger_llm, TERMS_DIR

DEFAULT_MODEL = 'gpt-5.1'

def construct_extracts(
        report:Report, term:Term, *,
        overlap_strategy:Literal["before", "after", "both", ]|None = None,
        overlap_amount:float = 0.2
) -> str:
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
    
    Returns
    -------
    str
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

        header = f"# Extract {i+1} (section {chunk_ind} / {n_chunks}):\n\n"
        
        extracts.append(f"{header + extract}")
    
    return "\n\n------------------\n\n".join(extracts)

def construct_prompt(
        report:Report, term:Term,
        *,
        #company:str|Company, report_type:str, year:int|str, term:str|Term, 
        #chunks:Iterable[str], chunk_inds:Iterable[int], 
        overlap_strategy:Literal["before", "after", "both", ]|None = None,
        overlap_amount:float = 0.2,
        save:bool = True,
        save_as:str|None = None, 
        overwrite:bool = False
):
    '''
    Construct a prompt to pass to the LLM, including task description, output format
    and the relevant extracts.
    
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
    save : bool, optional
        Whether to save the constructed prompt as a text file in the report's term directory (or at the path specified by `save_as` if provided). Defaults to True.
    save_as : str or None, optional
        The path at which to save the constructed prompt. If None (the default), the prompt will be saved in the report's term directory. Only has an effect if `save` is True.
    overwrite : bool, optional
        Whether to overwrite an existing file at the save path. Defaults to False.

    Returns
    -------
    str
        The constructed prompt, as a string.
    '''
    save_path = report._prompt_path(term.collection, term) if save_as is None else save_as
    if not save_path.endswith('.txt'):
        raise ValueError(f'save_as should be a path ending with .txt. Got {save_path}')

    co_alias = report.company.full_name
    year = report.year    
    report_alias = REPORT_TYPES_LONG.get(report.report_type, report.report_type.replace('-', ' '))
    
    chunks_str = construct_extracts(
        report=report, term=term,
        overlap_strategy=overlap_strategy, overlap_amount=overlap_amount
    )
    
    overlap_info = "\n"
    if overlap_strategy is not None and overlap_amount > 0:
        overlap_info += f"Extracts include {overlap_amount:.0%} of"
        if overlap_strategy in ['before', 'both']:
            overlap_info += " the previous "
        if overlap_strategy in ['after', 'both']:
            if overlap_strategy in ['both']:
                overlap_info += "and"
            overlap_info += " following section"
            if overlap_strategy in ['both']:
                overlap_info += "s"
        overlap_info += f" of the report to provide additional context."

    prompt = f'''The following extracts have been taken from {co_alias}'s {year} {report_alias} and all relate to {term.prompt_alias}. 

Use the contained information to answer the following questions:
1. "How positive or negative is the overall sentiment towards {term.prompt_alias}?"
2. "What key commitments has {co_alias} made to {term.prompt_alias}?"
3. "At what stage of implementation is {term.prompt_alias}?"
4. "How mature does {co_alias} consider {term.prompt_alias} to be?"

{overlap_info}

# Extracts:
-----------
{chunks_str}'''
    
    logger_llm.debug(f'PROMPT CONSTRUCTED: {co_alias}_{report_alias}_{year}_{term.prompt_fname.replace(".txt", "")} - overlap_strategy={overlap_strategy}, overlap_amount={overlap_amount}')
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
        term_alias = term
    elif isinstance(term, Term):
        term_alias = term.prompt_fname.replace('.txt', '')
    else:
        raise ValueError(f"term should be either the name of a term (as a str), or a Term object. Got {type(term)}")
    
    return f"{report.company.folder_name}_{report.report_type}_{report.year}_{term_alias}".lower().replace(' ', '-').replace('(', '').replace(')', '')

class PromptResponse(BaseModel):
    sentiment: float = Field(
        description="Question 1: Sentiment towards the technology, on a scale from -1 (totally negative) to 1 (totally positive)",
        ge=-1.0, le=1.0
    )
    commitments: list[str] = Field(
        description="Question 2: Key extracts or quotes showing commitments towards the technology."
    )
    stage: Literal["commercial-deployment", "demonstration", "pilot", "research and development", "unclear"] = Field(
        description="Question 3: Stage of implementation of the technology"
    )
    maturity: Literal["emerging", "established", "phasing-out", "unclear"] = Field(
        description="Question 4: How mature the company considers the technology to be"
    )
    notes: str|None = Field(default=None,
        description="Additional notes or caveats about the company's stance towards the term, or the quality of the extracts"
    )

class AnalysisRecord(BaseModel):
    company: str
    report_type: Literal["annual-report", "sustainability-report"]
    year: int
    term: str
    response: PromptResponse

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

def analyse_one(
        prompt_file:str, 
        *, 
        model:str = DEFAULT_MODEL, max_retries:int = 3,
        save_path:str|None = None, overwrite:bool = False
) -> AnalysisRecord:
    '''
    Analyse the given prompt using the specified model, and return the response.
    
    Parameters
    ----------
    prompt_file : str
        The path of the prompt file.
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
    load_dotenv()
    client = instructor.from_openai(OpenAI())

    if os.path.isfile(prompt_file):
        with open(prompt_file, 'r') as f:
            prompt_content = f.read()
    else:
        raise ValueError(f'prompt_file should be a path to a text file containing the prompt. Got {prompt_file}')

    response = client.chat.completions.create(  # chat completions only allows single prompts - fine for this application
        model=model,
        response_model=PromptResponse,
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

    if save:
        if save_as is None:
            term_coll = os.path.basename(os.path.dirname(os.path.dirname(prompt_paths[0])))  # use first prompt path for collection. if we have multiple later, we raise an error so we can assume this is correct.
            fname = uuid.uuid5(uuid.NAMESPACE_OID, ''.join(prompt_paths)).hex
            save_as = os.path.join(TERMS_DIR, term_coll, 'prompts', f'{fname}.jsonl')
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
    schema = make_strict_schema(PromptResponse.model_json_schema())
    term_collections = set()
    for file_path in prompt_paths:
        with open(file_path, 'r') as f:
            prompt_content = f.read()
        term = os.path.basename(file_path).split('.')[0]  # get term from filename
        term_coll = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # get term collection from parent folder name
        term_collections.add(term_coll)
        if len(term_collections) > 1:
            raise ValueError(f'Multiple TermCollections found in prompt_paths: {term_collections}. All prompts in a single batch must relate to the same TermCollection.')
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
                        'name': 'PromptResponse',
                        'schema': schema,
                        'strict': True
                    }
                }
            }
        })
    logger_llm.debug(f'JSONL DATA CREATED: {len(data)} records created from {len(prompt_paths)} file paths.')

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
    portion: list[dict]
    token_count: int
    start_index: int
    end_index: int

    def __len__(self):
        return len(self.portion)
    
    def __getitem__(self, key):
        return self.portion[key]
    
    def __repr__(self):
        return f'JSONLPortion(start_index={self.start_index}, end_index={self.end_index}, token_count={self.token_count}, portion_len={len(self.portion)})'

def jsonl_portions(
        jsonl_path:str, 
        max_tokens:int = 3_000,
        encoding_name:str = 'cl100k_base',
        save:bool = False, save_dir:str|None = None, overwrite:bool = False
) -> Iterator[JSONLPortion]:
    '''
    Take a jsonl file and yield portions of it as lists of dictionaries, where each portion contains at 
    most `max_tokens` tokens in total.

    Parameters
    ----------
    jsonl_path : str
        The path to the jsonl file to portion.
    max_tokens : int, optional
        The maximum number of tokens that each portion should contain. Defaults to 3,000.
    encoding_name : str, optional
        The name of the encoding to use for token counting. Defaults to 'cl100k_base'.
    save : bool, optional
        Whether to save the portions to files. Defaults to False.
    save_dir : str or None, optional
        The directory to save the portions to. If None, will save in the same directory as the jsonl file, 
        in a folder named 'portions-<jsonl_filename>'. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    '''
    if save:
        if save_dir is None:
            base_dir = os.path.dirname(jsonl_path)
            jsonl_fname = os.path.basename(jsonl_path).replace('.jsonl', '')
            save_dir = os.path.join(base_dir, f'portions-{jsonl_fname}')
        os.makedirs(save_dir, exist_ok=True)

    encoding = tiktoken.get_encoding(encoding_name)
    current_portion = []
    current_token_count = 0
    
    def save_portion(portion:list[dict], i:int):
        if save:
            portion_fname = f'portion-{i}.jsonl'
            portion_path = os.path.join(save_dir, portion_fname)
            if os.path.exists(portion_path) and not overwrite:
                logger_llm.debug(f'PORTION NOT SAVED: {portion_fname} overwrite=False, file present at {portion_path}')
            else:
                with open(portion_path, 'w') as f:
                    for record in portion:
                        f.write(json.dumps(record) + '\n')
                logger_llm.debug(f'PORTION SAVED: {portion_fname} at {portion_path}')

    start_index = 0
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            prompt_json = json.loads(line)
            token_count = len(encoding.encode(prompt_json.get('body', {}).get('messages', [{}])[0].get('content', '')))
            if current_token_count + token_count > max_tokens and current_portion:
                yield JSONLPortion(
                    portion=current_portion,
                    token_count=current_token_count,
                    start_index=start_index,
                    end_index=i-1
                )
                save_portion(current_portion, i)
                
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
            end_index=i
        )
        save_portion(current_portion, i+1)


def submit_openai_batch(
        jsonl:str|list[dict], 
        save_dir:str|None = None
) -> str:
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
    
    Returns
    -------
    str
        The ID of the submitted batch.
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

    # determine term collection for save location
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
        logger_llm.debug(f'JSONL UPLOADED -> OpenAI: {os.path.basename(jsonl)} with ID {batch_file.id}. From list, with len: {len(jsonl)}')
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
    os.makedirs(save_dir, exist_ok=True)
    batch_save_path = os.path.join(save_dir, f'{batch.id}.json')
    with open(batch_save_path, 'w') as f:
        f.write(json.dumps(batch.model_dump(), indent=4))
    logger_llm.debug(f'BATCH DETAILS SAVED: {batch.id} at {batch_save_path}')

    return batch.id

def check_batch_status(batch_id:str) -> dict:
    '''
    Check the status of the requested batch, and update the saved batch details file.

    Parameters
    ----------
    batch_id : str
        The ID of the batch to check (also the name of the batch json file), as returned by submit_openai_batch.

    Returns
    -------
    dict
        The updated batch details, as a dictionary.
    '''
    load_dotenv()
    client = OpenAI()

    batch = client.batches.retrieve(batch_id)
    logger_llm.debug(f'BATCH STATUS CHECKED: {batch.id} - status={batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}, created at {batch.created_at}, updated at {time.time()}')
    
    # find batch details file and update it
    batch_file_path = None

    for root, dirs, files in os.walk(TERMS_DIR):
        for file in files:
            if file == f'{batch_id}.json':
                batch_file_path = os.path.join(root, file)
                break
        if batch_file_path:
            break
    
    if batch_file_path:
        with open(batch_file_path, 'w') as f:
            f.write(json.dumps(batch.model_dump(), indent=4))
        logger_llm.debug(f'BATCH DETAILS UPDATED: {batch.id} at {batch_file_path}')
    else:
        raise FileNotFoundError(f'Searched for batch file "{batch_id}.json" in directory {TERMS_DIR}, but found no match.')
    
    return batch.model_dump()
    
def retrieve_batch_results(batch_id:str, *, 
                       if_incomplete:Literal['error', 'ignore'] = 'error',
                       save:bool = True, save_as:str = None) -> list[dict]|None:
    '''
    Retrieve the results of the requested batch.

    Parameters
    ----------
    batch_id : str
        The ID of the batch to retrieve results for, as returned by submit_openai_batch.
    if_incomplete : Literal['error', 'ignore']
        What to do if the batch is not completed yet. If 'error', a ValueError will be raised. If 'ignore', the function will return None and log a warning.

    Returns
    -------
    list[dict]|None
        The batch results, as a list of dictionaries, or None if return_results is False.
    '''
    load_dotenv()
    client = OpenAI()

    batch = client.batches.retrieve(batch_id)
    
    if batch.status != 'completed':
        if if_incomplete == 'error':
            raise ValueError(f'Batch {batch_id} is not yet complete. Current status: {batch.status}')
        elif if_incomplete == 'ignore':
            logger_llm.warning(f'Batch {batch_id} is not completed yet. Current status: {batch.status}. Returning None.')
            return None
        else:
            raise ValueError(f'Invalid value for if_incomplete: {if_incomplete}. Must be "error" or "ignore".')
    
    file = client.files.retrieve(batch.output_file_id)
    results = [json.loads(line) for line in file.read().splitlines()]
    
    logger_llm.debug(f'BATCH RESULTS RETRIEVED: {batch.id} - {len(results)} results retrieved.')

    if save:
        for root, dirs, files in os.walk(TERMS_DIR):
            for file in files:
                if file == f'{batch_id}.json':
                    batch_file_path = os.path.join(root, file)
                    break
            if batch_file_path:
                break
        term_coll = os.path.basename(os.path.dirname(os.path.dirname(batch_file_path)))
        save_path = save_as or os.path.join(OUT_DIR, term_coll, f'{batch_id}_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger_llm.debug(f'BATCH RESULTS SAVED: {batch.id} at {save_path}')

    return results