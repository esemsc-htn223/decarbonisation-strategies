from typing import Iterable, Literal
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
import json
import time

from ..companies import Company, COMPANY_ALIASES, file_metadata
from ..regex.terms import Term
from ..util import REPORT_TYPES_LONG, logger_llm, TERMS_DIR

def construct_extracts(
        chunks:Iterable[str], chunk_inds:Iterable[int], 
        overlap_strategy:Literal["before", "after", "both", ]|None = None,
        overlap_amount:float = 0.2
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
        Defaults to 0.2 (i.e., a fifth of the adjacent chunk(s)). Only has an effect if overlap_strategy is not None.
    
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
        overlap_amount:float = 0.2,
        save_path:str|None = None, overwrite:bool = False
):
    '''
    Construct a prompt to pass to the LLM, including task description, output format
    and the relevant extracts.
    
    Parameters
    ----------
    company : str or Company
        The company the report relates to, either as a string (e.g., "Shell") or as a Company object.
        If a Company object is passed, the company's full_name will be used in the prompt.
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
        Defaults to 0.2 (i.e., a fifth of the adjacent chunk(s)). Only has an effect if overlap_strategy is not None.
    
    Returns
    -------
    str
        The constructed prompt, as a string.
    '''

    if isinstance(company, str):
        co_alias = COMPANY_ALIASES.get(company.lower(), company)
    elif isinstance(company, Company):
        co_alias = company.full_name
    else:
        raise ValueError(f"company should be either the name of a company (as a str), or a Company object. Got {type(company)}")
    
    report_alias = REPORT_TYPES_LONG.get(report_type, report_type.replace('-', ' '))
    
    chunks_str = construct_extracts(
        chunks=chunks, chunk_inds=chunk_inds, 
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

Notes:
- {overlap_info}
- OCR errors may result in some incorrect text or formatting.

# Extracts:
-----------
{chunks_str}'''
    
    logger_llm.debug(f'PROMPT CONSTRUCTED: {co_alias}_{report_alias}_{year}_{term.prompt_fname.replace(".txt", "")} - {len(chunks)} chunks, indices {chunk_inds}, overlap_strategy={overlap_strategy}, overlap_amount={overlap_amount}')
    if save_path is not None:
        fname = os.path.basename(save_path)
        if os.path.exists(save_path) and not overwrite:
            logger_llm.debug(f'PROMPT NOT SAVED: {fname} overwrite=False, json present at {save_path}')
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(prompt)
            logger_llm.debug(f'PROMPT SAVED: {fname} at {save_path}')
    else:
        logger_llm.debug(f'PROMPT NOT SAVED: No save_path provided.')
    
    return prompt

def prompt_id(company:str|Company, report_type:str, year:int|str, term:str|Term) -> str:
    '''
    Create a unique ID for a prompt based on the company, report type, year, and term.
    
    Parameters
    ----------
    company : str or Company
        The company the report relates to, either as a string (e.g., "Shell") or as a Company object.
        If a Company object is passed, the company's full_name will be used in the ID.
    report_type : str
        The type of report the extracts are taken from (e.g., "annual-report" or "sustainability-report"). 
    year : int or str
        The year the report was published.
    term : str or Term
        The technology/term the extracts relate to, either as a string (e.g., "carbon capture") or as a Term object. 
        If a Term object is passed, the term's prompt_alias will be used in the ID.
    
    Returns
    -------
    str
        The generated prompt ID, in the format "{company}_{report_type}_{year}_{term}", with all values in lowercase and spaces replaced by hyphens.
    '''
    if isinstance(company, str):
        if company in COMPANY_ALIASES.values():
            co_alias = list(COMPANY_ALIASES.keys())[list(COMPANY_ALIASES.values()).index(company)]  # get key from value if we have been provided a long name
        else:
            co_alias = company
        co_alias = company
    elif isinstance(company, Company):
        co_alias = company.folder_name
    else:
        raise ValueError(f"company should be either the name of a company (as a str), or a Company object. Got {type(company)}")
    
    if isinstance(term, str):
        term_alias = term
    elif isinstance(term, Term):
        term_alias = term.prompt_fname.replace('.txt', '')
    else:
        raise ValueError(f"term should be either the name of a term (as a str), or a Term object. Got {type(term)}")
    
    return f"{co_alias}_{report_type}_{year}_{term_alias}".lower().replace(' ', '-').replace('(', '').replace(')', '')

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
        model:str = 'gpt-5.1', max_retries:int = 3,
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
        *, 
        file_paths:list[str]|None=None, report_dirs:list[str]|None=None, 
        report_type_dir:list[str]|None=None, co_dirs:list[str]|None=None,
        include_all:bool = False,
        model:str = 'gpt-5.1',
        save_path:str|None = None, overwrite:bool = False,
        return_data:bool = True
    )-> list[dict]:
    '''
    Create a jsonl file containing the prompts and metadata for analysis with OpenAI's API, based on the given file paths or directories.

    Parameters
    ----------
    file_paths : list[str] | None, optional
        A list of file paths to prompt text files to include in the jsonl. Each file should be a text file containing a prompt.
    report_dirs : list[str] | None, optional
        A list of directories containing report subdirectories to include in the jsonl. All .txt files in the report subdirectories will be included as prompts.
    report_type_dir : list[str] | None, optional
        A list of directories containing report type subdirectories (e.g., "annual-report", "sustaintability-report") to include in the jsonl. 
        All .txt files in the report type subdirectories will be included as prompts.
    co_dirs : list[str] | None, optional
        A list of directories containing company subdirectories to include in the jsonl. All .txt files in the company subdirectories will be included as prompts.
    include_all : bool, optional
        Whether to include all prompts in the data directory. Defaults to False. If True, the above options will be ignored and 
        all .txt files in the data directory and its subdirectories will be included as prompts.
    save_path : str | None, optional
        The path where the jsonl file should be saved. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite an existing file at save_path. Defaults to False.
    
    Returns
    -------
    list[dict]
        A list of dictionaries, each containing the metadata and prompt content for a single prompt, ready to be passed to OpenAI's API. If 
        save_path is provided and overwrite is True, the list will also be saved as a jsonl file at the specified path.
    '''
    if include_all:
        # get all .txt files in data directory and subdirectories
        file_paths = []
        for root, dirs, files in os.walk(os.path.join(TERMS_DIR)):
            for file in files:
                if file.endswith('.txt'):
                    if os.path.basename(file) == 'terms.txt':
                        continue  # skip terms metadata file
                    file_paths.append(os.path.join(root, file))
    
    elif file_paths is None:
        file_paths = []
        if report_dirs is not None:
            for report_dir in report_dirs:
                for root, dirs, files in os.walk(report_dir):
                    for file in files:
                        if file.endswith('.txt'):
                            file_paths.append(os.path.join(root, file))
        if report_type_dir is not None:
            for root, dirs, files in os.walk(report_type_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_paths.append(os.path.join(root, file))
        if co_dirs is not None:
            for co_dir in co_dirs:
                for root, dirs, files in os.walk(co_dir):
                    for file in files:
                        if file.endswith('.txt'):
                            file_paths.append(os.path.join(root, file))
        else:
            raise ValueError('If include_all is False, at least one of file_paths, report_dirs, report_type_dir, or co_dirs must be provided to specify which prompts to include. None of these were provided.')

    # validate file paths
    valid_file_paths = []
    for file_path in file_paths:
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            valid_file_paths.append(file_path)
        else:
            logger_llm.warning(f'Invalid file path: {file_path}. Skipping.')
    file_paths = valid_file_paths

    # gather jsons in jsonl format
    data = []
    schema = make_strict_schema(PromptResponse.model_json_schema())
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            prompt_content = f.read()
        file_mdata = file_metadata(file_path)
        term = os.path.basename(file_path).split('.')[0]  # get term from filename
        data.append({
            'custom_id': prompt_id(file_mdata['company'], file_mdata['report_type'], file_mdata['year'], term),  # custom ID for OpenAI API
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
    logger_llm.debug(f'JSONL DATA CREATED: {len(data)} records created from {len(file_paths)} file paths.')

    # write jsonl file
    if save_path is not None:
        fname = os.path.basename(save_path)
        if os.path.exists(save_path) and not overwrite:
            logger_llm.debug(f'JSONL NOT SAVED: {fname} overwrite=False, file present at {save_path}')
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                for record in data:
                    f.write(json.dumps(record) + '\n')
            logger_llm.debug(f'JSONL SAVED: {fname} at {save_path}')
    
    if return_data:
        return data
    else:
        return None

def submit_openai_batch(
        jsonl_path:str, *, 
        terms_dir:str|None=None
):
    '''
    Analyse the given prompts using the specified model, and return the responses.
    
    Parameters
    ----------
    jsonl_path : str
        The path to the JSONL data to submit to OpenAI's API.
    terms_dir : str | None, optional
        The path to the subfolder in the data directory containing the term files. If provided, the 
        batch will be saved in the 'batches' subfolder as a JSON file with the batch ID as the filename.
        Defaults to None.
    
    Returns
    -------
    dict
        The analysis results, as a dictionary.
    '''
    if not isinstance(jsonl_path, str):
        raise ValueError(f'jsonl_path should be a path to a jsonl file containing the prompts. Got {type(jsonl_path)}')
    if not os.path.isfile(jsonl_path):
        raise ValueError(f'jsonl_path should be a path to a jsonl file containing the prompts. Got {jsonl_path}')
    
    if terms_dir is None:
        terms = jsonl_path.split(os.path.sep)[-3]  # get terms collection name from path like 'data/terms/{collection-name}/prompts/prompts.jsonl'
    else:
        terms = terms_dir.split(os.path.sep)[-1]  # get terms collection name from path like 'data/terms/{collection-name}'
    jsonl_name = os.path.basename(jsonl_path)
    
    # set up client
    load_dotenv()
    client = OpenAI()

    batch_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch"
    )
    logger_llm.debug(f'JSONL UPLOADED -> OpenAI: {os.path.basename(jsonl_path)} with ID {batch_file.id}. Full path: {jsonl_path}')

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={}
    )
    logger_llm.debug(f'OPENAI BATCH CREATED: {batch.id} for file {batch_file.id}.')

    if terms_dir:
        batch_save_path = os.path.join(terms_dir, 'batches', f'{batch.id}.json')
        os.makedirs(os.path.dirname(batch_save_path), exist_ok=True)
        with open(batch_save_path, 'w') as f:
            f.write(json.dumps(batch.model_dump(), indent=4))
        logger_llm.debug(f'BATCH DETAILS SAVED: {batch.id} at {batch_save_path}')

    return batch.id

def check_batch_status(batch_id:str, *, 
                       dir_to_search:str = TERMS_DIR) -> dict:
    '''
    Check the status of the requested batch, and update the saved batch details file.

    Parameters
    ----------
    batch_id : str
        The ID of the batch to check (also the name of the batch json file), as returned by submit_openai_batch.
    dir_to_search : str, optional
        The directory to search for the batch details file. Defaults to the 'terms' subdirectory of the data directory.

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

    for root, dirs, files in os.walk(dir_to_search):
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
        raise FileNotFoundError(f'Could not find batch details file for batch ID {batch_id} in directory {dir_to_search}')
    
    return batch.model_dump()
    
def retrieve_batch_results(batch_id:str, *, 
                       save_path:str|None = None,
                       return_results:bool = True) -> list[dict]|None:
    '''
    Retrieve the results of the requested batch.

    Parameters
    ----------
    batch_id : str
        The ID of the batch to retrieve results for, as returned by submit_openai_batch.
    save_path : str|None, optional
        The path to save the results to. If None, the results are not saved.
    return_results : bool, optional
        Whether to return the results as a list of dictionaries. Defaults to True.

    Returns
    -------
    list[dict]|None
        The batch results, as a list of dictionaries, or None if return_results is False.
    '''
    load_dotenv()
    client = OpenAI()

    batch = client.batches.retrieve(batch_id)
    
    if batch.status != 'completed':
        raise ValueError(f'Batch {batch_id} is not completed yet. Current status: {batch.status}')
    
    file = client.files.retrieve(batch.output_file_id)
    results = [json.loads(line) for line in file.read().splitlines()]
    
    
    logger_llm.debug(f'BATCH RESULTS RETRIEVED: {batch.id} - {len(results)} results retrieved.')

    return results