from dataclasses import dataclass
import io
from typing import Literal, Iterator
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
import json
import time
import uuid
import tiktoken
from tqdm import tqdm

from decarb.companies import Company, COMPANY_ALIASES, Report, file_metadata
from decarb.terms import Term, TermCollection
from decarb.util import OUT_DIR, REPORT_TYPES_LONG, logger_llm, TERMS_DIR

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
    Construct a prompt to pass to the LLM, including task description and the relevant extracts.
    
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
    
    prompt = f'''The following extracts have been taken from {co_alias}'s {year} {report_alias} and all relate to {term.prompt_alias}. 

Use these extracts to answer the following questions. Base your answers only on the extracts, not on any other information you may 
have about {co_alias}'s use of {term.prompt_alias} in {year}. Answers must be evidenced by direct quotes from the extracts, along
with a justification of why each quote is relevant.

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

3. "At what stage of commercialisation is {term.prompt_alias} within {co_alias}?
Choose from: 
- 'research and development'
- 'pilot'
- 'demonstration'
- 'commercial-deployment'
- 'phasing-out'
- 'unclear'"

4. "What commitments has {co_alias} made towards {term.prompt_alias}?
Provide a list of commitments, flagging commitments that include specific measurable targets as verifiable."


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

class QuoteEvidence(BaseModel):
    quote: str = Field(description="Direct quote from the source text")
    justification: str = Field(description="Explanation of why this quote is relevant as evidence")

class Commitment(BaseModel):
    commitment: str = Field(description="A commitment made by the company towards the technology")
    quote: str = Field(description="Direct quote from the source text evidencing the commitment")
    justification: str = Field(description="Explanation of how the quote evidences the commitment")
    verifiable: bool = Field(description="True if the commitment includes specific, measurable targets. False if the commitment is aspirational or qualitative only.")

class PromptResponse(BaseModel):
    # Question 1: "How positive or negative is the overall sentiment towards {term.prompt_alias}?"
    q1_sentiment: float = Field(
        description="Question 1: Sentiment towards the technology, on a scale from -1 (totally negative) to 1 (totally positive)",
        ge=-1.0, le=1.0
    )
    q1_sentiment_quotes: list[QuoteEvidence] = Field(
        description="Quotes reflecting sentiment towards the technology and a justification for their inclusion. Given as evidence for the sentiment score given in q1_sentiment."  
    )

    # Question 2: "How mature is {term.prompt_alias} framed to be, in terms of Techonology Readiness Levels?"
    q2_maturity: int | None = Field(
        default=None,
        description="Question 2: The Technology Readiness Level (TRL) of the technology as framed in the extracts, 1-9. Definitions provided in the prompt. Null if insufficient information is available to assign a TRL.",
        ge=1, le=9
    )
    q2_maturity_quotes: list[QuoteEvidence] = Field(
        description="Quotes reflecting the Technology Readiness Level (TRL) of the technology and a justification for their inclusion. Given as evidence for the maturity given in q2_maturity."
    )

    # Question 3: "What stage of commercialisation is {term.prompt_alias} at within the company's operations?"
    q3_stage: Literal["research and development", "pilot", "demonstration", "commercial-deployment", "phasing-out", "unclear"] = Field(
        description="Question 3: The technology's stage of commercialisation within the company's operations."
    )
    q3_stage_quotes: list[QuoteEvidence] = Field(
        description="Quotes reflecting the technology's stage of commercialisation within the company's operations and a justification for their inclusion. Given as evidence for the stage of commercialisation given in q3_stage."
    )

    # Question 4: "What commitments does the company make towards {term.prompt_alias}?"
    q4_commitments: list[Commitment] = Field(
        description="Question 4: Commitments made towards the technology."
    )

    # overall notes - to be deprecated after tests.
    notes: str|None = Field(
        default=None,
        description="Only populate if there are any additional notes about the response that are not captured by the other fields. Leave null otherwise."
    )

class AnalysisRecord(BaseModel):
    company: str
    report_type: Literal["annual-report", "sustainability-report"]
    year: int
    term: str
    response: PromptResponse

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
            term_coll = TermCollection.from_folder(os.path.dirname(os.path.dirname(prompt_paths[0])))  # use first prompt path for collection. if we have multiple later, we raise an error so we can assume this is correct.
            fname = uuid.uuid5(uuid.NAMESPACE_OID, ''.join(prompt_paths)).hex
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
        jsonl_path:str, *,
        strategy:Literal['max_tokens', 'max_prompts', 'split_into_n'] = 'max_tokens',
        max_tokens:int = 1_000_000,
        max_prompts:int = 100,
        split_into_n:int = 10,
        encoding_name:str = 'o200k_base',
        save:bool = False, save_dir:str|None = None, overwrite:bool = False
) -> Iterator[JSONLPortion]:
    '''
    Take a jsonl file and yield portions of it as lists of dictionaries, where each portion contains at 
    most `max_tokens` tokens in total.

    Parameters
    ----------
    jsonl_path : str
        The path to the jsonl file to portion.
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
    '''
    if strategy not in ['max_tokens', 'max_prompts', 'split_into_n']:
        raise ValueError(f'strategy should be one of "max_tokens", "max_prompts", or "split_into_n". Got {strategy}')

    if save:
        if save_dir is None:
            base_dir = os.path.dirname(jsonl_path)
            jsonl_fname = os.path.basename(jsonl_path).replace('.jsonl', '')
            save_dir = os.path.join(base_dir, f'portions-{jsonl_fname}')
        os.makedirs(save_dir, exist_ok=True)

    encoding = tiktoken.get_encoding(encoding_name)
    current_portion = []
    current_token_count = 0
    current_threshold_count = 0  # either prompts or tokens, depending on strategy
    start_index = 0
    
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

    if strategy == 'split_into_n':
        # get length of jsonl file
        with open(jsonl_path, 'r') as f:
            total_lines = sum(1 for _ in f)
            max_prompts = total_lines // split_into_n + (1 if total_lines % split_into_n > 0 else 0)

            for i, line in enumerate(f):
                prompt_json = json.loads(line)
                current_portion.append(prompt_json)
                if len(current_portion) >= max_prompts:
                    yield JSONLPortion(
                        portion=current_portion,
                        token_count=sum(len(encoding.encode(p.get('body', {}).get('messages', [{}])[0].get('content', ''))) for p in current_portion),
                        start_index=start_index,
                        end_index=i
                    )
                    save_portion_if_requested(current_portion, i, requested = save)

                    current_portion = []
                    start_index = i + 1
        # yield any remaining portion
        if current_portion:
            yield JSONLPortion(
                portion=current_portion,
                token_count=sum(len(encoding.encode(p.get('body', {}).get('messages', [{}])[0].get('content', ''))) for p in current_portion),
                start_index=start_index,
                end_index=i
            )
            save_portion_if_requested(current_portion, i+1, requested = save)
    else:
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
                    save_portion_if_requested(current_portion, i, requested = save)
                    
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
            save_portion_if_requested(current_portion, i+1, requested = save)

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
    os.makedirs(save_dir, exist_ok=True)
    batch_save_path = os.path.join(save_dir, f'{batch.id}.json')
    with open(batch_save_path, 'w') as f:
        f.write(json.dumps(batch.model_dump(), indent=4))
    logger_llm.debug(f'BATCH DETAILS SAVED: {batch.id} at {batch_save_path}')

    return batch.id


def check_batch_status(batch_id:str | None = None, *, on_miss: Literal['raise', 'print', 'pass', 'create'] = 'raise') -> dict:
    '''
    Check the status of all batches, or the requested batch, and update the saved batch details file.

    Parameters
    ----------
    batch_id : str | None
        The ID of the batch to check (also the name of the batch json file), as returned by submit_openai_batch.
    on_miss : Literal['raise', 'print', 'pass', 'create'], optional
        What to do if the batch details file is not found.
        - `'raise'`: raise a FileNotFoundError (default)
        - `'print'`: print a warning message and continue
        - `'pass'`: silently continue
        - `'create'`: create a new batch details file with the current batch status
        
    Returns
    -------
    dict
        The updated batch details, as a dictionary.
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

    if batch_id is None:
        # Check status of all batches
        batches = client.batches.list()
        for batch in batches:
            logger_llm.debug(f'BATCH STATUS CHECKED: {batch.id} - status={batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}, created at {batch.created_at}, updated at {time.time()}')
            update_batch_file(batch)
        return [batch.model_dump() for batch in batches]
    else:
        batch = client.batches.retrieve(batch_id)
        logger_llm.debug(f'BATCH STATUS CHECKED: {batch.id} - status={batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}, created at {batch.created_at}, updated at {time.time()}')
    
    # find batch details file and update it
    update_batch_file(batch)
    
    return batch.model_dump()
    
def retrieve_batch_results(batch_id:str, *, 
                       if_incomplete:Literal['error', 'ignore', 'wait'] = 'error',
                       if_failed:Literal['error', 'ignore'] = 'error',
                       save:bool = True, save_as:str = None,
                       wait_interval: int = 120) -> list[dict]|None:
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

    out_file = client.files.content(batch.output_file_id)
    
    logger_llm.debug(f'BATCH RESULTS RETRIEVED: {batch.id} - {sum(1 for _ in out_file.iter_lines())} results retrieved.')

    if save:
        # batch details always saved (no option to not), so we can use this to produce the output file path
        batch_file_path = None
        for root, dirs, files in os.walk(TERMS_DIR):
            for file in files:
                if file == f'{batch_id}.json':
                    batch_file_path = os.path.join(root, file)
                    break
            if batch_file_path:
                break
        term_coll = os.path.basename(os.path.dirname(os.path.dirname(batch_file_path)))
        save_path = save_as or os.path.join(OUT_DIR, term_coll, 'batch-results', f'{batch_id}.jsonl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out_file.write_to_file(save_path)
        logger_llm.debug(f'BATCH RESULTS SAVED: {batch.id} at {save_path}')

    return [json.loads(line) for line in out_file.iter_lines()]
