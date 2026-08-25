import numpy as np
import pandas as pd
import json

from decarb.terms import TermCollection, TERMS_ALL
from decarb.companies import CompanyCollection, COMPANIES_ALL
from decarb.llm.prompt import PromptResponse
from decarb.util import logger_llm, resolve_path

__all__ = ['to_df', 'DEFAULT_MISSING']



DEFAULT_MISSING = {
    'q1_sentiment': np.nan,
    'q1_sentiment_quotes': [],
    'q2_maturity': np.nan,
    'q2_maturity_quotes': [],
    'q3_stage': pd.NA,
    'q3_stage_quotes': [],
    'q4_commitments': [],
    'notes': None
}

keys_in_prompt_response = set(PromptResponse.model_fields.keys())
keys_in_default_missing = set(DEFAULT_MISSING.keys())
if keys_in_prompt_response != keys_in_default_missing:
    missing_in_default = keys_in_prompt_response - keys_in_default_missing
    missing_in_prompt_response = keys_in_default_missing - keys_in_prompt_response
    if missing_in_default:
        logger_llm.warning(f'DEFAULT_MISSING MISSING FIELDS: {missing_in_default} from PromptResponse not found in DEFAULT_MISSING. Defaulting to None.')
        for field in missing_in_default:
            DEFAULT_MISSING[field] = None
    if missing_in_prompt_response:
        logger_llm.warning(f'DEFAULT_MISSING EXTRA FIELDS: {missing_in_prompt_response} in DEFAULT_MISSING not found in PromptResponse. They will be ignored.')

def _get_out_jsonl_paths(in_jsonl_paths: list[str]|str) -> list[str]:
    '''
    Find the result files produced by the given input jsonl(s), by resolving each to its Job.

    Parameters
    ----------
    in_jsonl_paths : list[str] | str
        Path(s) to, or job ID(s) of, the input jsonl file(s).

    Returns
    -------
    list[str]
        The result file paths for every batch of every job, de-duplicated and in job order.
    '''
    from decarb.llm.prompt import Job  # imported here to avoid a circular import at module level

    if isinstance(in_jsonl_paths, str):
        in_jsonl_paths = [in_jsonl_paths]

    out_paths = []
    for in_jsonl_path in in_jsonl_paths:
        for path in Job(in_jsonl_path).output_paths:
            if path not in out_paths:
                out_paths.append(path)
    return out_paths

def _mark_missing(df: pd.DataFrame, terms:TermCollection, prompt_dict: dict, *, 
                 cos: CompanyCollection = COMPANIES_ALL,
                 missing_report_values:dict = DEFAULT_MISSING, missing_term_values:dict = DEFAULT_MISSING) -> pd.DataFrame:
    '''
    Mark missing rows in the dataframe for each company, report type, year, and term combination.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing company, report type, year, and term information.
    terms : TermCollection
        A collection of terms to check for missing values.
    prompt_dict : dict
        A dictionary mapping prompt IDs to prompts.
    missing_report_values : dict, optional
        A dictionary specifying default values for missing report types. Any columns not found in this dictionary
        will be filled with pandas.NA. Default values:
        - 'q1_sentiment': np.nan
        - 'q1_sentiment_quotes': []
        - 'q2_maturity': -1
        - 'q2_maturity_quotes': []
        - 'q3_stage': pd.NA
        - 'q3_stage_quotes': []
        - 'q4_commitments': []
        - 'notes': None
    missing_term_values : dict, optional
        A dictionary specifying default values for missing terms. Any columns not found in this dictionary
        will be filled with pandas.NA. For default values, see `missing_report_values`.

    Returns
    -------
    pd.DataFrame
        A new dataframe with missing rows added for each company, report type, year, and term
    '''
    question_cols = df.drop(columns=['company', 'company_alias', 'report_type', 'term_alias', 'term_regex', 'term_fname', 'year', 'prompt', 'term_regex_found', 'report_type_published']).columns.to_list()
    for col in missing_report_values.keys():
        if col not in question_cols:
            raise ValueError(f'Column "{col}" specified in missing_report_values is not found in the dataframe.')
    for col in missing_term_values.keys():
        if col not in question_cols:
            raise ValueError(f'Column "{col}" specified in missing_term_values is not found in the dataframe.')

    new_rows = {}

    make_index = lambda company, report_type, year, term: f"{company}_{report_type}_{year}_{term.prompt_fname.replace('.txt', '')}"

    for company in df['company'].unique():
        for report_type in df['report_type'].unique():
            df_co_rt = df[(df['company'] == company) & (df['report_type'] == report_type)]

            # 1. check years report was not published
            years_report_missing = set(df['year'].unique()) - set(df_co_rt['year'].unique())

            if df_co_rt.empty:
                raise ValueError(f'No data found for company "{company}" and report type "{report_type}".')

            for term in terms:
                years_term_missing = set(df['year'].unique()) - set(df_co_rt.loc[df_co_rt['term_alias'] == term.alias, 'year'].unique()) - set(years_report_missing)
                for yr in years_report_missing:
                    index = make_index(company, report_type, yr, term)
                    new_rows[index] = {
                        'company': company,
                        'company_alias': cos.get(company, company),
                        'report_type': report_type,
                        'term_alias': term.alias,
                        'term_regex': term.regex,
                        'term_fname': term.prompt_fname,
                        'year': str(yr),
                        'prompt': prompt_dict.get(index, pd.NA),
                        **{col: missing_report_values.get(col, pd.NA) for col in question_cols},
                        'term_regex_found': False,
                        'report_type_published': False,
                    }

                for yr in years_term_missing:
                        index = make_index(company, report_type, yr, term)
                        new_rows[index] = {
                            'company': company,
                                'company_alias': cos.get(company, company),
                                'report_type': report_type,
                                'term_alias': term.alias,
                                'term_regex': term.regex,
                                'term_fname': term.prompt_fname,
                                'year': str(yr),
                                'prompt': prompt_dict.get(index, pd.NA),
                                **{col: missing_term_values.get(col, pd.NA) for col in question_cols},
                                'term_regex_found': False,
                                'report_type_published': True,
                            }
    return pd.concat([
        df, 
        pd.DataFrame(list(new_rows.values()), index=new_rows.keys())
    ], axis=0).sort_values(by=['company', 'report_type', 'year', 'term_alias'])

def to_df(*,
               job = None,
               in_jsonl_paths: list[str]|str|None = None, out_jsonl_paths:list[str]|str|None = None,
               terms: TermCollection = TERMS_ALL, cos: CompanyCollection = COMPANIES_ALL,
               label_missing: bool = True, missing_report_values: dict = DEFAULT_MISSING, missing_term_values: dict = DEFAULT_MISSING,
               fillna_values:dict|None = DEFAULT_MISSING) -> pd.DataFrame:
    '''
    Return a dataframe showing the input prompt, and returned values for every prompt, labelled by company, report type, year, and term.

    Either pass `job` (the usual case), or pass `in_jsonl_paths` and/or `out_jsonl_paths` directly.
    When only `in_jsonl_paths` is given, the output paths are resolved via `_get_out_jsonl_paths`.

    Parameters
    ----------
    job : Job | Iterable[Job] | None, optional
        The job(s) whose results to assemble. Takes its term collection from the job when `terms`
        is left at its default. Mutually exclusive with the path arguments.
    in_jsonl_paths : list[str] | str | None
        Path(s) to the input JSONL file(s) containing the prompts.
    out_jsonl_paths : list[str] | str | None
        Path(s) to the output JSONL file(s) containing the responses. If None, resolved from
        `in_jsonl_paths`.
    terms : TermCollection, optional
        The TermCollection used in the analysis. Default is `TERMS_ALL`.
    cos : CompanyCollection, optional
        The CompanyCollection used in the analysis. Default is `COMPANIES_ALL`.
    label_missing : bool, optional
        Whether to label missing values. Default is True.
    missing_report_values : dict, optional
        A dictionary specifying default values for missing report types. Any columns not found in this dictionary
        will be filled with pandas.NA. Default values:
        - 'q1_sentiment': np.nan
        - 'q1_sentiment_quotes': []
        - 'q2_maturity': -1
        - 'q2_maturity_quotes': []
        - 'q3_stage': pd.NA
        - 'q3_stage_quotes': []
        - 'q4_commitments': []
        - 'notes': None
    missing_term_values : dict, optional
        A dictionary specifying default values for missing terms. Any columns not found in this dictionary
        will be filled with pandas.NA. For default values, see `missing_report_values`.
    fillna_values : dict | None, optional
        A dictionary specifying values to fill NaN values in the dataframe. If None, no filling will be done.
        Default values are the same as for `missing_report_values`.
    
    Returns
    -------
    pd.DataFrame
    '''
    if job is not None:
        if in_jsonl_paths is not None or out_jsonl_paths is not None:
            raise ValueError('Pass either job, or in_jsonl_paths/out_jsonl_paths - not both.')
        jobs = [job] if not isinstance(job, (list, tuple, set)) else list(job)
        in_jsonl_paths = [j.input_path for j in jobs]
        out_jsonl_paths = []
        for j in jobs:
            for path in j.output_paths:
                if path not in out_jsonl_paths:
                    out_jsonl_paths.append(path)
        # a job knows its own term collection, so only fall back to the default when they agree
        job_terms = {j.terms.folder_name for j in jobs}
        if len(job_terms) > 1:
            raise ValueError(f'Jobs span multiple term collections ({sorted(job_terms)}). Build a dataframe per collection.')
        if terms is TERMS_ALL and job_terms != {terms.folder_name}:
            terms = jobs[0].terms
    elif in_jsonl_paths is None:
        raise ValueError('Provide either job or in_jsonl_paths.')

    if out_jsonl_paths is None:
        out_jsonl_paths = _get_out_jsonl_paths(in_jsonl_paths)

    prompt_dict = {}
    if isinstance(in_jsonl_paths, str):
        in_jsonl_paths = [in_jsonl_paths]
    for in_jsonl_path in in_jsonl_paths:
        with open(resolve_path(in_jsonl_path), 'r') as _f:
            for line in _f:
                json_line = json.loads(line)
                prompt_dict[json_line['custom_id']] = json_line['body']['messages'][0]['content']

    responses = []

    if isinstance(out_jsonl_paths, str):
        out_jsonl_paths = [out_jsonl_paths]

    for out_jsonl_path in out_jsonl_paths:
        for entry in open(resolve_path(out_jsonl_path), 'r'):  # file locally saved
            entry = json.loads(entry)
            id = entry['custom_id']
            co, rt, yr, term = id.split('_')
            entry = json.loads(entry['response']['body']['choices'][0]['message']['content'])
            term = terms[term]

            entry['id'] = id
            entry['company'] = co
            entry['report_type'] = rt
            entry['year'] = yr
            entry['term_alias'] = term.alias
            entry['company_alias'] = cos.get(co, co)
            entry['term_regex'] = term.regex
            entry['term_fname'] = term.prompt_fname
            entry['prompt'] = prompt_dict[id]
            entry['term_regex_found'] = True
            entry['report_type_published'] = True
            responses.append(entry)


    first_cols = ['company', 'report_type', 'year', 'term_alias']
    df = pd.DataFrame(responses).set_index('id')
    cols = first_cols + df.drop(columns=first_cols).columns.to_list()
    df = df[cols]

    if label_missing:
        df = _mark_missing(df, terms, prompt_dict, missing_report_values=missing_report_values, missing_term_values=missing_term_values)
    df = df.convert_dtypes()

    if fillna_values is not None:
        df = df.fillna({col: val for col, val in fillna_values.items() if col in df.columns and not isinstance(val, list)})

    return df