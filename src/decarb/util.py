__all__ = ['logger', 'logger_regex', 'logger_llm', 'DATA_DIR', 'OUT_DIR']

import logging
import os
import re

# Data directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, 'data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
COMPANIES_DIR = os.path.join(DATA_DIR, 'companies')
if not os.path.exists(COMPANIES_DIR):
    os.makedirs(COMPANIES_DIR)
TERMS_DIR = os.path.join(DATA_DIR, 'terms')
if not os.path.exists(TERMS_DIR):
    os.makedirs(TERMS_DIR)
LOG_DIR = os.path.join(DATA_DIR, 'log')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    if not os.path.exists(os.path.join(LOG_DIR, 'analysis.log')):
        with open(os.path.join(LOG_DIR, 'analysis.log'), 'w') as f:
            f.write('')
    if not os.path.exists(os.path.join(LOG_DIR, 'regex.log')):
        with open(os.path.join(LOG_DIR, 'regex.log'), 'w') as f:
            f.write('')
    if not os.path.exists(os.path.join(LOG_DIR, 'llm.log')):
        with open(os.path.join(LOG_DIR, 'llm.log'), 'w') as f:
            f.write('')


OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, 'out'))
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

DEFAULT_ENCODER = 'cl100k_base'


def resolve_path(path: str) -> str:
    '''
    Resolve a path against the repository root. Absolute paths are returned unchanged. Relative
    paths are taken as relative to BASE_DIR, not the current working directory.
    '''
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def record_id(company: str, report_type: str, year: int|str, term_fname: str) -> str:
    '''
    Standard identifier for objects combining a Report (company/report-type/year) 
    and a Term. Used for indexing and joining records across the pipeline.

    Parameters
    ----------
    company : str
        The company name, as used in the folder name of the report.
    report_type : str
        The report type, e.g. 'annual-report' or 'sustainability-report'.
    year : int|str
        The year of the report.
    term_fname : str
        The filename of the term, e.g. 'solar'. Accepts fnames with or without the '.txt' suffix.
    
    Returns
    -------
    str
        The record identifier, in the format "{company}_{report_type}_{year}_{term}".
        Whitespace is replaced with hyphens, and parentheses are removed. The identifier is lowercased.
    '''
    term_fname = term_fname.removesuffix('.txt')
    return f'{company}_{report_type}_{year}_{term_fname}'.lower().replace(' ', '-').replace('(', '').replace(')', '')

def tidy_md(md:str):
    '''Standardise markdown text for analysis.'''
    tidy_md = md.replace('$', r'\$')  # no maths in these reports
    tidy_md = re.sub(r'(-)\1{3,}', r'\1\1\1', tidy_md) # limit repeated - to 3
    tidy_md = re.sub(r'\n{3,}', r'\n\n', tidy_md) # limit repeated \n to 2
    return tidy_md


REPORT_TYPES = {'annual-report', 'sustainability-report'}
REPORT_TYPES_SHORT = {'annual-report': 'an', 'sustainability-report': 'sr'}
REPORT_TYPES_LONG = {'annual-report': 'financial report', 'sustainability-report': 'sustainability report'}

logger = logging.getLogger('analysis')
logger.setLevel(logging.DEBUG)
logger_fh = logging.FileHandler(os.path.join(LOG_DIR, 'analysis.log'))
logger_fh.setLevel(logging.DEBUG)
logger_fh.set_name('analysis_debug')
logger_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(logger_fh)

logger_regex = logging.getLogger('regex')
logger_regex.setLevel(logging.DEBUG)
logger_regex_fh = logging.FileHandler(os.path.join(LOG_DIR, 'regex.log'))
logger_regex_fh.setLevel(logging.DEBUG)
logger_regex_fh.set_name('regex_debug')
logger_regex_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger_regex.addHandler(logger_regex_fh)

logger_llm = logging.getLogger('llm')
logger_llm.setLevel(logging.DEBUG)
logger_llm_fh = logging.FileHandler(os.path.join(LOG_DIR, 'llm.log'))
logger_llm_fh.setLevel(logging.DEBUG)
logger_llm_fh.set_name('llm_debug')
logger_llm_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger_llm.addHandler(logger_llm_fh)