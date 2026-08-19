__all__ = ['search_file', 'search_files', 'regex_pipeline']

import pandas as pd
import re
import os
import pymupdf

from decarb.terms import TermCollection, TERMS_ALL
from decarb.companies import CompanyCollection
from decarb.util import REPORT_TYPES_SHORT

pymupdf.TOOLS.mupdf_display_errors(on = False)
pymupdf.TOOLS.mupdf_display_warnings(on = False)

def search_file(
    file:str|pymupdf.Document, 
    terms: TermCollection = TERMS_ALL, 
    *, 
    count_type: str = 'all', norm_type: str = 'words', norm:bool = True
) -> dict[str, int] | dict[str, int | float]:

    if isinstance(file, str): 
            doc = pymupdf.open(file) 
    elif isinstance(file, pymupdf.Document): 
            doc = file
    else:
        raise TypeError(f'file must be a filepath or Document instance. Got {type(file)}.')

    out= {term.alias: 0 for term in terms}

    n_words = 0

    for page in doc:
        txt = page.get_text().lower()

        n_words += len(re.findall(r"[^\W_]+", txt, re.M))

        for term in terms:
            matches = re.findall(term.regex, txt, re.I)
            if count_type == 'pages':
                out[term.alias] += 1
            else:
                out[term.alias] += len(matches)

    n_pages = len(doc)

    if norm:
        if norm_type.lower() == 'pages':
            out = {term: out[term]/n_pages for term in out}
        elif norm_type.lower() == 'words':
            out = {term: out[term]/n_words for term in out}
        else:
            raise ValueError(f'Invalid norm_type: {norm_type}. Must be "pages" or "words".')

    out['n_words'] = n_words
    out['n_pages'] = n_pages

    return out

def extract_year(filename: str) -> int:
    match = re.findall(r"\d{4}", filename)
    n_matches = len(match)
    if n_matches == 0:
        raise ValueError(f'filename does not contain a 4-digit year: {filename}')
    elif n_matches > 1:
        raise ValueError(f'more than 1 4-digit number found in filename: {filename}')
    else:
        return int(match[0])

def search_files(
    folder:str, 
    terms: TermCollection = TERMS_ALL, 
    *,
    count_type:str = 'all', norm: bool = True,
    year_index: bool = True,
    norm_type: str = 'words'
) -> pd.DataFrame:
    if not os.path.isdir(folder):
        raise ValueError('folder is not a folder!')

    out = {}
    for file in os.listdir(folder):
        if file.endswith('.pdf'):
            out[file] = search_file(os.path.join(folder, file), terms, count_type=count_type, norm=norm, norm_type=norm_type)
            out[file]['year'] = extract_year(file)

    out = pd.DataFrame(out).T
    out.index.name = 'filename'
    if year_index: out = out.reset_index().set_index('year')
    return out.sort_index()


def regex_pipeline(
        terms: TermCollection, cos: CompanyCollection, *, 
        count_type: str = 'all', norm_type: str = 'words',
        overwrite: bool = False, save: bool = True
    ):
        
        terms.log(f'REGEX PIPELINE START: terms={terms.folder_name}, {save=}')

        out = {}

        dfs = {}
        dfs_unnormed = {}

        if not os.path.exists(os.path.join(terms.folder, 'mentions')):
            os.makedirs(os.path.join(terms.folder, 'mentions'))
        if not os.path.exists(os.path.join(terms.folder, 'mentions-unnormed')):
            os.makedirs(os.path.join(terms.folder, 'mentions-unnormed'))

        for co in cos:
            for report_type in REPORT_TYPES_SHORT:
                reports_folder = os.path.join(co.raw_folder, report_type)

                out_fname_normed = os.path.join(terms.folder, 'mentions', f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}.csv')
                out_fname_unnormed = os.path.join(terms.folder, 'mentions-unnormed', f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}.csv')

                if overwrite or not os.path.exists(out_fname_normed):
                    dfs[f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}'] = search_files(reports_folder, terms.terms, norm=True, norm_type=norm_type, count_type=count_type, year_index=True)
                    if save:
                        dfs[f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}'].to_csv(out_fname_normed)
                else:
                    dfs[f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}'] = pd.read_csv(out_fname_normed, index_col=0)

                if overwrite or not os.path.exists(out_fname_unnormed):
                    dfs_unnormed[f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}'] = search_files(reports_folder, terms.terms, norm=False, norm_type=None, count_type=count_type, year_index=True)
                    if save:
                        dfs_unnormed[f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}'].to_csv(out_fname_unnormed)
                else:
                    dfs_unnormed[f'{co.folder_name}-{REPORT_TYPES_SHORT[report_type]}'] = pd.read_csv(out_fname_unnormed, index_col=0)

        out['dfs'] = dfs
        out['dfs_unnormed'] = dfs_unnormed
        return out