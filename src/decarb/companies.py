__all__ = ['Company', 'CompanyCollection', 'COMPANIES_ALL', 'COMPANY_ALIASES', 'co_dir_md_files']

from dataclasses import dataclass
import os
from typing import Iterable, Literal, Union
import uuid
import json

from decarb.util import COMPANIES_DIR, logger, REPORT_TYPES
from decarb.terms import Term, TermCollection

COMPANY_ALIASES = {
        'shell': 'Shell',
        'bp': 'BP',
        'exxon': 'ExxonMobil',
        'chevron': 'Chevron',
        'total': 'TotalEnergies'
    }

COMPANY_COLOURS = {
    'shell': '#FCD116',
    'bp': "#3ABE5B",
    'exxon': '#FF0000',
    'chevron': "#9BDAFF",
    'total': "#E69AFF"
}

class Report:
    '''A report, identified by the path to its directory, or to a file within it. Provides properties to access the paths of each associated file, and whether the file exists.'''
    def __init__(self, path: str):
        '''
        A report, identified by the path to its directory, or to a file within it.
        Provides properties to access the paths of each associated file, and whether the file exists.

        In general, properties with a leading underscore (e.g., _pdf_path) return the expected location of a file, without checking whether it actually exists, 
        while properties without a leading underscore (e.g., pdf_path) return the path if the file exists, and None otherwise. Methods indexing specific chunks
        will raise a ValueError if the requested chunk does not exist.

        Parameters
        ----------
        path : str
            The path to the report directory, or to a file within it. Expected structure of the
            path is: COMPANIES_DIR/co/report_type/report_name/...

        Raises
        ------
        AttributeError
            If any required metadata cannot be extracted from the file path.
        NotADirectoryError
            If the base folder for the report is not found.
        FileNotFoundError
            If the PDF file for the report is not found.
        '''
        # get company, report type, report name, and year from file path
        file_mdata = file_metadata(path)
        for entry in file_mdata:
            if file_mdata[entry] is None:
                raise AttributeError(f'Could not extract {entry} from file path: {path}')
            else:
                setattr(self, entry, file_mdata[entry])
        
        # identify base folder
        self.dir = os.path.join(COMPANIES_DIR, self.company.folder_name, self.report_type, self.report_name)
        if not os.path.exists(self.dir):
            raise NotADirectoryError(f'Base folder for report not found at expected location: {self.dir}')
        
        
        self._pdf_path = os.path.join(self.dir, f'{self.report_name}.pdf')
        self._raw_path = self._pdf_path
        self._md_path = os.path.join(self.dir, f'{self.report_name}.md')

        if not os.path.exists(self._pdf_path):
            # may have to change if dealing with reports in non-pdf form.
            raise FileNotFoundError(f'.pdf not found for report {self.report_name} @ {self._pdf_path}')

        self._chunks_dir = os.path.join(self.dir, 'chunks')

    @classmethod
    def from_metadata(cls, company: Union['Company', str], report_type: str, report_year: str):
        '''Create a Report object from its metadata.'''
        if isinstance(company, Company):
            company_folder = company.folder_name
        else:
            if company in COMPANY_ALIASES.values():
                company_folder = next(k for k, v in COMPANY_ALIASES.items() if v == company)
            else: 
                company_folder = company
            
            
            
        dir = os.path.join(COMPANIES_DIR, company_folder, report_type)
        # search for dir with year
        dir = next((os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d.endswith(report_year)), None)
        if dir is None:
            raise FileNotFoundError(f'Searched for requested year {report_year} report for company {company_folder} and report type {report_type}, but found no matches. Full dir searched: {os.path.join(COMPANIES_DIR, company_folder, report_type)}')

        return cls(path=dir)

    @classmethod
    def from_prompt_id(cls, prompt_id: str):
        '''Create a Report object from a prompt ID, which is in the format "{company_folder}_{report_type}_{report_name}_{term_folder}_{term.prompt_fname}".'''
        company_folder, report_type, report_name, term_fname = prompt_id.split('_')
        return cls.from_metadata(company_folder, report_type, report_name)

    def __format__(self, format_spec):
        if format_spec == 'name':
            return self.report_name
        elif format_spec == 'full':
            return f'{self.company.folder_name} - {self.report_type} - {self.report_name}'
        elif format_spec == 'path':
            return self.dir
        else:
            return self.__str__()
        
    def __str__(self):
        return f'Report(company={self.company.folder_name}, report_type={self.report_type}, report_name={self.report_name})'

    def __repr__(self):
        return self.__str__()

    ### Base files
    @property
    def pdf_path(self):
        '''Get the path to the PDF file for this report, if it exists. Otherwise, return None.'''
        return self._pdf_path if os.path.exists(self._pdf_path) else None

    @property
    def raw_path(self):
        '''Get the path to the PDF file for this report, if it exists. Otherwise, return None.'''
        # alias for pdf_path
        return self.pdf_path
    
    @property
    def md_path(self):
        '''Get the path to the markdown file for this report, if it exists. Otherwise, return None.'''
        return self._md_path if os.path.exists(self._md_path) else None


    ### Markdown chunks
    @property
    def chunks_dir(self):
        '''Get the path to the chunks directory for this report, if it exists. Otherwise, return None.'''
        return self._chunks_dir if os.path.exists(self._chunks_dir) else None
    
    @property
    def chunk_paths(self) -> list[str]:
        '''
        Get the paths of all chunk files in the given directory, as a sorted list.
        List is sorted by chunk index, which is extracted from the file name (assumes file names are in the format 'chunk-{index}.md').

        Parameters
        ----------
        chunks_folder : str
            The path to the directory containing the chunk files.

        Returns
        -------
        list[str]
            A list of the paths of all chunk files in the given directory.

        Raises
        ------
        ValueError
            If the chunks directory does not exist.
        '''
        if self.chunks_dir is None:
            raise ValueError(f'chunks_dir not found at expected location: {self._chunks_dir}')
        
        chunk_files = [f for f in os.listdir(self.chunks_dir) if f.endswith('.md') and f.startswith('chunk-')]
        chunk_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))  # Sort by chunk index
        return [os.path.join(self.chunks_dir, f) for f in chunk_files]
    
    def _chunk_path(self, index: int) -> str:
        '''The expected location of a chunk file for this report, given its index. Does not check whether the file actually exists.'''
        return os.path.join(self.chunks_dir, f'chunk-{index}.md')
    
    def chunk_path(self, index: int) -> str:
        '''Get the path to a specific chunk file, identified by its index, if it exists. Otherwise, return None.'''
        chunk_path = self._chunk_path(index)
        if not os.path.exists(chunk_path):
            raise IndexError(f'Chunk index {index} not found for report {self.report_name} - no chunk file found at expected location: {chunk_path}')
        return chunk_path
    
    @property
    def n_chunks(self):
        '''Number of chunk files for this report.'''
        return sum(1 for f in os.listdir(self.chunks_dir) if f.endswith('.md') and f.startswith('chunk-')) if self.chunks_dir else 0

    @property
    def chunks(self) -> list[str]:
        '''
        Get the markdown content of all chunk files in the given directory, as a list.
        List is sorted by chunk index, which is extracted from the file name (assumes file names are in the format 'chunk-{index}.md').

        Parameters
        ----------
        chunks_folder : str
            The path to the directory containing the chunk files.

        Returns
        -------
        list[str]
            A list of the markdown content of all chunk files in the given directory.
        '''
        return [open(chunk_file).read() for chunk_file in self.chunk_paths]

    def chunk(self, index: int) -> str:
        '''
        Get the markdown content of a specific chunk file, identified by its index.

        Parameters
        ----------
        index : int
            The index of the chunk to retrieve.

        Returns
        -------
        str
            The markdown content of the specified chunk file.
        
        Raises
        ------
        ValueError
            If the requested index is out of bounds.
        '''
        
        if index >= self.n_chunks:
            raise ValueError(f"Chunk index {index} is out of bounds (n_chunks={self.n_chunks})")
        path = self.chunk_path(index)
        if path is None:
            raise ValueError(f'Chunk file for index {index} not found at expected location: {self._chunk_path(index)}')
        return open(path).read()
    
    
    ### Term-specific files (search results, prompts)
    def term_coll_folder_names(self) -> list[str]:
        '''The names of all term folders for this report.'''
        is_term_folder = lambda f: os.path.isdir(os.path.join(self.dir, f)) and f != 'chunks' and not f.startswith('.')
        return [f for f in os.listdir(self.dir) if is_term_folder(f)]

    def _term_dir(self, terms: TermCollection | str):
        '''The expected location of the term directory for this report and term collection. Does not check whether the directory actually exists.'''
        if not isinstance(terms, str):
            try:
                terms = terms.folder_name
            except AttributeError:
                raise ValueError(f'terms should be either a string or a TermCollection object. Got {type(terms)}')
        return os.path.join(self.dir, terms) 
    
    def term_dir(self, terms: TermCollection | str):
        '''Get the path to the term directory for this report and term collection, if it exists. Otherwise, return None.'''
        term_folder_path = self._term_dir(terms)
        return term_folder_path if os.path.exists(term_folder_path) else None
    
    
    def _search_results_path(self, terms: TermCollection | str):
        '''The expected location of the search results file for this report and term collection. Does not check whether the file actually exists.'''
        return os.path.join(self._term_dir(terms), 'search-results.json')

    def search_results_path(self, terms: TermCollection | str):
        '''Get the path to the search results file for this report and term collection, if it exists. Otherwise, return None.'''
        search_results_path = self._search_results_path(terms)
        if not os.path.exists(search_results_path):
            return None
        return search_results_path
    
    def search_results(self, terms: TermCollection | str):
        '''Get the content of the search results file for this report and term collection, if it exists. Otherwise, return None.'''
        search_results_path = self.search_results_path(terms)
        if search_results_path is None:
            return None
        with open(search_results_path, 'r') as f:
            return json.load(f)
    
    def _prompts_dir(self, terms: TermCollection | str):
        '''The expected location of the prompts directory for this report and term collection. Does not check whether the directory actually exists.'''
        return os.path.join(self._term_dir(terms), 'prompts')

    def prompts_dir(self, terms: TermCollection | str):
        '''Get the path to the prompts directory for this report and term collection, if it exists. Otherwise, return None.'''
        prompts_dir = self._prompts_dir(terms)
        if not os.path.exists(prompts_dir):
            return None
        return prompts_dir
    
    def prompt_paths(self, terms: TermCollection | str) -> list[str]:
        '''Get the paths to all prompt files for this report and term collection, if they exist. Otherwise, raises a ValueError.'''
        prompts_dir = self.prompts_dir(terms)
        if prompts_dir is None:
            raise ValueError(f'Prompts directory not found for report {self.report_name} and term collection {terms} at expected location: {self._prompts_dir(terms)}')
        prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
        if len(prompt_files) == 0:
            raise ValueError(f'No prompt files found for report {self.report_name} and term collection {terms} in prompts directory at expected location: {prompts_dir}')
        return [os.path.join(prompts_dir, f) for f in prompt_files]
    

    def _prompt_path(self, terms: TermCollection | str, term: Term | str):
        '''The expected location of the prompt file for this report, term collection, and term alias. Does not check whether the file actually exists.'''
        if not isinstance(term, str):
            term = term.prompt_fname
        if not term.endswith('.txt'):
            term = term + '.txt'
        return os.path.join(self._prompts_dir(terms), f'{term}')

    def prompt_path(self, terms: TermCollection | str, term: Term | str):
        '''Get the path to the prompt file for this report, term collection, and term alias, if it exists. Otherwise, return None.'''
        prompt_path = self._prompt_path(terms, term)
        if not os.path.exists(prompt_path):
            return None
        return prompt_path
    
    def prompt(self, terms: TermCollection | str, term: Term | str):
        '''Get the content of the prompt file for this report, term collection, and term alias, if it exists. Otherwise, return None.'''
        prompt_path = self.prompt_path(terms, term)
        if prompt_path is None:
            return None
        return open(prompt_path).read()
    
    def _prompts_jsonl_path(self, terms: TermCollection | str):
        '''The expected location of the prompts jsonl file for this report and term collection. Does not check whether the file actually exists.'''
        return os.path.join(self._prompts_dir(terms), 'prompts.jsonl')
    
    def prompts_jsonl_path(self, terms: TermCollection | str):
        '''Get the path to the prompts jsonl file for this report and term collection, if it exists. Otherwise, return None.'''
        prompts_jsonl_path = self._prompts_jsonl_path(terms)
        if not os.path.exists(prompts_jsonl_path):
            return None
        return prompts_jsonl_path
            
@dataclass
class Company:
    folder_name: str
    full_name: str
    colour: str|None = None

    report_types = set(REPORT_TYPES)

    @classmethod
    def from_folder_name(cls, folder_name: str):
        folder_name = folder_name.lower()
        full_name = COMPANY_ALIASES.get(folder_name, None)
        if full_name is None:
            # try to read from metadata file
            metadata_file = os.path.join(COMPANIES_DIR, folder_name, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    full_name = metadata.get('full_name', None)
            else:
                raise ValueError(f'folder_name "{folder_name}" not found: no entry in COMPANY_ALIASES, no metadata file found at expected location: {metadata_file}')
        return cls(folder_name=folder_name, full_name=full_name)

    def __post_init__(self):
        self.folder_name = self.folder_name.lower()
        self.folder = os.path.join(COMPANIES_DIR, self.folder_name)

        if not os.path.exists(self.folder):
            # set up empty company folder for new companies if required
            os.makedirs(self.folder)
            for report_type in self.report_types:
                os.makedirs(os.path.join(self.folder, report_type), exist_ok=True)
        else:
            # enforce structure of existing company folders: companies/co.folder_name/report_type/report-year/report-year.pdf
            for report_type in self.report_types:
                if os.path.exists(os.path.join(self.folder, report_type)):
                    for filename in os.listdir(os.path.join(self.folder, report_type)):
                        file_path = os.path.join(self.folder, report_type, filename)
                        if os.path.isfile(file_path) and filename.endswith('.pdf'):
                            # convert to directory, and move file into it
                            new_folder_path = os.path.join(self.folder, report_type, filename.replace('.pdf', ''))
                            os.makedirs(new_folder_path, exist_ok=True)
                            os.rename(file_path, os.path.join(new_folder_path, filename))
                        elif os.path.isfile(file_path):
                            if file_path.endswith('.DS_Store'):  # ignore macOS folder metadata files
                                continue
                            raise ValueError(f'Unexpected file in company folder: {file_path}')
                else:
                    os.makedirs(os.path.join(self.folder, report_type), exist_ok=True)
        
        # create metadata file for company if it doesn't exist
        metadata_file = os.path.join(self.folder, 'metadata.json')
        if not os.path.exists(metadata_file):
            with open(metadata_file, 'w') as f:
                f.write(f'{{"full_name": "{self.full_name}", "folder_name": "{self.folder_name}"}}')

        logger.debug(f'CREATED Company: folder_name={self.folder_name}, full_name={self.full_name}')

    def __str__(self):
        return self.folder_name
    
    def reports(self, *, report_type:str = 'all', errors:Literal['raise', 'ignore']='ignore') -> list[Report]:
        '''
        Get a list of Report objects for this company and report type.

        Parameters
        ----------
        report_type : str, optional
            The type of report to get (e.g., "annual-report" or "sustainability-report"). 
            Must be a valid report type from REPORT_TYPES, or 'all' to get reports for all report types.
            Defaults to 'all'.
        errors : {'raise', 'ignore'}, optional
            How to handle errors when creating Report objects (e.g., due to missing files).
            'raise' will raise the error and stop execution, while 'ignore' will log a
            warning and skip the problematic report. Defaults to 'ignore'.

        Returns
        -------
        list[Report]
            A list of Report objects for this company and report type.
        '''
        if report_type not in self.report_types and report_type.lower() != 'all':
            raise ValueError(f'report_type should be one of {self.report_types}. Got {report_type}')
        
        out = []
        for report_folder in self.report_folders(report_type=report_type):
            try:
                out.append(Report(path=report_folder))
            except ValueError as e:
                if errors == 'raise':
                    raise e
                elif errors == 'ignore':
                    logger.warning(f'Error creating Report object for folder {report_folder}: {e}. Skipping this report.')
                else:
                    raise ValueError(f'Invalid value for errors parameter: {errors}. Must be one of {["raise", "ignore"]}')
        return out

    def report_folders(self, *, report_type: str = 'all') -> list[str]:
        '''
        Full paths to all report folders for this company and report type. 
        Report type can be 'all' to get folders for all report types.
        
        Parameters
        ----------
        report_type : str
            The type of report to get folders for (e.g., "annual-report" or "sustainability-report"). 
            Must be a valid report type from REPORT_TYPES, or 'all' to get folders for all report types.

        Returns
        -------
        list[str]
            A list of full paths to report folders for this company and report type.
        '''
        
        if report_type not in self.report_types:
            if report_type.lower() == 'all':
                report_folders = []
                for rt in self.report_types:
                    report_folders.extend(self.report_folders(report_type=rt))
                return report_folders
            raise ValueError(f'report_type should be one of {self.report_types}. Got {report_type}')
        
        report_type_folder = os.path.join(self.folder, report_type)
        
        if not os.path.exists(report_type_folder):
            raise ValueError(f'No folder found for {report_type} reports of company {self.folder_name} at expected location: {report_type_folder}')
        
        return [os.path.join(report_type_folder, fname) for fname in os.listdir(report_type_folder) if os.path.isdir(os.path.join(report_type_folder, fname))]

    def __hash__(self):
        return hash(self.folder_name)

class CompanyCollection:
    def __init__(self, companies: Iterable[Company]|dict[str, str]):
        '''
        A collection of companies. Can be initialized with either an iterable of Company objects, or a 
        dict mapping folder_name to full_name (e.g., {'shell': 'Shell'}). In the latter case, Company objects 
        will be created automatically, along with their corresponding folders.

        Parameters
        ----------
        companies : Iterable[Company] or dict[str, str]
            An iterable of Company objects, or a dict mapping folder_name to full_name (e.g., {'shell': 'Shell'}). 
            If a dict is provided, Company objects will be created automatically.
        '''
        
        if isinstance(companies, dict):
            companies = [Company(folder_name=folder_name, full_name=full_name, colour=COMPANY_COLOURS.get(folder_name, None)) for folder_name, full_name in companies.items()]
        self.companies = list(companies)
        self.folder = COMPANIES_DIR

        self.uuid = 'cos-' + uuid.uuid5(uuid.NAMESPACE_OID, ''.join([co.folder_name for co in self.companies])).hex

        logger.debug(f'CREATED CompanyCollection: uuid={self.uuid}, companies={[co.folder_name for co in self.companies]}')


    def __iter__(self):
        return iter(self.companies)
    
    def __getitem__(self, key) -> Union[Company, 'CompanyCollection']:
        if isinstance(key, slice):
            return CompanyCollection(self.companies[key])
        elif isinstance(key, str):
            try:
                return next(co for co in self.companies if co.folder_name == key or co.full_name.lower() == key)
            except StopIteration:
                raise KeyError(f'Company with folder_name or full_name "{key}" not found in collection.')
        elif isinstance(key, Iterable):
            return CompanyCollection([self[ki] for ki in key])
        elif isinstance(key, int):
            return self.companies[key]
        else:            
            raise TypeError(f'CompanyCollection indices must be str (folder_name or full_name) or int, not {type(key)}')

    def __len__(self):
        return len(self.companies)
    
    def __str__(self):
        return f'CompanyCollection(uuid="{self.uuid}", companies={[co.folder_name for co in self.companies]})'

    def __repr__(self):
        return self.__str__()

    def report_folders(self, *, report_type: str = 'all') -> list[str]:
        '''
        Full paths to all report folders for all companies in this collection and report type. 
        Report type can be 'all' to get folders for all report types.

        Parameters
        ----------
        report_type : str, optional
            The type of report to get folders for (e.g., "annual-report" or "sustainability-report"). 
            Must be a valid report type from REPORT_TYPES, or 'all' to get folders for all report types.
            Defaults to 'all'.

        Returns
        -------
        list[str]
            A list of full paths to report folders for all companies in this collection and report type.
        '''
        report_folders = []
        for co in self.companies:
            report_folders.extend(co.report_folders(report_type=report_type))
        return report_folders
    
    def reports(self, *, report_type: str = 'all') -> list[str]:
        '''
        Get a list of Report objects for all companies in this collection and report type.

        Parameters
        ----------
        report_type : str, optional
            The type of report to get files for (e.g., "annual-report" or "sustainability-report"). 
            Must be a valid report type from REPORT_TYPES, or 'all' to get files for all report types.
            Defaults to 'all'.
        
        Returns
        -------
        list[str]
            A list of full paths to report files for all companies in this collection and report type.
        '''
        report_files = []
        for co in self.companies:
            report_files.extend(co.reports(report_type=report_type))
        return report_files


### Utility functions

def co_files(
        yield_level: Literal['Report', 'report_folder', 'report_md', 'report_pdf', 'chunk_folder', 'chunk'], *, 
        companies: CompanyCollection|Company|None = None, report_types: Iterable[str]|None = None
    ) -> Iterable[str|Report]:
    '''
    Generator that yields all files (or Report objects) of the requested type for the given companies and report types.
    By default, will yield for all companies and report types.

    Parameters
    ----------
    yield_level : {'Report', 'report_folder', 'report_md', 'report_pdf', 'chunk_folder', 'chunk'}
        The type of file to yield. Each argument yields the following:
        - 'Report': `Report` objects.
        - 'report_folder': The path to the report folder - `report.dir`.
        - 'report_md': The path to the markdown file for the report, if it exists - `report.md_path`.
        - 'report_pdf': The path to the PDF file for the report, if it exists - `report.pdf_path`.
        - 'chunk_folder': The path to the chunks folder for the report, if it exists - `report.chunks_dir`.
        - 'chunk': The paths to all chunk files for the report, if they exist - `report.chunk_paths`.

        If a requested file does not exist for a given report, it will be skipped, so only valid file paths will be yielded.
    companies : CompanyCollection, Company, or None, optional
        The company or collection of companies to yield files for. If None, will yield for all companies 
        in the data directory. Defaults to None.
    report_types : Iterable[str] or None, optional
        The report types to yield files for (e.g., "annual-report" or "sustainability-report"). 
        If None, will yield for all report types. Defaults to None.
    '''
    if yield_level not in ['Report', 'report_folder', 'report_pdf', 'report_md',  'chunk_folder', 'chunk']:
        raise ValueError(f"yield_level must be one of 'Report', 'report_folder', 'report_pdf', 'report_md', 'chunk_folder', or 'chunk'. Got {yield_level}")
    
    if companies is None:
        companies = [co for co in os.listdir(COMPANIES_DIR) if os.path.isdir(os.path.join(COMPANIES_DIR, co))]
    else:
        if isinstance(companies, Company):
            companies = [companies.folder_name]
        else:
            companies = [co.folder_name for co in companies.companies]

    if report_types is None:
        report_types = REPORT_TYPES
    else:
        for rt in report_types:
            if rt not in REPORT_TYPES:
                raise ValueError(f'report_types should be a subset of {REPORT_TYPES}. Got {report_types}')

    for co in companies:
        co_dir = os.path.join(COMPANIES_DIR, co)
        if not os.path.isdir(co_dir):
            continue
        for report_type in report_types:
            if not os.path.isdir(os.path.join(co_dir, report_type)):
                # some companies may not have all report types
                continue
            report_type_dir = os.path.join(co_dir, report_type)
            if not os.path.isdir(report_type_dir):
                continue
            for report in os.listdir(report_type_dir):
                if report.startswith('.'):  # ignore hidden files/folders
                    continue
                try:
                    report = Report(os.path.join(report_type_dir, report))
                except (ValueError, NotADirectoryError, FileNotFoundError) as e:
                    logger.warning(f'Error creating Report object for report at {os.path.join(report_type_dir, report)}: {e}. Skipping this report.')
                    continue

                if yield_level == 'Report':
                    yield report
                elif yield_level == 'report_folder':
                    yield report.dir
                    continue
                elif yield_level == 'report_pdf':
                    if report.pdf_path is not None:
                        yield report.pdf_path
                    continue
                elif yield_level == 'report_md':
                    if report.md_path is not None:
                        yield report.md_path
                    continue
                elif yield_level == 'chunk_folder':
                    if report.chunks_dir is not None:
                        yield report.chunks_dir
                    continue
                elif yield_level == 'chunk':
                    if report.chunks_dir is not None:
                        for chunk_path in report.chunk_paths:
                            yield chunk_path
                    continue
                                    
def file_metadata(file_path:str) -> dict[str, str|Company|None]:
    '''
    Get the company, report type, report name, and year from a given filepath in the data directory.
    
    Parameters
    ----------
    file_path : str
        The path of the file to extract the company, report type, report name, and year from.
    
    Returns
    -------
    dict[str, str|Company|None]
        A dictionary with the following keys: 
        - 'company': The company associated with the file, as a Company object.
        - 'report_type': The type of the report, as a str.
        - 'report_name': The name of the report, as a str.
        - 'year': The year of the report, as a str.
        
        If any of these cannot be extracted from the file path, the corresponding value in the dictionary will be None.

    Raises
    ------
    ValueError
        If the company, report type, report name, or report year cannot be extracted from the file path.
    '''
    flist = os.path.relpath(file_path, COMPANIES_DIR).split(os.path.sep)
    co = Company.from_folder_name(flist[0]) if len(flist) > 0 else None
    rt = flist[1] if len(flist) > 1 else None
    rp = flist[2] if len(flist) > 2 else None
    if rp:
        ry = rp.split('-')[-1] if '-' in rp else None
    if co is None and rt is None and rp is None and ry is None:
        raise ValueError(f'Could not extract company, report type, report name, or report year from file path: {file_path}')
    
    return {
        'company': co,
        'report_type': rt,
        'report_name': rp,
        'year': ry
    }

### Initialize company collection
COMPANIES_ALL = CompanyCollection(COMPANY_ALIASES)