__all__ = ['Company', 'CompanyCollection', 'COS', 'COMPANY_ALIASES', 'co_dir_md_files']

from dataclasses import dataclass
import os
from typing import Iterable, Literal
import uuid
import json

from .util import COMPANIES_DIR, TERMS_DIR, logger, REPORT_TYPES


COMPANY_ALIASES = {
        'shell': 'Shell',
        'bp': 'BP',
        'exxon': 'ExxonMobil',
        'chevron': 'Chevron',
        'total': 'TotalEnergies'
    }

@dataclass
class Company:
    folder_name: str
    full_name: str

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

    def report_folders(self, report_type: str) -> list[str]:
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
                    report_folders.extend(self.report_folders(rt))
                return report_folders
            raise ValueError(f'report_type should be one of {self.report_types}. Got {report_type}')
        
        report_type_folder = os.path.join(self.folder, report_type)
        
        if not os.path.exists(report_type_folder):
            raise ValueError(f'No folder found for {report_type} reports of company {self.folder_name} at expected location: {report_type_folder}')
        
        return [os.path.join(report_type_folder, fname) for fname in os.listdir(report_type_folder) if os.path.isdir(os.path.join(report_type_folder, fname))]
    
    def report_files(self, report_type: str, file_type: str) -> list[str]:
        '''
        Full paths to all files of a specific type for this company and report type.
        Report type can  be 'all' to get files for all report types.
        Note if the files are not present (e.g., they have not yet been created), this will return an empty list.
        
        Parameters
        ----------
        report_type : str
            The type of report to get files for (e.g., "annual-report" or "sustainability-report"). 
            Must be a valid report type from REPORT_TYPES, or 'all' to get files for all report types.
        file_type : str
            The type of file to get (e.g., "pdf", "raw" or "md"). Can be specified with or without a leading dot (e.g., "pdf" or ".pdf").
            Accepted file types are:
            - "pdf", ".pdf", "raw": PDF report files
            - "md", ".md", "markdown": Markdown report files
            - "chunks": folders containing markdown chunks for each report
            - any valid term collection name (e.g., "all-terms"): folders containing search results, prompts, or jsonl files for the corresponding term collection.

        Returns
        -------
        list[str]
            A list of full paths to report files for this company and report type.
        '''

        file_type = file_type.lower()
        file_type_suffixes = {
            '.pdf': '.pdf',
            'pdf': '.pdf',
            'raw': '.pdf',

            '.md': '.md',
            'md': '.md',
            'markdown': '.md'
        }
        
        # report folders can contain chunks subfolder, or a terms folder if these chunks have been searched.
        folders = ['chunks'] + [term_collection for term_collection in os.listdir(TERMS_DIR) if os.path.isdir(os.path.join(TERMS_DIR, term_collection))]
        if file_type not in file_type_suffixes:
            # determine if file_type is an allowed folder
            if file_type in folders:
                return [os.path.join(foldername, file_type) for foldername in self.report_folders(report_type) if os.path.exists(os.path.join(foldername, file_type))]
            raise ValueError(f'file_type not found')
        
        if report_type not in self.report_types:
            if report_type.lower() == 'all':
                report_pdfs = []
                for rt in self.report_types:
                    report_pdfs.extend(self.report_pdfs(rt))
                return report_pdfs
            raise ValueError(f'report_type should be one of {self.report_types}. Got {report_type}')
        
        return [os.path.join(report_folder, filename) for report_folder in self.report_folders(report_type) for filename in os.listdir(report_folder) if filename.endswith(file_type_suffixes[file_type])]
    
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
            companies = [Company(folder_name=folder_name, full_name=full_name) for folder_name, full_name in companies.items()]
        self.companies = list(companies)
        self.folder = COMPANIES_DIR

        self.uuid = 'cos-' + uuid.uuid5(uuid.NAMESPACE_OID, ''.join([co.folder_name for co in self.companies])).hex

        logger.debug(f'CREATED CompanyCollection: uuid={self.uuid}, companies={[co.folder_name for co in self.companies]}')


    def __iter__(self):
        return iter(self.companies)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.lower()
            for co in self.companies:
                if co.folder_name == key or co.full_name.lower() == key:
                    return co
            raise KeyError(f'Company with folder_name or full_name "{key}" not found in collection.')
        elif isinstance(key, int):
            return self.companies[key]
        else:            
            raise TypeError(f'CompanyCollection indices must be str (folder_name or full_name) or int, not {type(key)}')

    def __len__(self):
        return len(self.companies)
    
    def report_folders(self, report_type: str) -> list[str]:
        '''
        Full paths to all report folders for all companies in this collection and report type. 
        Report type can be 'all' to get folders for all report types.

        Parameters
        ----------
        report_type : str
            The type of report to get folders for (e.g., "annual-report" or "sustainability-report"). 
            Must be a valid report type from REPORT_TYPES, or 'all' to get folders for all report types.

        Returns
        -------
        list[str]
            A list of full paths to report folders for all companies in this collection and report type.
        '''
        report_folders = []
        for co in self.companies:
            report_folders.extend(co.report_folders(report_type))
        return report_folders
    
    def report_files(self, report_type: str, file_type: str) -> list[str]:
        '''
        Full paths to all files of a specific type for all companies in this collection and report type.
        Report type can  be 'all' to get files for all report types.
        Note if the files are not present (e.g., they have not yet been created), this will return an empty list.

        Parameters
        ----------
        report_type : str
            The type of report to get files for (e.g., "annual-report" or "sustainability-report"). 
            Must be a valid report type from REPORT_TYPES, or 'all' to get files for all report types.
        file_type : str
            The type of file to get (e.g., "pdf", "raw" or "md"). Can be specified with or without a leading dot (e.g., "pdf" or ".pdf").
            Accepted file types are:
            - "pdf", ".pdf", "raw": PDF report files
            - "md", ".md", "markdown": Markdown report files
            - "chunks": folders containing markdown chunks for each report
            - any valid term collection name (e.g., "all-terms"): folders containing search results, prompts, or jsonl files for the corresponding term collection.
        
        Returns
        -------
        list[str]
            A list of full paths to report files for all companies in this collection and report type.
        '''
        report_files = []
        for co in self.companies:
            report_files.extend(co.report_files(report_type, file_type))
        return report_files


### Utility functions

def co_files(
        yield_level: Literal['report_folder', 'report_md', 'report_pdf', 'chunk_folder', 'chunk'], *, 
        companies: CompanyCollection|Company|None = None, report_types: Iterable[str]|None = None
    ) -> Iterable[str]:
    '''
    Generator that yields the paths of all markdown files/directories for all companies and report types.
    '''
    if yield_level not in ['report_folder', 'report_md', 'report_pdf', 'chunk_folder', 'chunk']:
        raise ValueError(f"yield_level must be one of 'report_folder', 'report_md', 'report_pdf', 'chunk_folder', or 'chunk'. Got {yield_level}")
    
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

                report = os.path.join(report_type_dir, report)
                if not os.path.isdir(report):
                    if report.endswith('.pdf') and yield_level == 'report_pdf':
                        yield report
                    continue
                elif yield_level == 'report_folder':
                    yield report
                    continue
                for report_file in os.listdir(report):
                    report_file = os.path.join(report, report_file)
                    
                    # full reports
                    if yield_level == 'report_md':
                        if report_file.endswith('.md'):
                            yield report_file

                    elif yield_level == 'report_pdf':
                        if report_file.endswith('.pdf'):
                            yield report_file
                    
                    # chunked reports
                    elif 'chunk' in yield_level:
                        if os.path.isdir(report_file) and os.path.basename(report_file) == 'chunks':
                            # chunk folders
                            if yield_level == 'chunk_folder':
                                yield report_file
                            # individual chunk files
                            elif yield_level == 'chunk':
                                for chunk_file in os.listdir(report_file):
                                    if chunk_file.endswith('.md'):
                                        yield os.path.join(report_file, chunk_file)
                                    
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
    flist = file_path.replace(COMPANIES_DIR + os.path.sep, '').split(os.path.sep)
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
COS = CompanyCollection(COMPANY_ALIASES)