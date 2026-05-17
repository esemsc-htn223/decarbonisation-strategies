__all__ = ['Company', 'COS', 'company_aliases']

from dataclasses import dataclass
import os

from .util import DATA_DIR

@dataclass
class Company:
    foldername: str
    alias: str

    def __post_init__(self):
        self.foldername = self.foldername.lower()
        
        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)
        if not os.path.exists(self.md_folder):
            os.makedirs(self.md_folder)

    def __str__(self):
        return self.foldername
    
    @property
    def raw_folder(self):
        return os.path.join(DATA_DIR, 'raw', self.foldername)
    
    @property
    def md_folder(self):
        return os.path.join(DATA_DIR, 'md', self.foldername)

company_aliases = {
        'shell': 'Shell',
        'bp': 'BP',
        'exxon': 'ExxonMobil',
        'chevron': 'Chevron',
        'total': 'TotalEnergies'
    }

COS = [Company(co, alias) for co, alias in company_aliases.items()]