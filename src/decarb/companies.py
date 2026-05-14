__all__ = ['Company', 'COS', 'company_aliases']

from dataclasses import dataclass
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')

@dataclass
class Company:
    foldername: str
    alias: str

    def __str__(self):
        return self.foldername
    
    @property
    def folder(self):
        return os.path.join(DATA_DIR, self.foldername)

company_aliases = {
        'shell': 'Shell',
        'bp': 'BP',
        'exxon': 'ExxonMobil',
        'chevron': 'Chevron',
        'total': 'TotalEnergies'
    }

COS = [Company(co, alias) for co, alias in company_aliases.items()]