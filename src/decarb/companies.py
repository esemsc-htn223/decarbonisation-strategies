__all__ = ['Company', 'COS', 'company_aliases']

from dataclasses import dataclass

@dataclass
class Company:
    foldername: str
    alias: str

    def __str__(self):
        return self.foldername

company_aliases = {
        'shell': 'Shell',
        'bp': 'BP',
        'exxon': 'ExxonMobil',
        'chevron': 'Chevron',
        'total': 'TotalEnergies'
    }

COS = [Company(co, alias) for co, alias in company_aliases.items()]