from ..companies import Company



def construct_prompt(company:str|Company, report_type:str, year:int|str):

    if isinstance(company, str):
        co_alias = company
    elif isinstance(company, Company):
        co_alias = company.alias
    else:
        raise ValueError(f"company should be either the name of a company (as a str), or a Company object. Got {type(company)}")
    
    report_alias = report_type.replace('-', ' ')
    
    header = f'''The following extracts are taken from the {report_alias} of {co_alias} from {year}.


'''
