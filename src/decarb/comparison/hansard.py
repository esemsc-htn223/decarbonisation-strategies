import requests
from pydantic import BaseModel, Field

HANSARD_API_BASE_URL = "https://hansard-api.parliament.uk"
HANSARD_API_SPEC = requests.get(f"{HANSARD_API_BASE_URL}/swagger/docs/v1").json()

def validate_parameters(params: dict, endpoint: str) -> None:
    """
    Validate the parameters against the Hansard API specification.

    Parameters
    ----------
    params : dict
        The parameters to validate.
    endpoint : str
        The endpoint to validate against.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    if endpoint not in HANSARD_API_SPEC['paths']:
        raise ValueError(f"Endpoint '{endpoint}' not found in Hansard API specification.")

    valid_params = {param['name'] for param in HANSARD_API_SPEC['paths'][endpoint]['get']['parameters']}
    
    n_params_with_qp_prefix = 0
    for param in params:
        if param not in valid_params and f'queryParameters.{param}' not in valid_params:
            raise ValueError(f"Parameter '{param}' is not valid for endpoint '{endpoint}'.")
        if param.startswith('queryParameters.'):
            n_params_with_qp_prefix += 1
    
    if n_params_with_qp_prefix != 0 and n_params_with_qp_prefix != len(params):
        raise ValueError("All parameters must either have the 'queryParameters.' prefix or none of them should.")
        

def search_debates(term:str | None = None, *, take: int | None = 100, skip: int | None = None, **kwargs) -> dict:
    '''
    Search debates in the UK Parliament Hansard.

    Parameters
    ----------
    term : str or None, optional
        The search term to filter debates. If None, all debates will be returned.
    take : int | None, optional
        The number of results to return, between 0 and 100. Defaults to 100.
    skip : int | None, optional
        The number of results to skip. Defaults to None.
    **kwargs : dict
        Additional keyword arguments to pass to the requests.get() function. See HANSARD_API_SPEC for valid parameters,
        or go to https://hansard-api.parliament.uk/swagger/ui/index.

    
    Returns
    -------
    dict
        A dictionary containing the search results from the Hansard API.
    '''
    path = "/search/debates.{format}"
    params = {
        "take": take,
        "skip": skip,
    }
    if term:
        params["searchTerm"] = term
    if kwargs:
        params.update(kwargs)
    
    validate_parameters(params, path)

    url = f"{HANSARD_API_BASE_URL}{path}".format(format="json")
    response = requests.get(url, params=params)
    return response.json()

