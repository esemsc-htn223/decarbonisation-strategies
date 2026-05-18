import marimo as mo
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .rag import search_chunks
from ..regex.terms import (
    Term, TermCollection, TERMS_ALL
)



def search_summary(chunks: list[str], terms: Term|TermCollection = TERMS_ALL, *, search_res: dict[str, list[int]]|None = None) -> mo.Html:
    '''
    Create a marimo accordian showing search results for the given terms and chunks.

    Parameters
    ----------
    chunks : list[str]
        The markdown chunks that were searched.
    search_res : dict[str, list[int]]
        The search results, as returned by `search_chunks`.
    terms : Term | TermCollection
        The term or collection of terms that were searched for.

    Returns
    -------
    mo.Html
        A marimo accordion showing the search results, with terms as top-level accordions and chunks
    '''
    

    if search_res is None:
        search_res = search_chunks(chunks, terms)

    master_accordion_content = {}
    for term in terms:
        # display the chunks where the term was found, with the term highlighted
        term_accordion_content = {f'Chunk {i}': chunks[i] for i in search_res[term.alias]}
        n_matches = {}
        for label, chunk in term_accordion_content.items():
            matches = re.finditer(term.regex, chunk, re.I)
            prev_shift = 0
            n_matches[label] = 0
            for match in matches:
                start, end = match.span()
                chunk = chunk[:start + prev_shift] + '<mark>' + chunk[start + prev_shift:end + prev_shift] + '</mark>' + chunk[end + prev_shift:]
                prev_shift += len('<mark></mark>')
                n_matches[label] += 1
            term_accordion_content[label] = mo.md(chunk)

        # rename with number of matches and formatting
        for label in list(term_accordion_content.keys()):
            if label in term_accordion_content:
                term_accordion_content[f'>**{label}:** {n_matches[label]} match{n_matches[label] > 1 and "es" or ""}'] = term_accordion_content.pop(label)
        
        # add to master accordion with number of chunks where term was found
        n_chunks = len(search_res[term.alias])
        _s = 's' if n_chunks != 1 else ''
        master_accordion_content[f'**{term.alias}:** {n_chunks} chunk{_s}'] = mo.vstack([
            #mo.md(f"Term: **{_term.alias}**"),
            mo.md(f"`{term.regex}`"),
            mo.accordion(term_accordion_content, multiple=True)
        ])


    return mo.accordion(master_accordion_content, multiple=False)


def plot_search_results(search_res, chunks, ax = None):
    _im = np.array(len(search_res) * [len(chunks) * [0]])
    for _i, _term in enumerate(search_res.keys()):
        _res = search_res[_term]
        _im[_i] = [1 if i in _res else 0 for i in range(len(chunks))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()
    ax.imshow(_im, aspect='auto', cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
    ax.set_yticks(range(len(search_res)))
    ax.set_yticklabels(list(search_res.keys()))
    ax.set_xlabel('Chunk Index')
    ax.set_title('Search Results for Terms in Chunks')
    ax.legend(handles = [mpl.patches.Patch(color='green', label='Term Found'), mpl.patches.Patch(color='white', label='Term Not Found')])

    return fig, ax