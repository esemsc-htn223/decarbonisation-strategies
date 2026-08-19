import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM pipeline
    """)
    return


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import matplotlib.pyplot as plt
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from decarb.terms import TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE
    from decarb.companies import Company, CompanyCollection, COMPANIES_ALL
    from decarb.llm import rag, inspection, pipeline, prompt

    return TERMS_ALL, inspection, mo, os, pd, pipeline, plt, prompt, rag


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Main pipeline
    """)
    return


@app.cell
def _(pipeline):
    pipeline.stage_convert(overwrite=False, progress_bar=True)
    return


@app.cell
def _(pipeline):
    pipeline.stage_chunk(overwrite=True)
    return


@app.cell
def _(pipeline):
    pipeline.stage_search(overwrite=True)
    return


@app.cell
def _(pipeline):
    pipeline.stage_draft(overwrite=True, progress_bar=True, overlap_strategy = None)
    return


@app.cell
def _(pipeline):
    pipeline.stage_submit(
        overwrite=True, 
        portion_strategy='max_tokens', max_tokens = 750_000, 
        progress_bar=False
    )
    return


@app.cell
def _(pipeline):
    pipeline.stage_retrieve(progress_bar=True, verbose=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prompt functionality
    """)
    return


@app.cell
def _(pd, prompt):

    jsonl_path = '/Users/hugoneely/Documents/4-work/1-current/PhD/repos/decarbonisation-strategies/data/terms/all-terms/prompts/da751a76a582531ca537f181f2acdab5.jsonl'

    df = { 'portion_index': [], 'n_prompts': [], 'token_count': [], 'start_index': [], 'end_index': [] }

    for i, portion in enumerate(prompt.jsonl_portions(jsonl_path)):
        df['portion_index'].append(i)
        df['n_prompts'].append(len(portion))
        df['token_count'].append(portion.token_count)
        df['start_index'].append(portion.start_index)
        df['end_index'].append(portion.end_index)

    df = pd.DataFrame(df)
    df
    return (jsonl_path,)


@app.cell
def _(jsonl_path, prompt):
    prompt.jsonl_stats(jsonl_path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspection
    """)
    return


@app.cell
def _(os):
    _chunks_dir = 'data/companies/bp/annual-report/bp-annual-report-accounts-2007/chunks'
    chunks = [open(os.path.join(_chunks_dir, _f), 'r').read() for _f in os.listdir(_chunks_dir) if _f.endswith('.md')]

    {
        #'len(md)': len(md),
        'sum(out)': sum(len(c) for c in chunks),
        'no. chunks': len(chunks)

    }
    return (chunks,)


@app.cell
def _(chunks, inspection):
    inspection.min_max_chunks(chunks)
    return


@app.cell
def _(chunks, inspection):
    # display a random sample of chunks
    inspection.random_chunks(chunks, n=5)
    return


@app.cell
def _(chunks, inspection, plt):
    inspection.chunk_size_hist(chunks)

    plt.show()
    return


@app.cell
def _(TERMS_ALL, chunks, inspection, rag):
    terms = TERMS_ALL

    search_res = rag.search_chunks(chunks, terms)
    inspection.search_summary(chunks, terms, search_res=search_res)
    return (search_res,)


@app.cell
def _(chunks, inspection, plt, search_res):
    fig, ax = inspection.plot_search_results(search_res, chunks)
    plt.grid(which = 'major', axis = 'y', color = 'k')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
