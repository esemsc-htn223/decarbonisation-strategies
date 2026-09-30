import marimo

__generated_with = "0.24.2"
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
    from decarb import regex, llm, output

    return TERMS_ALL, llm, mo, os, output, pd, plt, regex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Main pipeline
    """)
    return


@app.cell
def _(llm):
    llm.stage_convert(overwrite=False, progress_bar=True)
    return


@app.cell
def _(llm):
    llm.stage_chunk(overwrite=True)
    return


@app.cell
def _(llm):
    llm.stage_search(overwrite=True)
    return


@app.cell
def _(llm):
    llm.stage_draft(overwrite=True, progress_bar=True, overlap_strategy = None)
    return


@app.cell
def _(llm):
    job = llm.stage_submit(
        overwrite=True,
        portion_strategy='max_tokens', max_tokens = 750_000,
        progress_bar=False
    )
    job
    return (job,)


@app.cell
def _(job, llm):
    llm.stage_retrieve(job, progress_bar=True, verbose=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prompt functionality
    """)
    return


@app.cell
def _(llm, pd):
    inspect_job = llm.prompt.Job('da751a76a582531ca537f181f2acdab5')

    df = { 'portion_index': [], 'n_prompts': [], 'token_count': [], 'start_index': [], 'end_index': [] }

    for i, portion in enumerate(llm.prompt.jsonl_portions(inspect_job.prompt_jsonl)):
        df['portion_index'].append(i)
        df['n_prompts'].append(len(portion))
        df['token_count'].append(portion.token_count)
        df['start_index'].append(portion.start_index)
        df['end_index'].append(portion.end_index)

    df = pd.DataFrame(df)
    df
    return (inspect_job,)


@app.cell
def _(inspect_job):
    inspect_job.stats
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
def _(chunks, output):
    output.marimo.min_max_chunks(chunks)
    return


@app.cell
def _(chunks, output):
    # display a random sample of chunks
    output.marimo.random_chunks(chunks, n=5)
    return


@app.cell
def _(chunks, output, plt):
    output.plot.chunk_size_hist(chunks)

    plt.show()
    return


@app.cell
def _(TERMS_ALL, chunks, output, regex):
    terms = TERMS_ALL

    search_res = regex.search(chunks, terms)
    output.marimo.search_summary(chunks, terms, search_res=search_res)
    return (search_res,)


@app.cell
def _(chunks, output, plt, search_res):
    fig, ax = output.plot.term_search_results(search_res, chunks)
    plt.grid(which = 'major', axis = 'y', color = 'k')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
