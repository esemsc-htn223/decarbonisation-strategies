import marimo

__generated_with = "0.23.9"
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
    from decarb.companies import Company, CompanyCollection, COS
    from decarb.llm import rag, inspection, pipeline

    return COS, TERMS_ALL, inspection, mo, os, pipeline, plt, rag


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
def _():
    import json

    with open('/Users/hugoneely/Downloads/batch_6a5a23ec80f8819081fc92c27ea2c2ed_output.jsonl', 'r') as f:
        jsonl = f.read()

    jsonl = jsonl.splitlines()
    jsonl = [json.loads(line) for line in jsonl]
    jsonl
    return json, jsonl


@app.cell
def _(json, jsonl):
    json.loads(jsonl[0]['response']['body']['choices'][0]['message']['content'])
    return


@app.cell
def _(COS, pipeline):
    pipeline.stage_submit(overwrite=True, companies = COS['exxon'])
    return


@app.cell
def _(pipeline):
    pipeline.stage_retrieve(progress_bar=True, verbose=True)
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
