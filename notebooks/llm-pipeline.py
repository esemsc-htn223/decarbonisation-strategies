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
    from decarb.companies import Company, CompanyCollection, COMPANIES_ALL, REPORT_TYPES
    from decarb import regex, llm, output

    return COMPANIES_ALL, REPORT_TYPES, TERMS_SIMPLE, llm, mo


@app.cell
def _(COMPANIES_ALL, REPORT_TYPES, TERMS_SIMPLE, llm):
    job = llm.pipeline(
        COMPANIES_ALL, REPORT_TYPES, TERMS_SIMPLE,
        progress_bars = True,
        chunk_kwargs = dict(
            overwrite = True
        ),
        search_kwargs = dict(
            overwrite = True
        ),
        draft_kwargs = dict(
            overwrite = True, overlap_strategy = None
        ),
        submit_kwargs = dict(
            overwrite=False,
            portion_strategy='max_tokens', max_tokens = 750_000,
            progress_bar=False
        )
    )


    job
    return (job,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above can also be run step-wise:

    ```python
    llm.stage_convert(overwrite=False, progress_bar=True)
    llm.stage_chunk(overwrite=True)
    llm.stage_search(overwrite=True)
    llm.stage_draft(overwrite=True, progress_bar=True, overlap_strategy = None)
    job = llm.stage_submit(
        overwrite=True,
        portion_strategy='max_tokens', max_tokens = 750_000,
        progress_bar=False
    )
    ```
    """)
    return


@app.cell
def _(job):
    # retrieve results for completed prompts, but don't wait for those still working.
    job.retrieve(wait = False, progress_bar=True)
    return


@app.cell
def _(llm):
    # or, use stage retrieve to retrieve all jobs still working.
    llm.stage_retrieve(jobs = None,  # default - retrieve all jobs
                       wait = True, # wait for all jobs to complete before returning
                       progress_bar=True, verbose=True)
    return


@app.cell
def _(job):
    df_results = job.to_df()
    df_results
    return


if __name__ == "__main__":
    app.run()
