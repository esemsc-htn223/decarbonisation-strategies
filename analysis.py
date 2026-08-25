import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="decarb")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0. Imports & Setup
    """)
    return


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import tiktoken
    import json
    import sys
    import os
    import altair as alt
    import re
    import requests
    from typing import Literal
    from dotenv import load_dotenv
    load_dotenv()

    from decarb.terms import Term, TermCollection, TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE
    from decarb import companies, llm, util, output

    return TERMS_ALL, llm, mo, output, util


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial output analysis
    """)
    return


@app.cell
def _(llm):
    job_1 = llm.prompt.Job('a2d4064f54b85a0691382e53cc935b1e')
    job_2 = llm.prompt.Job('76a1c482f11652c5ba9e3f8c4b4771ae')

    # pull live batch statuses, adopt any batches that belong to these jobs but were never
    # recorded against them, and locate result files on disk
    for _job in (job_1, job_2):
        _job.refresh()

    job_1, job_2
    return job_1, job_2


@app.cell
def _(job_1, job_2):
    # coverage check: every prompt should either have a result, or show up as missing
    {
        _job.id: {
            'status': _job.status,
            'batches': _job.n_batch_statuses,
            'results': len(_job.output_paths),
            'completed': f'{len(_job.completed_prompts)}/{_job.n_prompts}',
            'missing': _job.missing_prompts,
        }
        for _job in (job_1, job_2)
    }
    return


@app.cell
def _(llm, job_1, job_2):
    df_test = llm.results.to_df(job=[job_1, job_2])
    df_test
    return (df_test,)


@app.cell
def _(TERMS_ALL, df_test, output):
    output.marimo.results_tabs(df_test, TERMS_ALL)
    return


@app.cell
def _(df_test, mo, output):
    mo.ui.tabs({
        'Sentiment': output.plot.llm_terms_heatmap(df_test, 'q1_sentiment', figsize=(9,15), return_type='figure'),
        'Maturity': output.plot.llm_terms_heatmap(df_test, 'q2_maturity', figsize=(9,15), return_type='figure'),
        'Stage': output.plot.llm_terms_heatmap(df_test, 'q3_stage', figsize=(9,15), return_type='figure')
    }, label = 'Question', lazy = False)
    return


@app.cell
def _(df_test):
    df_test['q1_sentiment'].round(decimals = 1).value_counts()
    return


@app.cell
def _(TERMS_ALL, df_test, plot_terms_heatmap, util):
    for _q in ['q1_sentiment', 'q2_maturity', 'q3_stage']:
        _fig = plot_terms_heatmap(df_test, _q, figsize=(9,15), return_type='figure')
        _fig.savefig(f'{util.OUT_DIR}/{TERMS_ALL.folder_name}/heatmap-{_q}-20260805.png', dpi=300)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
