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
    out_paths = [
        # 'out/all-terms/batch-results/batch_6a60caf7295481908c3c9cc52bd6f2c8.jsonl',
        # 'out/all-terms/batch-results/batch_6a60da0430d881909ea42022141aad34.jsonl',
        # 'out/all-terms/batch-results/batch_6a60fc542fd8819084bc166c9885ebc3.jsonl',
        'out/all-terms/batch-results/batch_6a610c19e480819085e5b33248cc5cfa.jsonl',
        'out/all-terms/batch-results/batch_6a610d0e82c48190a490bf81e9fc5b12.jsonl',
        'out/all-terms/batch-results/batch_6a610f6b8d408190855b85f4331e0c8f.jsonl',
        'out/all-terms/batch-results/batch_6a6112447eb481908e44b6f36573d747.jsonl',
        'out/all-terms/batch-results/batch_6a6113395564819087a411f5648cd79f.jsonl',    
        'out/all-terms/batch-results/batch_6a6113b453c481909f5f12c1c1c3a20b.jsonl',
        'out/all-terms/batch-results/batch_6a61142f989c81909c2c4f558cb3cf27.jsonl',
        'out/all-terms/batch-results/batch_6a61159cd1d08190891bb3ee7b4c0a6a.jsonl',
        'out/all-terms/batch-results/batch_6a611709108081909afaab1abe97cd3c.jsonl',
        'out/all-terms/batch-results/batch_6a6117fcb434819098a82b5c763bd8aa.jsonl',
        'out/all-terms/batch-results/batch_6a611968327481909482476eb9af0e68.jsonl',

        'out/all-terms/batch-results/batch_6a6343d861908190b26f31d713b50aef.jsonl',
        'out/all-terms/batch-results/batch_6a64aaa5a5548190906d00010bb79cf6.jsonl',
        'out/all-terms/batch-results/batch_6a65072ba3288190a045f3e9f00edab8.jsonl',
        'out/all-terms/batch-results/batch_6a65091189e88190886efe6d7ba4f0c3.jsonl',
        'out/all-terms/batch-results/batch_6a650d31840881908b56ccb999c7805f.jsonl',
    ]

    df_test = llm.results.to_df(
        in_jsonl_paths=[
            'data/terms/all-terms/prompts/a2d4064f54b85a0691382e53cc935b1e.jsonl', 
            'data/terms/all-terms/prompts/76a1c482f11652c5ba9e3f8c4b4771ae.jsonl'
        ],
        out_jsonl_paths=out_paths
    )

    df_test
    return (df_test,)


@app.cell
def _(TERMS_ALL, df_test, output):
    output.marimo.results_tabs(df_test, TERMS_ALL)
    return


@app.cell
def _(df_test, mo, output):
    mo.ui.tabs({
        'Sentiment': output.plot.llm_plot_terms_heatmap(df_test, 'q1_sentiment', figsize=(9,15), return_type='figure'),
        'Maturity': output.plot.llm_plot_terms_heatmap(df_test, 'q2_maturity', figsize=(9,15), return_type='figure'),
        'Stage': output.plot.llm_plot_terms_heatmap(df_test, 'q3_stage', figsize=(9,15), return_type='figure')
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
