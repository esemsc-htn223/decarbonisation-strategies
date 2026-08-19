import marimo

__generated_with = "0.23.16"
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

    return TERMS_ALL, TermCollection, companies, json, mo, np, output, pd, util


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial output analysis
    """)
    return


@app.cell
def _(df_test):
    df_test
    return


@app.cell
def _(TERMS_ALL, TermCollection, companies, json, np, pd):
    default_missing = {
        'q1_sentiment': np.nan,
        'q1_sentiment_quotes': [],
        'q2_maturity': np.nan,
        'q2_maturity_quotes': [],
        'q3_stage': pd.NA,
        'q3_stage_quotes': [],
        'q4_commitments': [],
        'notes': None,
    }
    def mark_missing(df: pd.DataFrame, terms:TermCollection, prompt_dict: dict, *, 
                     missing_report_values:dict = default_missing, missing_term_values:dict = default_missing) -> pd.DataFrame:
        '''
        Mark missing rows in the dataframe for each company, report type, year, and term combination.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe containing company, report type, year, and term information.
        terms : TermCollection
            A collection of terms to check for missing values.
        prompt_dict : dict
            A dictionary mapping prompt IDs to prompts.
        missing_report_values : dict, optional
            A dictionary specifying default values for missing report types. Any columns not found in this dictionary
            will be filled with pandas.NA. Default values:
            - 'q1_sentiment': np.nan
            - 'q1_sentiment_quotes': []
            - 'q2_maturity': -1
            - 'q2_maturity_quotes': []
            - 'q3_stage': pd.NA
            - 'q3_stage_quotes': []
            - 'q4_commitments': []
            - 'notes': None
        missing_term_values : dict, optional
            A dictionary specifying default values for missing terms. Any columns not found in this dictionary
            will be filled with pandas.NA. For default values, see `missing_report_values`.

        Returns
        -------
        pd.DataFrame
            A new dataframe with missing rows added for each company, report type, year, and term
        '''
        question_cols = df.drop(columns=['company', 'company_alias', 'report_type', 'term_alias', 'term_regex', 'term_fname', 'year', 'prompt', 'term_regex_found', 'report_type_published']).columns.to_list()
        for col in missing_report_values.keys():
            if col not in question_cols:
                raise ValueError(f'Column "{col}" specified in missing_report_values is not found in the dataframe.')
        for col in missing_term_values.keys():
            if col not in question_cols:
                raise ValueError(f'Column "{col}" specified in missing_term_values is not found in the dataframe.')

        new_rows = {}

        make_index = lambda company, report_type, year, term: f"{company}_{report_type}_{year}_{term.prompt_fname.replace('.txt', '')}"

        for company in df['company'].unique():
            for report_type in df['report_type'].unique():
                df_co_rt = df[(df['company'] == company) & (df['report_type'] == report_type)]

                # 1. check years report was not published
                years_report_missing = set(df['year'].unique()) - set(df_co_rt['year'].unique())

                if df_co_rt.empty:
                    raise ValueError(f'No data found for company "{company}" and report type "{report_type}".')

                for term in terms:
                    years_term_missing = set(df['year'].unique()) - set(df_co_rt.loc[df_co_rt['term_alias'] == term.alias, 'year'].unique()) - set(years_report_missing)
                    for yr in years_report_missing:
                        index = make_index(company, report_type, yr, term)
                        new_rows[index] = {
                            'company': company,
                            'company_alias': companies.COMPANY_ALIASES.get(company, company),
                            'report_type': report_type,
                            'term_alias': term.alias,
                            'term_regex': term.regex,
                            'term_fname': term.prompt_fname,
                            'year': str(yr),
                            'prompt': prompt_dict.get(index, pd.NA),
                            **{col: missing_report_values.get(col, pd.NA) for col in question_cols},
                            'term_regex_found': False,
                            'report_type_published': False,
                        }

                    for yr in years_term_missing:
                            index = make_index(company, report_type, yr, term)
                            new_rows[index] = {
                                'company': company,
                                    'company_alias': companies.COMPANY_ALIASES.get(company, company),
                                    'report_type': report_type,
                                    'term_alias': term.alias,
                                    'term_regex': term.regex,
                                    'term_fname': term.prompt_fname,
                                    'year': str(yr),
                                    'prompt': prompt_dict.get(index, pd.NA),
                                    **{col: missing_term_values.get(col, pd.NA) for col in question_cols},
                                    'term_regex_found': False,
                                    'report_type_published': True,
                                }
        return pd.concat([
            df, 
            pd.DataFrame(list(new_rows.values()), index=new_rows.keys())
        ], axis=0).sort_values(by=['company', 'report_type', 'year', 'term_alias'])

    def results_df(*, in_jsonl_paths: list[str]|str, out_jsonl_paths:list[str]|str, terms: TermCollection = TERMS_ALL,
                   label_missing: bool = True, missing_report_values: dict = default_missing, missing_term_values: dict = default_missing, 
                   fillna_values:dict|None = default_missing) -> pd.DataFrame:
        prompt_dict = {}
        if isinstance(in_jsonl_paths, str):
            in_jsonl_paths = [in_jsonl_paths]
        for in_jsonl_path in in_jsonl_paths:
            with open(in_jsonl_path, 'r') as _f:
                for line in _f:
                    json_line = json.loads(line)
                    prompt_dict[json_line['custom_id']] = json_line['body']['messages'][0]['content']

        responses = []

        if isinstance(out_jsonl_paths, str):
            out_jsonl_paths = [out_jsonl_paths]

        for out_jsonl_path in out_jsonl_paths:
            for entry in open(out_jsonl_path, 'r'):  # file locally saved
                entry = json.loads(entry)
                id = entry['custom_id']
                co, rt, yr, term = id.split('_')
                entry = json.loads(entry['response']['body']['choices'][0]['message']['content'])
                term = terms[term]

                entry['id'] = id
                entry['company'] = co
                entry['report_type'] = rt
                entry['year'] = yr
                entry['term_alias'] = term.alias
                entry['company_alias'] = companies.COMPANY_ALIASES.get(co, co)
                entry['term_regex'] = term.regex
                entry['term_fname'] = term.prompt_fname
                entry['prompt'] = prompt_dict[id]
                entry['term_regex_found'] = True
                entry['report_type_published'] = True
                responses.append(entry)


        first_cols = ['company', 'report_type', 'year', 'term_alias']
        df = pd.DataFrame(responses).set_index('id')
        cols = first_cols + df.drop(columns=first_cols).columns.to_list()
        df = df[cols]

        if label_missing:
            df = mark_missing(df, terms, prompt_dict, missing_report_values=missing_report_values, missing_term_values=missing_term_values)
        df = df.convert_dtypes()

        if fillna_values is not None:
            df = df.fillna({col: val for col, val in fillna_values.items() if col in df.columns and not isinstance(val, list)})

        return df

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

    df_test = results_df(
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
