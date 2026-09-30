import marimo

__generated_with = "0.24.2"
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
    from decarb import companies, llm, util, output, regex



    return (
        TERMS_ALL,
        TERMS_SIMPLE,
        companies,
        llm,
        mo,
        np,
        output,
        pd,
        plt,
        regex,
        util,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IEA TRL alignment
    """)
    return


@app.cell
def _(TERMS_SIMPLE, pd):
    df_iea = pd.read_csv('data/comparisons/IEACleanTechGuidepublicdataset.csv')

    #df_iea['sector'].str.split(',').explode().str.strip().unique().tolist()
    df_iea.loc[
        #df_iea['category.1'].isin(['Renewables', 'Hydrogen', 'Energy networks and storage', 'Carbon Capture and Storage', 'Fossil fuels'])
        #& df_iea['category.2'].str.contains('|'.join(TERMS_SIMPLE.regexes), regex=True, case=False)
        df_iea['category.2'].str.contains(TERMS_SIMPLE.regexes[1], regex=True, case=False)
        ]
    return (df_iea,)


@app.cell
def _(df_iea):
    df_iea.loc[df_iea['Energy Storage.cc'].notna()]
    return


@app.cell
def _(df_iea):
    df_iea['Energy Storage.cc'].value_counts().sort_index()
    return


@app.cell
def _(df_iea):
    df_iea['category.2'].value_counts().sort_index()
    return


app._unparsable_cell(
    r"""
    {
      "Solar": {"category.2":"Solar"},
      "Wind": {"category.2":"Wind"},
      "Hydrogen": {"category.2":"H₂ infrastructure"},
      "CCS" : {"category.2":["CO2 capture", "CO2 storage"]},
      "Biofuel": {"category.2":"Bioenergy"},
      "Geothermal":{"category.2":"Geothermal"},
      "Tidal": None,
      "Storage": {'category.2': 'Electrochemical storage', 'Energy Storage.cc': 'Batteries'},
      "EVs",
      "Nature",
      "Nuclear",
      "Fusion"
    }
    """,
    name="_"
)


@app.cell
def _(TERMS_SIMPLE):
    TERMS_SIMPLE.aliases
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple terms results
    """)
    return


@app.cell
def _(llm):
    llm.stage_convert(overwrite=True, num_workers = 4)
    return


@app.cell
def _(llm):
    llm.stage_chunk(overwrite=True)
    return


@app.cell
def _(TERMS_SIMPLE, llm):
    llm.stage_search(TERMS_SIMPLE, overwrite=True)
    return


@app.cell
def _(TERMS_SIMPLE, regex):
    df_attention = regex.analysis.df_metrics(TERMS_SIMPLE)
    df_attention
    return (df_attention,)


@app.cell
def _(TERMS_SIMPLE, companies, df_attention, mo, np, plt):
    col = 'rate'

    fig_an, axs_an = plt.subplots(6,3, sharex=True, sharey=True, figsize=(8,15), constrained_layout=True)
    fig_sus, axs_sus = plt.subplots(6,3, sharex=True, sharey=True, figsize=(8,15), constrained_layout=True)

    decarb_terms = df_attention.loc[df_attention['group'] == 'decarbonisation']['term_alias'].unique().tolist()
    fossil_terms = df_attention.loc[df_attention['group'] == 'fossil']['term_alias'].unique().tolist()
    axs_map = {term: (i%6, i//6) for i, term in enumerate(decarb_terms)}
    axs_map.update({term: (i%6, i//6 + 2) for i, term in enumerate(fossil_terms)})

    for term in df_attention['term_alias'].unique():
        _ax_an = axs_an[*axs_map[term]]
        _ax_sus = axs_sus[*axs_map[term]]
        _ax_an.set_title(term)
        _ax_sus.set_title(term)
        term_obj = TERMS_SIMPLE[term]
        for company in df_attention['company_alias'].unique():
            ind_co_term = (df_attention['company_alias'] == company) & (df_attention['term_alias'] == term)
            company_obj = companies.COMPANIES_ALL[company]
            data_an = df_attention.loc[ind_co_term & (df_attention['report_type'] == 'annual-report')].set_index('year').sort_index()[col].values
            data_sus = df_attention.loc[ind_co_term & (df_attention['report_type'] == 'sustainability-report')].set_index('year').sort_index()[col].values
            yrs_an = df_attention.loc[ind_co_term & (df_attention['report_type'] == 'annual-report')].set_index('year').sort_index().index.values
            yrs_sus = df_attention.loc[ind_co_term & (df_attention['report_type'] == 'sustainability-report')].set_index('year').sort_index().index.values
            _ax_an.plot(yrs_an, data_an, color = company_obj.colour, label = term)
            _ax_sus.plot(yrs_sus, data_sus, color = company_obj.colour, label = term)

        for ax in (_ax_an, _ax_sus):
            ax.set_xlabel('Report year')
            ax.set_ylabel('Share of mentions')
            ax.set_ylim(0,1)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.grid(axis='y', alpha=0.5)


    mo.ui.tabs(
        {'Annual Report': fig_an, 
         'Sustainability Report': fig_sus}
        )
    return


@app.cell
def _(TERMS_SIMPLE, df_attention, plt):
    _df = df_attention.loc[(df_attention['company_alias'] == 'BP') & (df_attention['report_type'] == 'annual-report')].pivot(index='year', columns = 'term_alias', values = 'share')#.values
    _vals = _df.T.values
    _x = _df.index.astype(int).values
    plt.stackplot(
        _x,
        _vals,
        colors = TERMS_SIMPLE.colours
    )
    return


@app.cell
def _(TERMS_SIMPLE, df_attention, mo, plt):
    _fig_an, _axs_an = plt.subplots(3,2, sharex=True, sharey=True, figsize=(8,15), constrained_layout=True)
    _fig_sus, _axs_sus = plt.subplots(3,2, sharex=True, sharey=True, figsize=(8,15), constrained_layout=True)

    for i, _co in enumerate(df_attention['company_alias'].unique()):
        _df_plot_an = df_attention.loc[(df_attention['company_alias'] == _co) & (df_attention['report_type'] == 'annual-report')].pivot(index='year', columns = 'term_alias', values = 'share')
        _df_plot_sus = df_attention.loc[(df_attention['company_alias'] == _co) & (df_attention['report_type'] == 'sustainability-report')].pivot(index='year', columns = 'term_alias', values = 'share')

        _vals_an = _df_plot_an.T.values
        _vals_sus = _df_plot_sus.T.values
        _x_an = _df_plot_an.index.astype(int).values
        _x_sus = _df_plot_sus.index.astype(int).values
        _axs_an[i%3, i//3].stackplot(
            _x_an,
            _vals_an,
            colors = TERMS_SIMPLE.colours
        )
        _axs_sus[i%3, i//3].stackplot(
            _x_sus,
            _vals_sus,
            colors = TERMS_SIMPLE.colours
        )
        _axs_an[i%3, i//3].set_title(_co)
        _axs_sus[i%3, i//3].set_title(_co)

    mo.ui.tabs(
        {'Annual Report': _fig_an, 
         'Sustainability Report': _fig_sus}
        )
    return


@app.cell
def _(df_attention):
    _df = df_attention.loc[(df_attention['company_alias'] == 'Shell') & (df_attention['report_type'] == 'annual-report') & (df_attention['group'] == 'decarbonisation')].pivot_table(index='year', columns = 'group', values = 'share', aggfunc='sum')
    _vals = _df.T.values[0]
    _x = _df.index.astype(int).values

    _vals, _x
    return


@app.cell
def _(df_attention, mo, np, plt):
    _fig_an, _axs_an = plt.subplots(3,2, sharex=True, sharey=True, figsize=(8,5), constrained_layout=True)
    _fig_sus, _axs_sus = plt.subplots(3,2, sharex=True, sharey=True, figsize=(8,5), constrained_layout=True)

    for _i, _co in enumerate(df_attention['company_alias'].unique()):
        _ax_an = _axs_an[_i%3, _i//3]
        _ax_sus = _axs_sus[_i%3, _i//3]
        _df_plot_an = df_attention.loc[(df_attention['company_alias'] == _co) & (df_attention['report_type'] == 'annual-report') & (df_attention['group'] == 'decarbonisation')].pivot_table(index='year', columns = 'group', values = 'share', aggfunc='sum')
        _df_plot_sus = df_attention.loc[(df_attention['company_alias'] == _co) & (df_attention['report_type'] == 'sustainability-report') & (df_attention['group'] == 'decarbonisation')].pivot_table(index='year', columns = 'group', values = 'share', aggfunc='sum')

        _vals_an = _df_plot_an.T.values[0] * 100
        _vals_sus = _df_plot_sus.T.values[0] * 100
        _x_an = _df_plot_an.index.astype(int).values
        _x_sus = _df_plot_sus.index.astype(int).values
        # _axs_an[_i%3, _i//3].stackplot(
        #     _x_an,
        #     _vals_an,
        #     colors = ['tab:green', 'tab:gray']
        # )
        # _axs_sus[_i%3, _i//3].stackplot(
        #     _x_sus,
        #     _vals_sus,
        #     colors = ['tab:green', 'tab:gray']
        # )

        _ax_an.bar(_x_an, _vals_an, color = 'tab:green')
        _ax_sus.bar(_x_sus, _vals_sus, color = 'tab:green')
        _ax_an.set_title(_co)
        _ax_sus.set_title(_co)
        _ax_an.set_yticks(np.arange(0, 110, 20))
        _ax_sus.set_yticks(np.arange(0, 110, 20))
        _ax_an.grid(axis='y', alpha=0.5)
        _ax_sus.grid(axis='y', alpha=0.5)
        _ax_an.set_ylabel(r'% of Sections')
        _ax_sus.set_ylabel(r'% of Sections')
        _ax_an.set_xlabel('Report year')
        _ax_sus.set_xlabel('Report year')
        _ax_an.label_outer()
        _ax_sus.label_outer()

    _fig_an.suptitle('Annual Report: Sections Mentioning Decarbonisation Technologies')
    _fig_sus.suptitle('Sustainability Report: Sections Mentioning Decarbonisation Technologies')

    mo.ui.tabs(
        {'Annual Report': _fig_an, 
         'Sustainability Report': _fig_sus}
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### New job
    """)
    return


@app.cell
def _(llm):
    job_rerun = llm.prompt.Job('81c733b7aefc5b949330e814c11bf8e6')
    job_ctrl = llm.prompt.Job('2e4ea8b7cb055fc1a6c7d87aed5f1321')
    [job_rerun, job_ctrl]
    return job_ctrl, job_rerun


@app.cell
def _(job_rerun):
    job_rerun.refresh(adopt_orphans = False, progress_bar = True)
    job_rerun
    return


@app.cell
def _(job_rerun):
    df_rerun = job_rerun.to_df()
    df_rerun
    return


@app.cell
def _(job_rerun):
    job_rerun.results_comprehensive
    return


@app.cell
def _(job_ctrl):
    job_ctrl.to_df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual labelling
    """)
    return


@app.cell
def _():
    from decarb.regex import LabellingSession

    return (LabellingSession,)


@app.cell
def _(LabellingSession, TERMS_SIMPLE):
    session = LabellingSession(TERMS_SIMPLE, 'out/regex-labelling-sample.csv', 'out/regex-labels.jsonl')
    session.apply_labels()
    return (session,)


@app.cell
def _(mo):
    get_session_pos, set_session_pos = mo.state(0)
    get_session_err, set_session_err = mo.state(None)
    return get_session_err, get_session_pos, set_session_err, set_session_pos


@app.cell
def _(session, set_session_err, set_session_pos):
    # shortcuts off while the original labelling cells above remain: shortcut listeners are
    # document-wide and the first-mounted button wins, so a keypress would label the chunk shown in
    # the original card rather than this one
    session_buttons = session.buttons(set_session_pos, set_session_err, shortcuts=False)
    return (session_buttons,)


@app.cell
def _():
    type({a for a in ['a','b']})
    return


@app.cell
def _(get_session_err, get_session_pos, session, session_buttons):
    session.view(get_session_pos(), get_session_err(), session_buttons)
    return


@app.cell
def _(session):
    session.scores()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial output analysis
    """)
    return


@app.cell
def _(llm):
    job = llm.prompt.Job('14ea7e40c9c757c28629d5cb4c906445')
    job
    return (job,)


@app.cell
def _(job, llm):
    df_test = llm.results.to_df(job=job)
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
def _(TERMS_ALL, df_test, output, util):
    for _q in ['q1_sentiment', 'q2_maturity', 'q3_stage']:
        _fig = output.plot.llm_terms_heatmap(df_test, _q, figsize=(9,15), return_type='figure')
        _fig.savefig(f'{util.OUT_DIR}/{TERMS_ALL.folder_name}/heatmap-{_q}-20260805.png', dpi=300)
    return


if __name__ == "__main__":
    app.run()
