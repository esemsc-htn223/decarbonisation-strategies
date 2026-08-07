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
    from decarb import companies, llm, util


    return (
        Literal,
        TERMS_ALL,
        Term,
        TermCollection,
        alt,
        companies,
        json,
        llm,
        mo,
        mpl,
        np,
        pd,
        plt,
        re,
        util,
    )


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
    return df_test, out_paths, results_df


@app.cell
def _(np, pd):
    COLOUR_STG = '#8AC4FF'
    COLOUR_MAT = '#E5B8FF'
    COLOUR_SENT = '#FF8A8A'
    COLOR_CMT = "#E8FF8A"

    stg_map = {
        "research and development": {
            'alias': "1/4 - R&D",
            'style': {'color': '#8AC4FF'}
        },
        "pilot": {
            'alias': "2/4 - Pilot",
            'style': {'color': "#1F8FFF"}
        },
        "demonstration": {
            'alias': "3/4 - Demonstration",
            'style': {'color': '#0080FF'}
        },
        "commercial-deployment": {
            'alias': "4/4 - Deployment",
            'style': {'color': '#0052A3'}
        },
        "unclear": {
            'alias': "-. - Unclear",
            'style': {'color': '#cccccc'}
        },
        "phasing-out": {
            'alias': "0/4 - Phasing Out",
            'style': {'color': '#e65054'}
        },
        pd.NA: {
            'alias': "-. - Unclear",
            'style': {'color': '#cccccc'}
        },
        np.nan: {
            'alias': "-. - Unclear",
            'style': {'color': '#cccccc'}
        },
        None: {
            'alias': "-. - Unclear",
            'style': {'color': '#cccccc'}
        }
    }


    mat_map = {
        1: {
            'alias': "1/9 - Initial idea reported",
            'style': {'color': '#F8CDF9'}
        },
        2: {
            'alias': "2/9 - Concept development",
            'style': {'color': '#F5BCF7'}
        },
        3: {
            'alias': "3/9 - Early-stage testing",
            'style': {'color': '#F1ABF5'}
        },
        4: {
            'alias': "4/9 - Pilot implementation",
            'style': {'color': '#ED9AF3'}
        },
        5: {
            'alias': "5/9 - Scale-up planning",
            'style': {'color': '#E988F0'}
        },
        6: {
            'alias': "6/9 - Commercial deployment",
            'style': {'color': '#E475EE'}
        },
        7: {
            'alias': "7/9 - Market penetration",
            'style': {'color': '#DF61EB'}
        },
        8: {
            'alias': "8/9 - Optimization",
            'style': {'color': '#D94AE9'}
        },
        9: {
            'alias': "9/9 - Mature technology",
            'style': {'color': '#D32CE6'}
        },
        None:{
            'alias': "-. - Unclear",
            'style': {'color': '#cccccc'}
        }
    }
    return COLOR_CMT, COLOUR_MAT, COLOUR_SENT, COLOUR_STG, mat_map, stg_map


@app.cell
def _(
    COLOR_CMT,
    COLOUR_MAT,
    COLOUR_SENT,
    COLOUR_STG,
    Term,
    TermCollection,
    companies,
    llm,
    mat_map,
    mo,
    mpl,
    pd,
    re,
    stg_map,
):
    def mark_term_mentions(md: str, term: Term):
        matches = re.finditer(term.regex, md, re.I)
        prev_shift = 0
        for match in matches:
            start, end = match.span()
            md = md[:start + prev_shift] + f'<u><mark>{md[start + prev_shift:end + prev_shift]}</mark></u>' + md[end + prev_shift:]
            prev_shift += 20  # len('<u><mark></mark></u>')
        return md

    def mark_quote(md: str, quotes: list, colour: str = '#ffcccb', add_label:str = '') -> str:

        for quote in quotes:
            if isinstance(quote,dict):
                quote = quote['quote']
            if '...' in quote:
                for q in quote.split('...'):
                    if q.strip():
                        md = md.replace(q, f'<mark style="background-color: {colour}">{q}{add_label}</mark>')
            md = md.replace(quote, f'<mark style="background-color: {colour}">{quote}{add_label}</mark>')
        return md

    def count_quote_mentions(md: str, quotes: list) -> int:
        if not quotes:
            return 0
        count = 0
        for quote in quotes:
            if isinstance(quote,dict):
                quote = quote['quote']
            if '...' in quote:
                n_total = len(quote.split('...'))
                for q in quote.split('...'):
                    if q.strip() and q in md:
                        count += 1
                count = count/n_total  # return as a fraction of how many parts of the quote were present
            if quote in md:
                count += 1  # quotes wont appear more than once
        return count

    def quote_stack(quotes: list) -> mo.Html:
        quotes_stk = []
        for q in quotes:
            quotes_stk.append(
                mo.vstack([
                    mo.md(
                        f"> *\"{q['quote']}\"*"
                    ),
                    mo.ui.text_area(
                        q['justification'],
                        full_width = True,
                        rows = 3,
                    ),
                ])
                )
        return mo.vstack(quotes_stk)

    def commitment_accordion(commitments: list) -> mo.Html:
        quotes_acc = {}
        for c in commitments:
            quotes_acc[f"*{c['commitment']}*"] = mo.vstack([
                    mo.md(
                        f"> *\"{c['quote']}\"*"
                    ),
                    mo.ui.text_area(
                        c['justification'],
                        full_width = True,
                        rows = 3,
                    ),
                    mo.ui.button(
                        label = 'Verifiable' if c['verifiable'] else 'Not Verifiable',
                        kind = 'success' if c['verifiable'] else 'danger',
                        disabled = True, full_width = True
                    )
                ])
        return mo.accordion(quotes_acc)

    def _entry_styles(entry: pd.Series) -> tuple:
        '''Look up the maturity/stage display details and inline styles for an entry.'''
        mat_details = mat_map.get(entry['q2_maturity'], {'alias': str(entry['q2_maturity']), 'style': {}})
        stg = entry['q3_stage']
        stg_details = stg_map.get(stg, {'alias': stg.title() if isinstance(stg, str) else str(stg), 'style': {}})
        style = lambda details: '; '.join(f"{k}: {v}" for k, v in details["style"].items())
        return mat_details, style(mat_details), stg_details, style(stg_details)

    def term_year_key(entry: pd.Series) -> str:
        '''
        Build the accordion label for a single term-year entry.

        Depends only on `entry`, so it can be computed eagerly for every year while the
        matching body (which reads chunk files) stays behind a lazy loader.
        '''
        yr = entry['year']
        if not entry['report_type_published']:
            return f'<span style="color: lightcoral">**{yr}:** Report type "{entry["report_type"]}" not published</span>'

        mat_details, mat_style, stg_details, stg_style = _entry_styles(entry)
        snt = entry['q1_sentiment']

        # handle missing / non-numeric sentiment
        if pd.isna(snt):
            snt_display = "Unclear"
            snt_colour = "#cccccc"
        else:
            snt_display = snt
            try:
                snt_cmap = mpl.colormaps['RdYlGn'].resampled(21)
                snt_colour = mpl.colors.to_hex(snt_cmap((float(snt) + 1) / 2))
            except (TypeError, ValueError):
                snt_colour = "#cccccc"

        key = f'**{yr}:** {snt_display}<span style="color: {snt_colour}">█</span> \t|\t <span style="{stg_style}">{stg_details["alias"]}</span> \t|\t <span style="{mat_style}">{mat_details["alias"]}</span>'
        n_commitments = len(entry['q4_commitments'])
        if n_commitments > 0:
            key += f' \t|\t {n_commitments} commitment{"s" if n_commitments != 1 else ""}'
        return key

    def term_year_analysis(entry: pd.Series, terms: TermCollection, *, return_key: bool = False, lazy_chunks: bool = True) -> mo.Html:

        if not entry['report_type_published']:
            out = mo.md(f"> Report type '{entry['report_type']}' not published in {entry['year']}")
            if return_key:
                out = (term_year_key(entry), out)
            return out

        co = entry['company_alias']
        rt = entry['report_type']
        yr = entry['year']
        term = terms[entry['term_alias']]

        mat_details, mat_style, stg_details, stg_style = _entry_styles(entry)
        snt = entry['q1_sentiment']

        if not entry['report_type_published']:
            chunks = [f"Report type '{rt}' not published in {yr}"]
        else:
            # This is the only file I/O in the pipeline, and it runs lazily (only when a
            # year is expanded). Guard it so a missing search-result or chunks directory
            # renders a message rather than leaving the lazy loader spinning forever.
            try:
                chunks = llm.inspection.search_results_chunks(companies.Report.from_metadata(co, rt, yr), terms, entry['term_alias'])
            except (FileNotFoundError, KeyError, ValueError) as e:
                out = mo.md(f'> <span style="color: lightcoral">Chunks unavailable for {co} {rt} {yr} / {entry["term_alias"]}: {e}</span>')
                if return_key:
                    out = (term_year_key(entry), out)
                return out

        if pd.notna(entry['notes']):
            notes = f"> {entry['notes']}"
        else:
            notes = '> no notes provided'

        # mark all quotes
        chunks = [mark_quote(chunk, entry['q1_sentiment_quotes'], add_label=' (q1)', colour = COLOUR_SENT) for chunk in chunks]
        chunks = [mark_quote(chunk, entry['q2_maturity_quotes'], add_label=' (q2)', colour = COLOUR_MAT) for chunk in chunks]
        chunks = [mark_quote(chunk, entry['q3_stage_quotes'], add_label=' (q3)', colour = COLOUR_STG) for chunk in chunks]
        chunks = [mark_quote(chunk, entry['q4_commitments'], add_label=' (q4)', colour = COLOR_CMT) for chunk in chunks]
        # and the mentions of the term
        chunks = [mo.md(mark_term_mentions(chunk, term)) for chunk in chunks]
        n_commitments = len(entry['q4_commitments'])

        out =  mo.vstack([
                    mo.vstack([
                        mo.md(notes),
                        mo.md('---'),
                        mo.hstack([
                            mo.md(f'**Sentiment:** {snt}'),
                            mo.md('|'),
                            mo.md(f'**Stage:** <span style="{stg_style}">{stg_details["alias"]}</span>'),
                            mo.md('|'),
                            mo.md(f'**Maturity:** <span style="{mat_style}">{mat_details["alias"]}</span>'),
                        ], justify = 'center')
                    ]),
                    mo.accordion({
                        f'**Q1. Sentiment** {snt}': quote_stack(entry['q1_sentiment_quotes']),
                        f'**Q2. Maturity** <span style="{mat_style}">{mat_details["alias"]}</span>': quote_stack(entry['q2_maturity_quotes']),
                        f'**Q3. Stage** <span style="{stg_style}">{stg_details["alias"]}</span>': quote_stack(entry['q3_stage_quotes']),
                        f'**Q4. Commitments** ({n_commitments})': commitment_accordion(entry['q4_commitments']),
                    }, multiple = True),
                    mo.md('---'),

                    mo.accordion({
                        f'{len(chunks)} Chunk{"s" if len(chunks) != 1 else ""}:': mo.vstack(chunks)
                    }, lazy = lazy_chunks),
                ])
        if return_key:
            out = (term_year_key(entry), out)
        return out

    return term_year_analysis, term_year_key


@app.cell
def _(Literal, Term, alt, pd):
    def plot_timeseries(df:pd.DataFrame, variable: Literal['q1_sentiment', 'q2_maturity'], 
                        *, 
                        company:str, term:Term, report_type:str, 
                        width_px:int = 1000, height_px:int = 250) -> alt.Chart:
        '''
        Create an Altair chart showing sentiment over time for a given term and report type.
        Missing data is indicated by shaded regions, with black for years where the report was not published, and gray for years where the term was not mentioned.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the sentiment analysis results.
        company : str
            The company for which to plot sentiment over time.
        term : Term
            The term for which to plot sentiment over time.
        report_type : str
            The report type for which to plot sentiment over time.
        width_px : int, optional
            Width of the chart in pixels (default is 1000).
        height_px : int, optional
            Height of the chart in pixels (default is 250).

        Returns
        -------
        alt.Chart
        '''

        if variable not in ['q1_sentiment', 'q2_maturity']:
            raise ValueError(f"Variable '{variable}' should be either 'q1_sentiment' or 'q2_maturity'.")
        elif variable == 'q1_sentiment':
            domain = [-1, 1]
        elif variable == 'q2_maturity':
            domain = [1, 9]

        df_co_rt = df[(df["company"] == company) & (df["report_type"] == report_type)].copy()
        all_years = sorted(df['year'].unique())
        chart = alt.Chart(df_co_rt[(df_co_rt['term_alias'] == term.alias)])
        lines = chart.mark_point(color = term.colour).encode(
                    x = alt.X('year:O')
                        .scale(domain = all_years)
                        .axis(title = None, labelAngle = -45, offset = -height_px//2, 
                              labelPadding = height_px//2 + 5, # +5 to account for missing ticks (default height of 5px)
                              tickSize = 0
                              )
                    ,
                    y = alt.Y(f'{variable}:Q', title = variable.split('_')[1].capitalize())
                        .scale(domain = domain)
                    ,

                    tooltip = [
                        alt.Tooltip('year:O', title = 'Report Year'),
                        alt.Tooltip('q1_sentiment:Q', title = 'Sentiment'),
                        alt.Tooltip('q2_maturity:N', title = 'Maturity'),
                        alt.Tooltip('q3_stage:N', title = 'Stage'),
                    ]
                ).properties(
                    #title = f'Sentiment over Time for {term.alias}',
                    width = width_px,
                    height = height_px
            )
        # mark missing data with shaded region
        # get years where report wasn't published
        has_pub_flag = "report_type_published" in df_co_rt.columns
        has_term_flag = "term_regex_found" in df_co_rt.columns

        no_reports_years: list[str] = []
        if has_pub_flag:
            no_reports_years = (
                df_co_rt.loc[df_co_rt["report_type_published"] == False, "year"]  # noqa: E712
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
        else:
            no_reports_years = sorted(set(all_years) - set(df_co_rt['year'].unique()))


        missing_term_years: list[str] = []
        if has_term_flag:
            # Missing term = report was published but term regex not found.
            missing_term_years = (
                df_co_rt[
                    (df_co_rt["term_alias"] == term.alias)
                    & (df_co_rt["report_type_published"] == True)   # noqa: E712
                    & (df_co_rt["term_regex_found"] == False)        # noqa: E712
                ]["year"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
        else:
            missing_term_years = sorted(set(all_years) - set(df_co_rt[(df_co_rt['term_alias'] == term.alias) & (df_co_rt['report_type_published'] == True)]['year'].unique()) - set(no_reports_years))

        if no_reports_years:
            no_reports_chart = (
                alt.Chart(pd.DataFrame({"year": no_reports_years}))
                .mark_rect(opacity=0.2, color="black")
                .encode(
                    x=alt.X("year:O", title="Report Year"),
                    y=alt.value(0),
                    y2=alt.value(height_px),
                    tooltip=alt.value("Report not published"),
                )
            )
            lines = no_reports_chart + lines

        if missing_term_years:
            missing_chart = (
                alt.Chart(pd.DataFrame({"year": missing_term_years}))
                .mark_rect(opacity=0.2, color="gray")
                .encode(
                    x=alt.X("year:O", title="Report Year"),
                    y=alt.value(0),
                    y2=alt.value(height_px),
                    tooltip=alt.value("Term not mentioned"),
                )
            )
            lines = missing_chart + lines

        return lines

    return (plot_timeseries,)


@app.cell
def _(df_test):
    df_test
    return


@app.cell
def _(df_test):
    df_test[(df_test['company'] == 'exxon') & (df_test['term_alias'] == 'Nuclear fusion') & (df_test['report_type'] == 'annual-report')]['prompt'].isna().all()
    return


@app.cell
def _(
    TERMS_ALL,
    TermCollection,
    companies,
    mo,
    pd,
    plot_timeseries,
    term_year_analysis,
    term_year_key,
    util,
):
    def company_report_results(df: pd.DataFrame, company_fname: str, terms: TermCollection, report_type: str, lazy: bool = True) -> mo.Html:
        '''
        Get the results for a given company and report type, with tabs for each term containing the sentiment timeseries, regex, and year-by-year LLM analysis.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the results of the analysis.
        company_fname : str
            The filename of the company to get the results for. Should be one of the recognised companies, as in the keys of `companies.COMPANY_ALIASES`.
        terms : TermCollection
            The collection of terms to analyze.
        report_type : str
            The report type to analyze. Should be one of the recognised report types, as in the keys of `util.REPORT_TYPES_LONG`.

        Returns
        -------
        mo.Html
            A mo.ui.tabs object containing the results for the given company and report type, with tabs for each term.
        '''

        if company_fname not in df['company'].unique():
            raise ValueError(f'Company "{company_fname}" not found in dataframe. Available companies: {df["company"].unique()}')
        if report_type not in df['report_type'].unique():
            raise ValueError(f'Report type "{report_type}" not found in dataframe. Available report types: {df["report_type"].unique()}')

        company_alias = companies.COMPANY_ALIASES.get(company_fname, company_fname)
        term_tabs = {}
        rt_long = util.REPORT_TYPES_LONG.get(report_type, report_type).title()
        for term in terms:
            df_term = df[(df['report_type'] == report_type) & (df['term_alias'] == term.alias) & (df['company'] == company_fname)]

            if df_term['prompt'].isna().all():
                term_tabs[f'<span style="color: lightcoral">{term}</span>'] = mo.vstack([
                    mo.ui.text_area(value = term.regex, full_width = True, label = '**Regex:**'),
                    mo.md(f'> <span style="color: lightcoral"> Term "{term}" not found in {company_alias}\'s {rt_long.lower()}s.</span>'),
                ])
                continue

            # A plain callable: mo.ui.tabs(lazy=True) wraps each value in mo.lazy exactly
            # once. Wrapping it here as well gives mo.lazy(mo.lazy(...)), which never loads.
            def make_term_panel(term=term, df_term=df_term):
                # Keys are built eagerly (cheap, no file I/O); the bodies are plain
                # callables so mo.accordion(lazy=True) defers the chunk reads until a
                # year is expanded.
                year_accordion = {
                    term_year_key(entry): (lambda entry=entry: term_year_analysis(entry, terms, return_key=False))
                    for entry in (df_term[df_term['year'] == year].iloc[0] for year in df_term['year'].unique()) if entry['term_regex_found']
                }

                return mo.vstack([
                    mo.ui.text_area(value = term.regex, full_width = True, label = '**Regex:**'),
                    mo.hstack([
                        plot_timeseries(df_term, 'q1_sentiment', company=company_fname, term=term, report_type=report_type, width_px=550, height_px=250),
                        plot_timeseries(df_term, 'q2_maturity', company=company_fname, term=term, report_type=report_type, width_px=550, height_px=250)
                    ], gap = 0, widths = 'equal', align = 'center'),
                    mo.accordion(year_accordion, lazy = lazy)
                ])

            term_tabs[term.alias] = make_term_panel

        return mo.ui.tabs(term_tabs, label = 'Term', lazy = lazy)


    def results_tabs(df: pd.DataFrame, terms: TermCollection = TERMS_ALL, lazy: bool = True) -> mo.Html:
        '''
        Company -> report type -> term -> year drill-down over the analysis results.

        Every level is lazy, so a cell only ever renders the branch that is currently
        open. Building the whole tree eagerly exceeds marimo's `output_max_bytes`.
        '''
        def make_company_tabs(co, terms=terms, lazy=lazy):
            # Each value is a plain callable, wrapped in mo.lazy once by lazy=True.
            rt_tabs = {
                util.REPORT_TYPES_LONG.get(report_type, report_type).title():
                    (lambda co=co, rt=report_type: company_report_results(
                        df, company_fname=co, terms=terms, report_type=rt, lazy=lazy
                    ))
                for report_type in df['report_type'].unique()
            }
            return mo.ui.tabs(rt_tabs, label = 'Report Type', lazy = lazy)

        co_tabs = {
            companies.COMPANY_ALIASES.get(co, co): (lambda co=co: make_company_tabs(co))
            for co in df['company'].unique()
        }

        return mo.ui.tabs(co_tabs, label = 'Company', lazy = lazy)

    return (results_tabs,)


@app.cell
def _(TERMS_ALL, df_test, results_tabs):
    results_tabs(df_test, TERMS_ALL)
    return


@app.cell
def _(df_test):
    df_test
    return


@app.cell
def _(TERMS_ALL, df_test, results_tabs):
    results_tabs(df_test.loc[df_test['company'].isin(['shell']) & df_test['term_alias'].isin(['Solar', 'Wind']) & (df_test['report_type'] == 'annual-report') & (df_test['year'].astype(int) > 2019)], TERMS_ALL)
    return


@app.cell
def _(TERMS_ALL, out_paths, results_df, results_tabs):
    results_tabs(
        results_df(
            in_jsonl_paths=[
                'data/terms/all-terms/prompts/a2d4064f54b85a0691382e53cc935b1e.jsonl', 
                'data/terms/all-terms/prompts/76a1c482f11652c5ba9e3f8c4b4771ae.jsonl'
            ],
            out_jsonl_paths=out_paths,
            fillna_values=None,
            label_missing=True
    ), TERMS_ALL)
    return


@app.cell
def _(TERMS_ALL, TermCollection, companies, df_test, np, pd):
    def __mark_missing_years(df: pd.DataFrame, terms:TermCollection, company: str, report_type: str) -> pd.DataFrame:
        df_copy = df.copy()
        df_co_rt = df[(df['company'] == company) & (df['report_type'] == report_type)]
        if df_co_rt.empty:
            raise ValueError(f'No data found for company "{company}" and report type "{report_type}".')
        for yr in np.arange(df_co_rt['year'].astype(int).min(), df_co_rt['year'].astype(int).max() + 1, 1):
            if yr not in df_co_rt['year'].unique():
                for term in terms:
                    index = f"{company}_{report_type}_{yr}_{term.prompt_fname.replace('.txt', '')}"
                    df_copy = pd.concat([df_copy, pd.DataFrame([{
                        'company': company,
                        'report_type': report_type,
                        'term_alias': term.alias,
                        'year': str(yr),
                        'q1_sentiment': np.nan,
                        'q2_maturity': np.nan,
                        'q3_stage': np.nan,
                        'q4_commitments': [],
                        'q1_sentiment_quotes': [],
                        'q2_maturity_quotes': [],
                        'q3_stage_quotes': [],
                        'company_alias': companies.COMPANY_ALIASES.get(company, None),
                        'term_regex': term.regex,
                        'term_fname': term.prompt_fname,
                        'prompt': pd.NA,
                        'term_regex_found': False,
                        'report_type_published': False,
                    }], index=[index])])
        return df_copy

    def mark_all_missing_years(df: pd.DataFrame, terms:TermCollection) -> pd.DataFrame:
        df_copy = df.copy()
        for company in df['company'].unique():
            for report_type in df['report_type'].unique():
                df_copy = __mark_missing_years(df_copy, terms, company, report_type)
        return df_copy

    mark_all_missing_years(df_test, TERMS_ALL)
    return


@app.cell
def _(TERMS_ALL, Term, df_test, mo, np, pd, plt, util):
    from typing import Tuple
    from decarb.companies import COS, CompanyCollection
    _term = TERMS_ALL['solar']

    def dodge_points(points, component_index, offset):
        """Dodge every point by a multiplicative offset (multiplier is based on frequency of appearance)

        Args:
            points (array-like (2D)): Array containing the points
            component_index (int): Index / column on which the offset will be applied 
            offset (float): Offset amount. Effective offset for each point is `index of appearance` * offset

        Returns:
            array-like (2D): Dodged points
        """

        # Extract uniques points so we can map an offset for each
        uniques, inv, counts = np.unique(
            points, return_inverse=True, return_counts=True, axis=0
        )

        for i, num_identical in enumerate(counts):
            # Prepare dodge values
            dodge_values = np.array([offset * i for i in range(num_identical)])
            # Find where the dodge values must be applied, in order
            points_loc = np.where(inv == i)[0]
            #Apply the dodge values
            points[points_loc, component_index] += dodge_values

        return points

    def plot_term_co_comparison(df: pd.DataFrame, term: Term, *, cos: CompanyCollection = COS) -> Tuple[plt.Figure, plt.Axes]:
        '''
        Plot the sentiment and maturity over time for a given term, comparing across companies and report types.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the results of the analysis.
        term : Term
            The term to plot.
        cos : CompanyCollection, optional
            The collection of companies to include in the plot. Defaults to `COS`.

        Returns
        -------
        plt.Figure, plt.Axes
            The matplotlib Figure and Axes objects containing the plot.
        '''

        yrs = np.arange(df['year'].astype(int).min(), df['year'].astype(int).max() + 1, 5)

        fig, axs = plt.subplots(2, 2, figsize=(12,8), sharey=False, sharex=True)

        for i, rt in enumerate(df['report_type'].unique()):
            for co in df['company'].unique():
                df_co = df[(df['report_type'] == rt) & (df['company'] == co) & (df['term_fname'] == term.prompt_fname)]
                df_co['year'] = df_co['year'].astype(int)
                df_co.sort_values('year', inplace=True)
                co = cos[co]
                axs[0, i].plot(
                    df_co['year'], df_co['q1_sentiment'], 
                    label=co.full_name, color = co.colour,
                    #linestyle = _term.dash_ls if _term.dash else '-',
                    #gapcolor = _term.edgecolour,
                    linestyle = 'None',
                    marker = 'o', alpha = 1
                )
                axs[1, i].plot(
                    df_co['year'], df_co['q2_maturity'], 
                    label=co.full_name, color = co.colour,
                    linestyle = 'None',
                    #gapcolor = _term.edgecolour,
                    #linestyle = 'None',
                    marker = 'o', alpha = 1
                )

            axs[0, i].set_title(util.REPORT_TYPES_LONG.get(rt, rt).title() + 's')
            if i == 0:
                axs[0, i].set_ylabel('Sentiment')
                axs[1, i].set_ylabel('Maturity')
            elif i == 1:
                axs[0, i].set_yticklabels([])
                axs[1, i].set_yticklabels([])

            axs[0, i].grid(axis = 'y')
            axs[1, i].grid(axis = 'y')
            axs[0, i].set_ylim(-1.1, 1.1)
            axs[1, i].set_ylim(0.8, 9.2)
            axs[0, i].tick_params(axis='x', rotation=45)
            axs[1, i].tick_params(axis='x', rotation=45)
            axs[0, i].set_xticks(yrs)
            axs[1, i].set_xticks(yrs)

        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncol=17, bbox_to_anchor=(0.5, 0.0))
        #fig.suptitle(f'{term.alias} Sentiment over Time')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig, axs

    _acc = {}
    for _term in TERMS_ALL:
        _fig, _axs = plot_term_co_comparison(df_test, _term)
        _acc[_term.alias] = _fig

    mo.ui.tabs(_acc, label = 'Term', lazy = False)
    return


@app.cell
def _(TERMS_ALL, Term, companies, df_test, discrete_cmap, np, pd, plt, util):
    def plot_sentiment_maturity_heatmap(df: pd.DataFrame, term: Term, figsize: tuple = (12, 3)) -> plt.Figure:
        '''
        Plot a heatmap of sentiment and maturity for a given term across companies and report types.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the results of the analysis.
        term : Term
            The term to plot.
        figsize : tuple, optional
            The size of the figure in inches. Default is (12, 3).

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the heatmap.
        '''
        if not isinstance(term, Term):
            try:
                term = TERMS_ALL[term]
            except KeyError:
                raise ValueError(f'Term "{term}" not found in TERMS_ALL. Available terms: {list(TERMS_ALL.keys())}')

        fig, axs = plt.subplots(df['report_type'].unique().shape[0], 2, figsize=figsize, sharey=True, sharex=True)

        cmaps = {
            'q1_sentiment': 'RdYlGn',
            'q2_maturity': discrete_cmap([1, 2, 3, 4, 5, 6, 7, 8, 9], cmap_name='cividis'),
            'q3_stage': discrete_cmap([1, 2, 3, 4, 5], cmap_name='plasma'),
        }

        for i, rt in enumerate(df['report_type'].unique()):
            for j, question in enumerate(('q1_sentiment', 'q2_maturity')):
                df_question = df.loc[(df['report_type'] == rt) & (df['term_alias'] == term.alias)].pivot(index='company', columns='year', values=question).astype(float)

                im = axs[i, j].imshow(
                    df_question.replace(-1, np.nan).values,
                    cmap=cmaps[question]
                )
                axs[i, j].set_title(f"{util.REPORT_TYPES_LONG.get(rt, rt).title()} - {question.split('_')[1].capitalize()}")
                axs[i, j].set_yticks(np.arange(len(df_question.index)), df_question.reset_index()['company'].apply(lambda x: companies.COMPANY_ALIASES.get(x, x)))
                axs[i, j].set_xticks(np.arange(len(df_question.columns)),minor = True)
                axs[i, j].set_xticks(
                    np.arange(len(df_question.columns), step = 5),
                    np.arange(df_question.columns.astype('int').min(), df_question.columns.astype('int').max() + 1, 5), 
                    minor = False, 
                    rotation = 90, #ha = 'right'
                )

        fig.suptitle(f"{term.alias.upper()}", fontweight='bold', fontfamily = 'monospace', fontsize = 16)
        fig.set_constrained_layout(True)

        return fig, axs

    _fig, _axs = plot_sentiment_maturity_heatmap(df_test, TERMS_ALL['solar'])
    _fig
    return


@app.cell
def _(
    Literal,
    TERMS_ALL,
    TermCollection,
    companies,
    mpl,
    n_rts,
    np,
    pd,
    plt,
    util,
):
    def discrete_cmap(bounds:list|np.ndarray, cmap_name:str = 'RdYlGn') -> mpl.colors.ListedColormap:
        '''
        Create a discrete colormap for quantum sentiment/maturity values.

        Parameters
        ----------
        bounds : list | np.ndarray
            The boundaries for the colormap.
        cmap_name : str, optional
            The name of the matplotlib colormap to use. Default is 'RdYlGn'.

        Returns
        -------
        mpl.colors.ListedColormap
            The discrete colormap.
        '''
        from matplotlib import colors as colours
        cmap = mpl.colormaps[cmap_name]
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)

        # extract all colours
        cmaplist = np.array([cmap(i) for i in range(cmap.N)])

        # create the new map
        cmap = colours.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, len(bounds))

        # define the bins and normalize
        norm = colours.BoundaryNorm(bounds, len(bounds))

        return cmap
        # cmap = mpl.colormaps.get_cmap(cmap_name)
        # colors = [cmap(i / (len(bins) - 1)) for i in range(len(bins) - 1)]
        # return mpl.colors.ListedColormap(colors, name=f'{cmap_name}_quantum')

    def plot_terms_heatmap(
            df: pd.DataFrame, question: Literal['q1_sentiment', 'q2_maturity', 'q3_stage', 'sentiment', 'maturity', 'stage'], *, 
            terms: TermCollection = 'all', 
            figsize: tuple = (12, 15),
            cmap: str|None = None,
            return_type: Literal['figure', 'axs', 'both'] = 'both'
        ) -> plt.Figure:
        '''
        Plot a heatmap of sentiment and maturity for all terms across companies and report types.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the results of the analysis.
        question : Literal['q1_sentiment', 'q2_maturity', 'sentiment', 'maturity']
            The question to plot.
        terms : TermCollection, optional
            The collection of terms to plot. Default is TERMS_ALL.
        figsize : tuple, optional
            The size of the figure in inches. Default is (12, 15).

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the heatmap.
        '''

        if question not in ['q1_sentiment', 'q2_maturity', 'q3_stage']:
            if question.lower() in ['sentiment', 'maturity', 'stage']:
                question = f'q{1 if question.lower() == "sentiment" else 2 if question.lower() == "maturity" else 3}_{question.lower()}'
            else:
                raise ValueError(f"Question '{question}' should be either 'q1_sentiment', 'q2_maturity', 'q3_stage', 'sentiment', 'maturity', or 'stage'.")
        if not isinstance(terms, TermCollection):
            if terms.lower() == 'all':
                terms = df['term_alias'].unique()
                terms = [TERMS_ALL[t] for t in terms]
            elif isinstance(terms, list):
                terms = [TERMS_ALL[t] for t in terms]
            else:
                raise ValueError(f"Terms should be a TermCollection or 'all', got {type(terms)} instead.")

        #fig, axs = plt.subplots(len(terms), df['report_type'].unique().shape[0], figsize=figsize, sharey=True, sharex=True)

        fig = plt.figure(layout="constrained", figsize=figsize)

        outer = fig.add_gridspec(
            nrows=2,
            ncols=3,
            height_ratios=[60, 1],
            width_ratios = [1, 18, 1]
        )

        main = outer[0, :].subgridspec(len(terms), n_rts)

        axs = np.array([
            [fig.add_subplot(main[i, j]) for j in range(n_rts)]
            for i in range(len(terms))
        ])

        cmaps = {
            'q1_sentiment': 'RdYlGn',
            'q2_maturity': discrete_cmap([1, 2, 3, 4, 5, 6, 7, 8, 9], cmap_name='cividis'),
            'q3_stage': discrete_cmap([1, 2, 3, 4, 5], cmap_name='plasma'),
        }
        if cmap is not None:
            cmaps[question] = cmap


        for i, term in enumerate(terms):
            # add term alias as a y-axis label for the first column
            axs[i, 0].set_ylabel(term.alias.replace(' ', '\n'), rotation=25, labelpad=40, fontweight='bold', va='center', fontfamily='monospace')
            for j, rt in enumerate(df['report_type'].unique()):
                df_question = df.loc[(df['report_type'] == rt) & (df['term_alias'] == term.alias)].pivot(index='company', columns='year', values=question)
                if question == 'q3_stage':
                    df_question = df_question.replace({
                        "research and development": 1,
                        "pilot": 2,
                        "demonstration": 3,
                        "commercial-deployment": 4,
                        "phasing-out": 5,
                        "unclear": -1,
                        None: -1
                    }).astype(float)
                else:
                    df_question = df_question.astype(float)
                im = axs[i, j].imshow(
                    df_question.replace(-1, np.nan).values,
                    cmap=cmaps[question]
                )
                if i == 0:
                    axs[i, j].set_title(f"{util.REPORT_TYPES_LONG.get(rt, rt).title()}s", fontweight='bold', fontfamily='monospace')
                #axs[i, j].set_title(f"{util.REPORT_TYPES_LONG.get(rt, rt).title()} - {question.split('_')[1].capitalize()}")
                axs[i, j].set_yticks(np.arange(len(df_question.index)),
                                      df_question.reset_index()['company'].apply(lambda x: companies.COMPANY_ALIASES.get(x, x)),
                                      rotation = 0,# va = 'center', 
                                      fontfamily='monospace',
                                      fontsize = 8)
                axs[i, j].set_xticks(np.arange(len(df_question.columns)),minor = True)
                axs[i, j].set_xticks(
                    np.arange(len(df_question.columns), step = 5),
                    np.arange(df_question.columns.astype('int').min(), df_question.columns.astype('int').max() + 1, 5), 
                    minor = False, 
                    rotation = 90,
                )

        for ax in axs.flat:
            ax.label_outer()

        # add colorbar to the bottom of the figure, spanning all columns
        #cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.01])
        if 'sentiment' in question.lower():
            ticks = [-1, -0.5, 0, 0.5, 1]
            min_max = (-1, 1)
            cbar_label = 'Sentiment'
        elif 'maturity' in question.lower():
            ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            tick_labels = ticks
            min_max = (0, 9)
            cbar_label = 'Technology Readiness Level'
        elif 'stage' in question.lower():
            ticks = [1, 2, 3, 4, 5]
            tick_labels = ['R&D', 'Pilot', 'Demonstration', 'Deployment', 'Phasing Out']
            min_max = (0, 5)
            cbar_label = 'Commercialisation Stage'

        # cr = mpl.colorizer.Colorizer(cmap=cmaps[question], norm=mpl.colors.Normalize(vmin=min_max[0], vmax=min_max[1]))
        # cbar = fig.colorbar(mpl.colorizer.ColorizingArtist(cr), cax=cbar_ax, orientation='horizontal')

        mappable = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=min_max[0], vmax=min_max[1]),
            cmap=cmaps[question],
        )
        mappable.set_array([])

        cax = fig.add_subplot(outer[1, 1])
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")

        if 'sentiment' in question.lower():    
            cbar.set_ticks(ticks)
        else:
            bounds = [0] + ticks
            cbar.set_ticks([(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds) - 1)])
            cbar.set_ticklabels(tick_labels)
            cbar.minorticks_off()
        cbar.set_label(cbar_label, rotation=0, fontweight='bold', fontfamily='monospace')
        #fig.subplots_adjust(bottom=0.20)

        fig.suptitle(f"{question.split('_')[1].upper()}", fontsize=16, fontweight='bold', fontfamily='monospace')
        fig.set_constrained_layout(True)

        if return_type.lower() == 'figure':
            return fig
        elif return_type.lower() == 'axs' or return_type.lower() == 'axes':
            return axs
        elif return_type.lower() == 'both':
            return fig, axs


    return discrete_cmap, plot_terms_heatmap


@app.cell
def _(df_test, mo, plot_terms_heatmap):
    mo.ui.tabs({
        'Sentiment': plot_terms_heatmap(df_test, 'q1_sentiment', figsize=(9,15), return_type='figure'),
        'Maturity': plot_terms_heatmap(df_test, 'q2_maturity', figsize=(9,15), return_type='figure'),
        'Stage': plot_terms_heatmap(df_test, 'q3_stage', figsize=(9,15), return_type='figure')
    }, label = 'Question', lazy = False)
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
