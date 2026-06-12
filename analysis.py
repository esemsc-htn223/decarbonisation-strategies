import marimo

__generated_with = "0.23.9"
app = marimo.App()


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
    import pymupdf4llm
    import pymupdf
    import tiktoken
    import json
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from decarb import (
        regex_pipeline, plot_comparison, plot_mentions, plot_rankings, 
        Term, TermCollection, TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE,
        COS, Company
    )
    from decarb import util, companies, plotting
    from decarb.llm import rag, inspection, pipeline, prompt

    return (
        COS,
        TERMS_ALL,
        TERMS_SIMPLE,
        TERMS_SUMMARY,
        Term,
        TermCollection,
        companies,
        inspection,
        json,
        load_dotenv,
        mo,
        mpl,
        np,
        os,
        pd,
        pipeline,
        plot_comparison,
        plt,
        prompt,
        rag,
        regex_pipeline,
        tiktoken,
        util,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Workshop floor
    """)
    return


@app.cell
def _(companies, os, prompt):
    model = 'gpt-4.1'
    overwrite = True

    file_paths = []
    for report in companies.co_files(yield_level='Report', companies=None):
        for term_folder in report.term_folder_names():
            if report.prompts_dir(term_folder) is None:
                continue
            for prompt_file in os.listdir(report.prompts_dir(term_folder)):
                if prompt_file.endswith('.txt'):
                    file_paths.append(report.prompt_path(term_folder, prompt_file))
    #logger_llm.info(f'COLLECTED PROMPTS: {len(file_paths)} prompt files found in .{len(list(companies.co_files(yield_level="Report", companies=companies)))} reports.')

    # create jsonl for openai batch
    jsonl_path_test = prompt.create_openai_jsonl(
        prompt_paths=file_paths,
        model=model,
        save=True,
        overwrite=overwrite,
        return_type='path'
    )
    jsonl_path_test
    return (jsonl_path_test,)


@app.cell
def _(jsonl_path_test, mo, np, plt, prompt):
    _mt = 5000
    _total = 0
    #print(f'Max tokens per batch:\n\t{_mt:,}')
    #print(f'Rate limits:\n\t500,000/min --> ~{5e5//_mt} batches/min\n\t900,000/day --> ~{9e5//_mt} batches/day')
    tokens = []
    lens = []
    for _i, _portion in enumerate(prompt.jsonl_portions(
        jsonl_path_test, max_tokens = _mt
    )):
        #print(f'{_i}:\t{_portion}')
        _total += _portion.token_count
        lens.append(len(_portion))
        tokens.append(_portion.token_count)
        #if _i == 6:
        #    break
    #print(f'Total tokens: {_total:,}')

    _violin_kwargs = dict(
        orientation='horizontal',
        #showmeans=True, 
        showmedians=True, 
        quantiles=[[0.25, 0.75]],
        side = 'high',
        showextrema=True
    )
    _fig, _axs = plt.subplots(2, 1, figsize=(10, 6))
    _ax_tk = _axs[0]
    _ax_ln = _axs[1]
    #_ax_tk.plot(tokens, marker='x', linestyle='None')
    _ax_tk.violinplot(tokens, **_violin_kwargs)
    _ax_tk.set_xlabel('Tokens in portion')
    _ax_tk.set_ylabel('PDF')
    #_ax_tk.set_yscale('log')

    #_ax_ln.plot(lens, marker='x', linestyle='None')
    #_ax_ln.violinplot(lens, **_violin_kwargs)
    _ax_ln.hist(lens, bins=np.arange(0.5, max(lens)+1.5, 1), alpha = 0.7)
    _ax_ln.set_xlabel('Portion length')
    mo.ui.matplotlib(_ax_tk)
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
    return search_res, terms


@app.cell
def _(chunks, inspection, plt, search_res):
    fig, ax = inspection.plot_search_results(search_res, chunks)
    plt.grid(which = 'major', axis = 'y', color = 'k')
    plt.show()
    return


@app.cell
def _(pipeline):
    pipeline.stage_convert(overwrite=False, progress_bar=True)
    return


@app.cell
def _(pipeline):
    pipeline.stage_chunk(overwrite=False)
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
def _(companies, os, pd, terms, tiktoken, util):
    # count number of md docs in data/md

    # count total no. tokens
    encoding = tiktoken.get_encoding("cl100k_base")

    # create dataframe with columns: company, report_type, report_year, technology, num_tokens
    df_tokens = {'company': [], 'report_type': [], 'report_year': [], 'technology': [], 'num_tokens': [], 'length_chars': []}

    term_map = {'solar':'Solar',
    'wind':'Wind',
    'hydrogen_green':'Hydrogen (green)',
    'hydrogen_blue':'Hydrogen (blue)',
    'hydrogen':'Hydrogen',
    'ccs':'CCS',
    'biofuel':'Biofuel',
    'biogas':'Biogas',
    'geothermal':'Geothermal',
    'tidal':'Tidal',
    'storage':'Storage',
    'evs':'EVs',
    'nature':'Nature',
    'efficiency':'Efficiency',
    'nuclear_conventional':'Nuclear (conventional)',
    'nuclear_smr':'Nuclear (SMR)',
    'nuclear_fusion':'Nuclear fusion'}

    # add to dataframe in loop below instead of tokens dict
    for co in os.listdir(util.COMPANIES_DIR):
        co_dir = os.path.join(util.COMPANIES_DIR, co)
        if not os.path.isdir(co_dir):
            continue

        for report_type_folder_name in os.listdir(co_dir):
            report_type_dir = os.path.join(co_dir, report_type_folder_name)
            if not os.path.isdir(report_type_dir):
                continue

            for report_folder_name in os.listdir(report_type_dir):
                report_dir = os.path.join(report_type_dir, report_folder_name)

                if not os.path.isdir(report_dir):
                    continue

                prompts_dir = os.path.join(report_dir, terms.folder_name, 'prompts')

                for prompt_filename in os.listdir(prompts_dir):
                    if not prompt_filename.endswith('.txt'):
                        continue
                    with open(os.path.join(prompts_dir, prompt_filename), 'r') as f:
                        prompt_content = f.read()
                        num_tokens = len(encoding.encode(prompt_content))
                        df_tokens['company'].append(companies.COMPANY_ALIASES.get(co, co))
                        df_tokens['report_type'].append(report_type_folder_name.split('-')[0].capitalize())
                        df_tokens['report_year'].append(report_folder_name)
                        df_tokens['technology'].append(term_map.get(prompt_filename.split('.')[0], prompt_filename.split('.')[0]))
                        df_tokens['num_tokens'].append(num_tokens)
                        df_tokens['length_chars'].append(len(prompt_content))
    df_tokens = pd.DataFrame(df_tokens)
    df_tokens['report_year'] = df_tokens['report_year'].str.split('-').str[-1].astype(int)
    df_tokens['cost_batch_gpt5_1'] = df_tokens['num_tokens'] * 0.625/1000000
    df_tokens['cost_batch_gpt5_4'] = df_tokens['num_tokens'] * 1.25/1000000

    print(f'No. prompts: \n\t{len(df_tokens):,}')
    print(f'No. tokens to pass to model: \n\t{df_tokens["num_tokens"].sum():,}')
    print(f'Approx. cost (not accounting for header and metadata tokens):\n\t(batch, gpt-5.1) ${df_tokens["cost_batch_gpt5_1"].sum():.2f}\n\t(batch, gpt-5.4) ${df_tokens["cost_batch_gpt5_4"].sum():.2f}')
    return df_tokens, prompts_dir


@app.cell
def _(df_tokens):
    df_tokens.sort_values('num_tokens', ascending=False)
    return


@app.cell
def _(TERMS_ALL, df_tokens, plt):

    _fig, _axs = plt.subplots(2,5, figsize=(21,5), sharex=True, 
                              sharey=True)
    _axs_an = _axs[0,:5]
    _axs_sr = _axs[1,:5]
    for _i, _co in enumerate(df_tokens['company'].unique()):
        _df_co = df_tokens[df_tokens['company'] == _co]

        for _tech in _df_co['technology'].unique():
            _df_tech_an = _df_co[(_df_co['technology'] == _tech) & (_df_co['report_type'] == 'Annual')]
            _axs_an[_i].scatter(
                _df_tech_an['report_year'], _df_tech_an['num_tokens'], 
                label=_tech, color = TERMS_ALL[_tech].colour,
                marker = 'x'
            )
            _df_tech_sr = _df_co[(_df_co['technology'] == _tech) & (_df_co['report_type'] == 'Sustainability')]
            _axs_sr[_i].scatter(
                _df_tech_sr['report_year'], _df_tech_sr['num_tokens'], 
                label=_tech, color = TERMS_ALL[_tech].colour,
                marker = 'x'
            )

        _axs_an[_i].set_title(_co)
        _axs_sr[_i].set_xlabel('Report Year')
        if _i == 0:
            _axs_an[_i].set_ylabel(r'$\bf{Annual}$'+'\nNo. Tokens')
            _axs_sr[_i].set_ylabel(r'$\bf{Sustainability}$'+'\nNo. Tokens')
        _axs_an[_i].grid(axis = 'y')
        _axs_sr[_i].grid(axis = 'y')
        #_axs[_i].legend()
    _fig.legend(*_axs_an[0].get_legend_handles_labels(), loc='upper center', ncol=17, bbox_to_anchor=(0.5, 0.0))
    _fig.suptitle('Tokens per Prompt')
    _fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return


@app.cell
def _(df_tokens):
    df_tokens.loc[df_tokens['company'] != 'TotalEnergies', 'cost_batch_gpt5_1'].sum()
    return


@app.cell
def _(os):
    for root, dirs, files in os.walk('data/terms/all-terms/prompts/'):
        for file in files:
            if file.endswith('.txt'):
                os.rename(os.path.join(root, file), os.path.join(root, file.replace('_', '-')))
                print(file + ' -> ' + file.replace('_', '-'))
    return file, root


@app.cell
def _(file, os, root):
    os.path.join(root, file.replace('_', '-'))
    return


@app.cell
def _(TERMS_ALL, os, prompt, prompts_dir):
    _exclude = ['total', 'exxon', 'chevron']
    _co_dirs = [os.path.join(prompts_dir, d) for d in os.listdir(prompts_dir) if os.path.isdir(os.path.join(prompts_dir, d)) and d not in _exclude]

    prompt.create_openai_jsonl(co_dirs=_co_dirs, save_path=os.path.join(TERMS_ALL.folder, 'prompts', 'prompts.jsonl'), overwrite=True, return_data=False)
    return


@app.cell
def _(companies):
    len(list(companies.co_files(yield_level='Report')))
    return


@app.cell
def _(pipeline):
    jsonl_path = pipeline.stage_analyse()
    jsonl_path
    return


@app.cell
def _():
    import uuid
    _report_paths = [
        'data/companies/bp/annual-report/bp-annual-report-accounts-2007',
        'data/companies/exxon/annual-report/exxon-annual-report-2010',
        'data/companies/chevron/annual-report/chevron-annual-report-2015'
    ]
    _uuid1 = uuid.uuid5(uuid.NAMESPACE_OID, ''.join(_report_paths)).hex

    print(_uuid1+'.jsonl')
    #print(_report.report_name)
    return


@app.cell
def _(os):
    os.path.basename(os.path.dirname(os.path.dirname('data/companies/bp/annual-report/bp-annual-report-accounts-2007/all-terms/prompts/biofuel.txt')))
    return


@app.cell
def _(TERMS_ALL, os, prompt, prompts_dir):
    #_exclude = ['total', 'exxon', 'chevron']
    #_co_dirs = [os.path.join(prompts_dir, d) for d in os.listdir(prompts_dir) if os.path.isdir(os.path.join(prompts_dir, d)) and d not in _exclude]
    _include = ['exxon']
    _co_dirs = [os.path.join(prompts_dir, d) for d in os.listdir(prompts_dir) if os.path.isdir(os.path.join(prompts_dir, d)) and d in _include]

    test_jsonl = prompt.create_openai_jsonl(
        co_dirs=_co_dirs, 
        save_path=os.path.join(TERMS_ALL.folder, 'prompts', 'exxon-test.jsonl'),
        model='gpt-4.1',
        overwrite=True,
        return_data=True
    )
    return (test_jsonl,)


@app.cell
def _(test_jsonl, tiktoken):
    # get token estimate for test_jsonl using best estimate for gpt-4.1
    tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens_test_jsonl = sum(len(tiktoken_encoding.encode(entry['body']['messages'][0]['content'])) for entry in test_jsonl)
    print(f'Estimated number of tokens for test_jsonl: {num_tokens_test_jsonl:,}')
    return


@app.cell
def _(TERMS_ALL, os, prompt):
    batch_id = prompt.submit_openai_batch(
       jsonl_path=os.path.join(TERMS_ALL.folder, 'prompts', 'exxon-test.jsonl'),
       terms_dir=os.path.join(TERMS_ALL.folder)
    )
    #batch_id = 'batch_6a185de3bdc48190b5f1a02b6d20de32'
    batch_id
    return (batch_id,)


@app.cell
def _(TERMS_ALL, batch_id, os, prompt):
    prompt.check_batch_status(
        batch_id, 
        dir_to_search=os.path.join(TERMS_ALL.folder)
    )
    return


@app.cell
def _(json, load_dotenv, mo):
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()

    for line in client.files.content('file-V2nb8HDBHYMuDvoFZwrR1A').iter_lines():
        _acc = mo.accordion(json.loads(line))
        break

    _acc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial output analysis
    """)
    return


@app.cell
def _(TERMS_ALL, companies, json, pd):
    exxon_test_in = {}
    with open('data/terms/all-terms/prompts/ccdc991a9fab55ad80badf4ca78d2bd1.jsonl', 'r') as _f:
        for _line in _f:
            _json_line = json.loads(_line)
            exxon_test_in[_json_line['custom_id']] = _json_line['body']['messages'][0]['content']

    exxon_test_out = []

    #for _entry in client.files.content('file-EpvoDg9q4jE5xPikkq2ZGb').iter_lines():  # to get from openai API
    for _entry in open('/Users/hugoneely/Documents/4-work/1-current/PhD/repos/decarbonisation-strategies/out/test_output.jsonl', 'r'):  # file locally saved
        _entry = json.loads(_entry)
        _id = _entry['custom_id']
        _co, _rt, _yr, _term = _id.split('_')
        _entry = json.loads(_entry['response']['body']['choices'][0]['message']['content'])
        _term = TERMS_ALL[_term]

        _entry['id'] = _id
        _entry['company'] = _co
        _entry['company_alias'] = companies.COMPANY_ALIASES.get(_co, _co)
        _entry['report_type'] = _rt
        _entry['term_alias'] = _term.alias
        _entry['term_regex'] = _term.regex
        _entry['term_fname'] = _term.prompt_fname
        _entry['year'] = _yr
        _entry['prompt'] = exxon_test_in[_id]
        exxon_test_out.append(_entry)


    _first_cols = ['company', 'report_type', 'term_alias', 'year']
    df_exxon_test = pd.DataFrame(exxon_test_out).set_index('id')
    _cols = _first_cols + df_exxon_test.drop(columns=_first_cols).columns.to_list()
    df_exxon_test = df_exxon_test.sort_values(_first_cols)[_cols]

    df_exxon_test
    return (df_exxon_test,)


@app.cell
def _():
    stg_map = {
        "research and development": {
            'alias': "1/4 - R&D",
            'style': {'color': '#facfd2'}
        },
        "pilot": {
            'alias': "2/4 - Pilot",
            'style': {'color': "#ff8085"}
        },
        "demonstration": {
            'alias': "3/4 - Demonstration",
            'style': {'color': '#e65054'}
        },
        "commercial-deployment": {
            'alias': "4/4 - Deployment",
            'style': {'color': '#b32d2e'}
        },
        "unclear": {
            'alias': "-. - Unclear",
            'style': {'color': '#cccccc'}
        }
    }


    mat_map = {
        "emerging": {
            'alias': "1/3 - Emerging",
            'style': {'color': '#a8e6cf'}
        },
        "established": {
            'alias': "2/3 - Established",
            'style': {'color': '#5d8aa8'}
        },
        "phasing-out": {
            'alias': "3/3 - Phasing Out",
            'style': {'color': '#e65054'}
        },
        "unclear": {
            'alias': "-. - Unclear",
            'style': {'color': '#cccccc'}
        }
    }
    return mat_map, stg_map


@app.cell
def _(TERMS_ALL, companies, df_exxon_test, inspection, mat_map, mo, stg_map):
    _entry = df_exxon_test.iloc[0]
    _co = _entry['company_alias']
    _rt = _entry['report_type']
    _yr = _entry['year']

    _mat = _entry['maturity']
    _mat_map = mat_map.get(_mat, {'alias': _mat.title(), 'style': {}})
    _mat_style = '; '.join(f"{_k}: {_v}" for _k, _v in _mat_map["style"].items())
    _stg = _entry['stage']
    _stg_map = stg_map.get(_stg, {'alias': _stg.title(), 'style': {}})
    _stg_style = '; '.join(f"{_k}: {_v}" for _k, _v in _stg_map["style"].items())
    _snt = _entry['sentiment']

    _chunks = inspection.search_results_chunks(companies.Report.from_metadata(_co, _rt, _yr), TERMS_ALL, _entry['term_alias'])
    mo.accordion(
        {
            _entry['year']: mo.vstack([
                mo.vstack([
                    _entry['notes'],
                    mo.md('---'),
                    mo.hstack([
                        mo.md(f'**Sentiment:** {_snt}'),
                        mo.md('|'),
                        mo.md(f'**Stage:** <span style="{_stg_style}">{_stg_map["alias"]}</span>'),
                        mo.md('|'),
                        mo.md(f'**Maturity:** <span style="{_mat_style}">{_mat_map["alias"]}</span>'),
                    ], justify = 'center')
                ]),
                mo.md('---'),
                mo.accordion({
                    f'{len(_chunks)} Chunk{"s" if len(_chunks) != 1 else ""}:': mo.vstack(_chunks.values()),
                }),

            ])
        }
    )
    return


@app.cell
def _(df_exxon_test, mo):
    term_dropdown = mo.ui.dropdown(
        df_exxon_test['term_alias']
    )
    rt_dropdown = mo.ui.dropdown(
        df_exxon_test['report_type']
    )
    return


@app.cell
def _(
    Term,
    TermCollection,
    companies,
    inspection,
    mat_map,
    mo,
    mpl,
    pd,
    stg_map,
):
    def mark_term_mentions(md: str, term: Term):
        import re
        matches = re.finditer(term.regex, md, re.I)
        prev_shift = 0
        for match in matches:
            start, end = match.span()
            md = md[:start + prev_shift] + f'<u><mark>{md[start + prev_shift:end + prev_shift]}</mark></u>' + md[end + prev_shift:]
            prev_shift += 20  # len('<u><mark></mark></u>')
        return md

    def mark_commitments(md: str, commitments: list) -> str:
        import re
        for commitment in commitments:
            matches = re.finditer(commitment, md, re.I)
            prev_shift = 0
            for match in matches:
                start, end = match.span()
                md = md[:start + prev_shift] + f'<mark style="background-color: #ffcccb">{md[start + prev_shift:end + prev_shift]}</mark>' + md[end + prev_shift:]
                prev_shift += 47  # len('<mark style="background-color: #ffcccb"></mark>')
        return md

    def count_commitment_mentions(md: str, commitments: list) -> int:
        import re
        count = 0
        for commitment in commitments:
            matches = re.findall(commitment, md, re.I)
            count += len(list(matches))
        return count

    def term_year_analysis(entry: pd.Series, terms: TermCollection, *, return_key: bool = False) -> mo.Html:
        co = entry['company_alias']
        rt = entry['report_type']
        yr = entry['year']
        term = terms[entry['term_alias']]

        mat = entry['maturity']
        mat_details = mat_map.get(mat, {'alias': mat.title(), 'style': {}})
        mat_style = '; '.join(f"{k}: {v}" for k, v in mat_details["style"].items())
        stg = entry['stage']
        stg_details = stg_map.get(stg, {'alias': stg.title(), 'style': {}})
        stg_style = '; '.join(f"{k}: {v}" for k, v in stg_details["style"].items())
        snt = entry['sentiment']

        chunks = inspection.search_results_chunks(companies.Report.from_metadata(co, rt, yr), terms, entry['term_alias'])

        notes = f"> {entry['notes']}"
        if entry['commitments']:
            cmts = ''
            for cmt in entry['commitments']:
                n_quotes_cmt = sum(count_commitment_mentions(chunk, [cmt]) for chunk in chunks)
                if n_quotes_cmt > 0:
                    cmts += f'\n- QUOTE: <mark style="background-color: #ffcccb">{cmt}</mark>'
                else:
                    cmts += f'\n- {cmt}'
            n_quotes = sum(count_commitment_mentions(chunk, entry['commitments']) for chunk in chunks)
            notes += f'\n\n**Commitments:** (quotes: {n_quotes}/{len(entry["commitments"])})\n' + cmts
    
        chunks = [mo.md(
            mark_term_mentions(
                mark_commitments(chunk, entry['commitments']), 
                term
            )) for chunk in chunks]

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
                    mo.md('---'),

                    mo.accordion({
                        f'{len(chunks)} Chunk{"s" if len(chunks) != 1 else ""}:': mo.vstack(chunks),
                    }),
                ])
        if return_key:
            snt_cmap = mpl.colormaps['RdYlGn'].resampled(21)
            snt_colour = mpl.colors.to_hex(snt_cmap((snt + 1) / 2))
            n_commitments = len(entry['commitments'])
            key = f'**{yr}:** {snt}<span style="color: {snt_colour}">█</span> \t|\t <span style="{stg_style}">{stg_details["alias"]}</span> \t|\t <span style="{mat_style}">{mat_details["alias"]}</span>'
            if n_commitments > 0:
                key += f' \t|\t {n_commitments} commitment{"s" if n_commitments != 1 else ""}'
            out = (key, out)
        return out

    return (term_year_analysis,)


@app.cell
def _(TERMS_ALL, companies, df_exxon_test, mo, term_year_analysis, util):
    co_tabs = {}
    for _co in df_exxon_test['company'].unique():
        rt_tabs = {}
        for _rt in df_exxon_test['report_type'].unique():
            _term_tabs = {}
            _rt_long = util.REPORT_TYPES_LONG.get(_rt, _rt).title()
            for _term in df_exxon_test['term_alias'].unique():
                _regex = df_exxon_test[df_exxon_test['term_alias'] == _term]['term_regex'].iloc[0]
                _df_term = df_exxon_test[(df_exxon_test['report_type'] == _rt) & (df_exxon_test['term_alias'] == _term)]
                if _df_term.empty:
                    continue
                _yr_accordion = {}
                for _yr in _df_term['year'].unique():
                    _entry = _df_term[_df_term['year'] == _yr].iloc[0]
                    _key, _value = term_year_analysis(_entry, TERMS_ALL, return_key=True)
                    _yr_accordion[_key] = _value

                _term_tabs[_term] = mo.vstack([
                    mo.ui.text_area(value = _regex, full_width = True, label = '**Regex:**'),
                    mo.accordion(_yr_accordion)
                ])
            rt_tabs[_rt_long] = mo.ui.tabs(_term_tabs, label = 'Term')
        co_tabs[companies.COMPANY_ALIASES.get(_co, _co)] = mo.ui.tabs(rt_tabs, label = 'Report Type')
    mo.ui.tabs(co_tabs, label = 'Company')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notes:
    - 'commitments'
        - field isn't made up of quotes - it's summaries from the text.
        - doesn't seem to be future-looking commitments. sometimes its just extracts of certain statements of some maybe related thing.
            - should separate out to 'key quotes' and 'future commitments'

    - 'maturity'
        - feel like this conceptually has too much overlap with 'stage'
        - Should clarify if this is the maturity of the company's implementation, or the maturity of the company's

    -
    """)
    return


@app.cell
def _(inspection):
    inspection.search_summary()
    return


@app.cell
def _(TERMS_ALL):
    TERMS_ALL['solar'].dash
    return


@app.cell
def _(TERMS_ALL, df_exxon_test, plt):


    term_fnames_to_term = {term.prompt_fname.replace('.txt', ''): term for term in TERMS_ALL}

    _fig, _axs = plt.subplots(2,1, figsize=(12,10), sharey=True, sharex=True)
    for _i, _rt in enumerate(df_exxon_test['report-type'].unique()):
        for _tech in df_exxon_test['technology'].unique():
            _df_tech = df_exxon_test[(df_exxon_test['report-type'] == _rt) & (df_exxon_test['technology'] == _tech)]
            _df_tech['year'] = _df_tech['year'].astype(int)
            _df_tech.sort_values('year', inplace=True)
            term = term_fnames_to_term[_tech]
            _axs[_i].plot(
                _df_tech['year'], _df_tech['sentiment'], 
                label=_tech, color = term.colour,
                linestyle = term.dash_ls if term.dash else '-',
                gapcolor = term.edgecolour,
                marker = term.marker if term.marker else 'o', alpha = 0.7
            )

        _axs[_i].set_title(_rt)
        _axs[_i].set_xlabel('Report Year')
        if _i == 0:
            _axs[_i].set_ylabel('Sentiment')
        _axs[_i].grid(axis = 'y')
        _axs[_i].set_ylim(-1,1)
        _axs[_i].tick_params(axis='x', rotation=45)

    _fig.legend(*_axs[0].get_legend_handles_labels(), loc='upper center', ncol=17, bbox_to_anchor=(0.5, 0.0))
    _fig.suptitle('Exxon - Technology Sentiment over Time')
    _fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regex Pipeline
    """)
    return


@app.cell
def _(COS, TERMS_ALL, TERMS_SIMPLE, TERMS_SUMMARY, regex_pipeline):
    out = regex_pipeline(TERMS_ALL, COS, reprocess=False, plot=True)
    out_sum = regex_pipeline(TERMS_SUMMARY, COS, reprocess=False, plot=True)
    out_simple = regex_pipeline(TERMS_SIMPLE, COS, reprocess=True, plot=True)
    return (out_simple,)


@app.cell
def _(TERMS_SIMPLE, out_simple, plot_comparison, plt):
    plot_comparison(out_simple['dfs'], terms = TERMS_SIMPLE, suptitle='Technology mentions - normalised by word count')
    plt.show()
    return


@app.cell
def _(COS, TERMS_SIMPLE, out_simple, plot_comparison, plt):
    plot_comparison(out_simple['dfs_unnormed'], terms=TERMS_SIMPLE, cos=COS, normed_by=None,
                    suptitle='Technology mentions - unnormalised')
    plt.show()
    return


@app.cell
def _(COS, TERMS_SIMPLE, out_simple, plot_comparison, plt):
    plot_comparison(out_simple['dfs_unnormed'], terms=TERMS_SIMPLE, cos=COS[:2], rank=True,
                    suptitle='Technology mention rankings')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
