import marimo

__generated_with = "0.23.6"
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

    from src.decarb import (
        regex_pipeline, plot_comparison, plot_mentions, plot_rankings, 
        Term, TermCollection, TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE,
        COS, Company
    )
    from src.decarb import util, companies

    from src.decarb.llm import rag
    from src.decarb.llm import inspection
    from src.decarb.llm import pipeline
    from src.decarb.llm import prompt

    return (
        COS,
        TERMS_ALL,
        TERMS_SIMPLE,
        TERMS_SUMMARY,
        companies,
        inspection,
        json,
        load_dotenv,
        mo,
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
    return (client,)


@app.cell
def _(client, json):
    exxon_test_out = []
    with open('out/test_output.jsonl', 'w') as _f:
        for _line in client.files.content('file-EpvoDg9q4jE5xPikkq2ZGb').iter_lines():
            exxon_test_out.append(json.loads(_line))
            _f.write(_line + '\n')

    exxon_test_out[:5]
    return (exxon_test_out,)


@app.cell
def _(exxon_test_out, json, pd):

    exxon_test_responses = []
    for i, entry in enumerate(exxon_test_out):
        _id = entry['custom_id']
        _co, _rt, _yr, _tech = _id.split('_')
        exxon_test_responses.append(json.loads(entry['response']['body']['choices'][0]['message']['content']))
        exxon_test_responses[-1]['id'] = _id
        exxon_test_responses[-1]['company'] = _co
        exxon_test_responses[-1]['report-type'] = _rt
        exxon_test_responses[-1]['year'] = _yr
        exxon_test_responses[-1]['technology'] = _tech

    df_exxon_test = pd.DataFrame(exxon_test_responses).set_index('id').sort_values(['report-type','technology', 'year'])

    df_exxon_test
    return (df_exxon_test,)


@app.cell
def _(TERMS_ALL):
    TERMS_ALL[0].dash
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
            term = term_fnames_to_term.get(_tech)
            _axs[_i].plot(
                _df_tech['year'], _df_tech['sentiment'], 
                label=_tech, color = term.colour if term else 'k',
                linestyle = term.dash if term else '-',
                marker = 'x', alpha = 0.7
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


@app.cell
def _():
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
