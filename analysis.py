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

    return (
        COS,
        TERMS_ALL,
        TERMS_SIMPLE,
        TERMS_SUMMARY,
        companies,
        inspection,
        mo,
        os,
        pd,
        pipeline,
        plot_comparison,
        plt,
        rag,
        regex_pipeline,
        tiktoken,
    )


@app.cell
def _(os):
    _chunks_dir = 'data/md/shell/annual-report/shell-annual-report-2025'
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
def _(TERMS_SIMPLE, chunks, inspection, rag):
    terms = TERMS_SIMPLE

    search_res = rag.search_chunks(chunks, terms)
    inspection.search_summary(chunks, terms, search_res=search_res)
    return (search_res,)


@app.cell
def _(chunks, inspection, plt, search_res):
    fig, ax = inspection.plot_search_results(search_res, chunks)
    plt.grid(which = 'major', axis = 'y', color = 'k')
    plt.show()
    return


@app.cell
def _(pipeline):
    pipeline.stage_convert(overwrite=False)
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
def _(TERMS_ALL, companies, os, pd, tiktoken):
    # count number of md docs in data/md
    prompts_dir = os.path.join(TERMS_ALL.folder, 'prompts')

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
    for co in os.listdir(prompts_dir):
        co_dir = os.path.join(prompts_dir, co)
        if not os.path.isdir(co_dir):
            continue
    
    
        for report_type_foldername in os.listdir(co_dir):
            report_type_dir = os.path.join(co_dir, report_type_foldername)
            if not os.path.isdir(report_type_dir):
                continue
        
            for report_foldername in os.listdir(report_type_dir):
                report_dir = os.path.join(report_type_dir, report_foldername)
                if not os.path.isdir(report_dir):
                    continue
            
                for prompt_filename in os.listdir(report_dir):
                    if not prompt_filename.endswith('.txt'):
                        continue
                    with open(os.path.join(report_dir, prompt_filename), 'r') as f:
                        prompt_content = f.read()
                        num_tokens = len(encoding.encode(prompt_content))
                        df_tokens['company'].append(companies.company_aliases.get(co, co))
                        df_tokens['report_type'].append(report_type_foldername.split('-')[0].capitalize())
                        df_tokens['report_year'].append(report_foldername)
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
    return (df_tokens,)


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
def _(mo):
    sample_response = {
      "sentiment": 0.3,
      "key-commitments": [
        "The production facilities for these fields will mainly be powered by renewable energy from solar power offshore and hydroelectric power onshore.",
        "In the first quarter of 2025, Shell subsidiary Savion achieved full commercial operation of two solar facilities in the USA: the 111 MW Martin County Solar Project in Kentucky... and the 100 MW Kiowa County Solar Project in Oklahoma.",
        "In July 2025, Savion established Tango Holdings, LLC, a joint venture with a fund managed by Ares Infrastructure Opportunities. The venture will oversee 496 MW of solar capacity.",
        "We enable renewable power generation by owning and participating in joint ventures for and operating solar plants, primarily in India, the USA and Europe.",
        "Capex for five economic activities was assessed as taxonomy-aligned, including... electricity from solar.",
        "Cutting emissions from oil and gas production in our Upstream business... by looking at ways to electrify our offshore oil facilities, including using wind and solar power to reduce operational emissions."
      ],
      "stage": "commercial-deployment",
      "maturity": "established",
      "notes": "Shell's stance on solar is pragmatic and financially-driven rather than expansionary. While solar is clearly at commercial-deployment scale (multiple operational GW-scale plants, taxonomy-aligned capex of $568M), the overall strategic direction shows deliberate portfolio reduction: Shell sold Italian solar projects, divested Cleantech Renewable Assets (India/SE Asia solar platform), transferred majority ownership of US solar assets to the Tango JV, and exited Madison Fields Solar. The framing is explicitly one of 'selectively develops renewable generation projects and reduces ownership as assets mature' and a shift 'from capital-intensive renewables development to a capital-light, trading-led, asset-backed model'. Solar is treated as established and mature enough to monetise and recycle capital from, rather than a growth priority. The moderately positive sentiment score reflects genuine operational commitment and ongoing capex, tempered by the clear strategic pivot away from ownership."
    }
    mo.accordion(sample_response)
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
