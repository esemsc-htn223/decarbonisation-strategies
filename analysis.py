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

    from src.decarb.llm.rag import search_chunks
    from src.decarb.llm import inspection
    from src.decarb.llm import pipeline

    return (
        COS,
        TERMS_ALL,
        TERMS_SIMPLE,
        TERMS_SUMMARY,
        inspection,
        json,
        mo,
        os,
        pd,
        pipeline,
        plot_comparison,
        plt,
        pymupdf4llm,
        regex_pipeline,
        search_chunks,
        tiktoken,
    )


@app.cell
def _(os):
    _chunks_dir = 'data/md/total/annual-report/total-form-20-f-2023'
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
def _(TERMS_ALL, chunks, inspection, search_chunks):
    terms = TERMS_ALL

    search_res = search_chunks(chunks, terms)
    inspection.search_summary(chunks, terms, search_res=search_res)
    return (search_res,)


@app.cell
def _(chunks, inspection, plt, search_res):
    fig, ax = inspection.plot_search_results(search_res, chunks)
    plt.show()
    return


@app.cell
def _(pipeline):
    pipeline.stage_convert(overwrite=False)
    return


@app.cell
def _(pipeline):
    pipeline.stage_search(progress_bar=True)
    return


@app.cell
def _(json, os, pipeline, tiktoken):
    # count number of md docs in data/md
    md_dir = os.path.join(pipeline.DATA_DIR, 'md')

    # count total no. tokens
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = {}
    md_docs = {}
    md_docs_unique = {}
    total_md_docs = 0

    for root, dirs, files in os.walk(md_dir):
        for filename in files:
            if not filename.endswith('.md'):
                continue
            total_md_docs += 1
        
            json_path = root.replace('data/md', 'out/all-terms/chunk-search-results') + '.json'
            _fname = json_path.split('/')[-1].replace('.json', '')
            if _fname not in tokens:
                tokens[_fname] = 0

            with open(json_path, "r") as _f:
                _relevant_inds = sum(json.load(_f).values(), [])
        
            _chunk_i = int(filename.replace('chunk-', '').replace('.md', ''))
            if _chunk_i in set(_relevant_inds):
                md_docs[_fname] = md_docs.get(_fname, 0) + 1 * _relevant_inds.count(_chunk_i)
                md_docs_unique[_fname] = md_docs_unique.get(_fname, 0) + 1
                with open(os.path.join(root, filename), "r") as f:
                    content = f.read()
                    tokens[_fname] += len(encoding.encode(content)) * _relevant_inds.count(_chunk_i)

    total_tokens = sum(tokens.values())
    print(f'No. chunks to pass to model: \n\t{sum(md_docs.values()):,}')
    print(f'\t--> no. unique: \n\t\t{sum(md_docs_unique.values()):,} / {total_md_docs:,}')
    print(f'No. tokens to pass to model: \n\t{total_tokens:,}')
    print(f'Approx min. cost (not accounting for header and metadata tokens):\n\t(batch, gpt-5.1) ${total_tokens * 0.625/1000000:.2f}\n\t(batch, gpt-5.4) ${total_tokens * 1.25/1000000:.2f}')
    return (tokens,)


@app.cell
def _(pd, tokens):
    df_tokens = pd.Series(tokens, name = 'tokens').sort_values(ascending=False).to_frame()
    df_tokens['co'] = df_tokens.index.str.split('-').str[0]
    df_tokens['year'] = df_tokens.index.str.split('-').str[-1]
    df_tokens.index.name = 'file'
    df_tokens.loc[df_tokens['co'] == 'total']
    return


@app.cell
def _(pymupdf4llm):
    #pymupdf4llm.use_layout(False)
    #doc = pymupdf.open('data/raw/total/annual-report/total-form-20-f-2025 copy.pdf')

    pymupdf4llm.to_markdown('data/raw/total/annual-report/total-form-20-f-2025 copy.pdf')

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipeline run
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
