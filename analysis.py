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

    from src.decarb import (
        regex_pipeline, plot_comparison, plot_mentions, plot_rankings, 
        Term, TermCollection, TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE,
        COS, Company
    )

    return (
        COS,
        TERMS_ALL,
        TERMS_SIMPLE,
        TERMS_SUMMARY,
        mo,
        np,
        plot_comparison,
        plt,
        regex_pipeline,
    )


@app.cell
def _():
    import pymupdf4llm
    import re
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    import os

    def tidy_md(md:str):
        tidy_md = md.replace('$', '\$')  # no maths in these reports
        tidy_md = re.sub(r'(-)\1{3,}', r'\1\1\1', tidy_md) # limit repeated - to 3
        tidy_md = re.sub(r'\n{3,}', r'\n\n', tidy_md) # limit repeated \n to 2
        return tidy_md

    pymupdf4llm.use_layout(True)
    md = pymupdf4llm.to_markdown('data/raw/bp/annual/bp-annual-report-accounts-2007.pdf')
    md = tidy_md(md)

    with open('test2_layout.md', 'w') as _f:
        _f.write(md[:3000])
    return (
        Document,
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
        md,
        os,
    )


@app.cell
def _(
    Document,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    md,
    os,
):
    TEST_DIR = 'test'
    os.makedirs(TEST_DIR, exist_ok=True)

    MAX_CHUNK_SIZE = 2000
    MIN_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 0

    def split(md, max_chunk_size=MAX_CHUNK_SIZE, min_chunk_size=MIN_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        headers_to_split_on = [
            #("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        chunks = md_splitter.split_text(md)

        # enforce maximum in the token space, with some overlap, and a minimum to avoid too-short chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            #separators=["\n\n", "\n", " ", ""],
            chunk_size=max_chunk_size, 
            chunk_overlap=chunk_overlap, 
            keep_separator=True
        )
        chunks = text_splitter.split_documents(chunks)

        # enforce minimum chunk size by merging short chunks with the next one
        i = 0
        while i < len(chunks):
            if len(chunks[i].page_content) < min_chunk_size:
                if i + 1 < len(chunks):
                    chunks[i] = Document(page_content=chunks[i].page_content + chunks[i + 1].page_content)
                    chunks.pop(i + 1)
                else:
                    chunks.pop(i)
            else:
                i += 1

        return [chunk.page_content for chunk in chunks]

    chunks = split(md)

    {
        'len(md)': len(md),
        'sum(out)': sum(len(c) for c in chunks),
        'no. chunks': len(chunks)
    
    }
    return MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, chunks


@app.cell
def _(chunks, mo):
    min_len = min(len(c) for c in chunks)
    max_len = max(len(c) for c in chunks)
    print(f'Min chunk length: {min_len} characters')
    print(f'Max chunk length: {max_len} characters')

    # identify chunk with minimum length
    min_chunk = min(chunks, key=lambda c: len(c))
    max_chunk = max(chunks, key=lambda c: len(c))

    mo.accordion(
        {
            'Min Chunk - 1': mo.md(chunks[chunks.index(min_chunk) - 1]),
            'Min Chunk': mo.md(min_chunk),
            'Min Chunk + 1': mo.md(chunks[chunks.index(min_chunk) + 1]),
            'Max Chunk - 1': mo.md(chunks[chunks.index(max_chunk) - 1]),
            'Max Chunk': mo.md(max_chunk),
            'Max Chunk + 1': mo.md(chunks[chunks.index(max_chunk) + 1]),
        },
        multiple=True
    )
    return


@app.cell
def _(chunks, mo, np):
    _inds = np.random.randint(0, len(chunks), size=5)

    mo.accordion({f'Chunk {i}': mo.md(chunks[i].page_content) for i in _inds}, multiple=True)
    return


@app.cell
def _(MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, chunks, plt):
    plt.figure(figsize=(10, 5))
    plt.hist([len(c.page_content) for c in chunks], bins = 100)
    plt.title('Histogram of Chunk Lengths')
    plt.xlabel('Length of Chunks (characters)')
    plt.ylabel('Frequency')
    plt.axvline(MAX_CHUNK_SIZE, color='red', linestyle='dashed', linewidth=1, label='Max Chunk Size')
    plt.axvline(MIN_CHUNK_SIZE, color='black', linestyle='dashed', linewidth=1, label='Min Chunk Size')
    #plt.legend()
    plt.show()
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
