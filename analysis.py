import marimo

__generated_with = "0.23.5"
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

    from src.decarb import (
        run_pipeline, plot_comparison, plot_mentions, plot_rankings, 
        Term, TermCollection, TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE,
        COS, Company
    )

    return (
        COS,
        TERMS_ALL,
        TERMS_SIMPLE,
        TERMS_SUMMARY,
        mo,
        plot_comparison,
        plt,
        run_pipeline,
    )


@app.cell
def _(COS, TERMS_ALL, TERMS_SIMPLE, TERMS_SUMMARY, run_pipeline):
    out = run_pipeline(TERMS_ALL, COS, reprocess=False, plot=True)
    out_sum = run_pipeline(TERMS_SUMMARY, COS, reprocess=False, plot=True)
    out_simple = run_pipeline(TERMS_SIMPLE, COS, reprocess=True, plot=True)
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
