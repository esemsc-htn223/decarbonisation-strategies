import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regex pipeline
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from decarb.terms import (
        TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE
    )
    from decarb.companies import COMPANIES_ALL
    from decarb.regex import regex_pipeline
    from decarb.output.plot import regex_comparison
    from decarb.util import OUT_DIR

    return (
        COMPANIES_ALL,
        OUT_DIR,
        TERMS_SIMPLE,
        mo,
        plt,
        regex_comparison,
        regex_pipeline,
    )


@app.cell
def _(COMPANIES_ALL, TERMS_SIMPLE, regex_pipeline):
    out_simple = regex_pipeline(TERMS_SIMPLE, COMPANIES_ALL)
    return


@app.cell
def _(OUT_DIR):
    import pandas as pd
    df_normed = pd.read_csv(OUT_DIR + '/simple-terms/regex/normed-words.csv', index_col=[0,1,2])
    df_unnormed = pd.read_csv(OUT_DIR + '/simple-terms/regex/unnormed.csv', index_col=[0,1,2])
    return df_normed, df_unnormed


@app.cell
def _(TERMS_SIMPLE, df_normed, plt, regex_comparison):
    regex_comparison(df_normed, terms = TERMS_SIMPLE, suptitle='Technology mentions - normalised by word count')
    plt.show()
    return


@app.cell
def _(COMPANIES_ALL, TERMS_SIMPLE, df_unnormed, plt, regex_comparison):
    regex_comparison(df_unnormed, terms=TERMS_SIMPLE, cos=COMPANIES_ALL, normed_by=None,
                    suptitle='Technology mentions - unnormalised')
    plt.show()
    return


@app.cell
def _(
    COMPANIES_ALL,
    OUT_DIR,
    TERMS_SIMPLE,
    df_unnormed,
    plt,
    regex_comparison,
):
    regex_comparison(df_unnormed, terms=TERMS_SIMPLE, cos=COMPANIES_ALL, rank=True,
                    suptitle='Technology mention rankings')
    plt.savefig(OUT_DIR + '/simple-terms/regex/plots/rankings.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
