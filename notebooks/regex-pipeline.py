import marimo

__generated_with = "0.23.16"
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

    from decarb import (
        TERMS_ALL, TERMS_SUMMARY, TERMS_SIMPLE,
        COMPANIES_ALL
    )
    from decarb.regex import regex_pipeline
    from decarb.output.plot import regex_plot_comparison

    return (
        COMPANIES_ALL,
        TERMS_ALL,
        TERMS_SIMPLE,
        TERMS_SUMMARY,
        mo,
        plt,
        regex_pipeline,
        regex_plot_comparison,
    )


@app.cell
def _(COMPANIES_ALL, TERMS_ALL, TERMS_SIMPLE, TERMS_SUMMARY, regex_pipeline):
    out = regex_pipeline(TERMS_ALL, COMPANIES_ALL, reprocess=False)
    out_sum = regex_pipeline(TERMS_SUMMARY, COMPANIES_ALL, reprocess=False)
    out_simple = regex_pipeline(TERMS_SIMPLE, COMPANIES_ALL, reprocess=True)
    return (out_simple,)


@app.cell
def _(TERMS_SIMPLE, out_simple, plt, regex_plot_comparison):
    regex_plot_comparison(out_simple['dfs'], terms = TERMS_SIMPLE, suptitle='Technology mentions - normalised by word count')
    plt.show()
    return


@app.cell
def _(COMPANIES_ALL, TERMS_SIMPLE, out_simple, plt, regex_plot_comparison):
    regex_plot_comparison(out_simple['dfs_unnormed'], terms=TERMS_SIMPLE, cos=COMPANIES_ALL, normed_by=None,
                    suptitle='Technology mentions - unnormalised')
    plt.show()
    return


@app.cell
def _(COMPANIES_ALL, TERMS_SIMPLE, out_simple, plt, regex_plot_comparison):
    regex_plot_comparison(out_simple['dfs_unnormed'], terms=TERMS_SIMPLE, cos=COMPANIES_ALL[:2], rank=True,
                    suptitle='Technology mention rankings')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
