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
        Term,
        TermCollection,
        mo,
        plot_comparison,
        plt,
        run_pipeline,
    )


@app.cell
def _(Term, TermCollection):
    TERMS_SIMPLE = TermCollection(
        [
            Term('Solar', r'solar|\bPV\b|\bP\.V\.\b|photovoltaic|\bCSP\b|\bC\.S\.P\.\b', 'gold'),
            Term('Wind', r'\bwind\b', 'powderblue'),
            Term('Hydrogen', r'hydrogen|h2', 'turquoise'),
            Term('CCS', r'(carbon\s+capture(\s+and\s+(storage|sequestration|utili[sz]ation))|ccs)', 'dimgray'),
            Term('Biofuel', 'biofuel|biodiesel|bioethanol', 'tab:brown', dash=True, hatch = '....', edgecolour='peru'),
            Term('Biogas', r'biomethane|renewable\s+natural\s+gas|\bRNG\b|\bR\.N\.G\.\b|biogas', 'tab:brown', dash=True, hatch = '....', edgecolour='lightgrey'),
            Term('Geothermal', r'geothermal|ground\.source|ground\s+source', 'tab:red'),
            Term('Tidal', r'tidal|wave', 'steelblue'),
            Term('Storage', r'\b(batter(y|ies)|BESS|B\.E\.S\.S\.|compressed\s+air|CAES|C\.A\.E\.S\.)\b', 'pink'),
            Term('EVs', r'\b(electric\s+vehicle|EV|E\.V\.)s?\b', 'lightgrey'),
            Term('Nature', r'\b(nature.based|nature\s+based|(re|a)forestation)\b', 'darkgreen'),
            Term('Nuclear (conventional)', r'nuclear\s+(power|plants?|energy|reactor|fleet|fission)', 'tab:olive'),
            Term('Nuclear (SMR)', r'\b((small|advanced)\s+modular\s+reactor)\b', 'tab:olive', dash=True, hatch = '....', edgecolour='tab:grey'),
            Term('Nuclear fusion', r'nuclear\s+fusion|fusion\s+(energy|power|reactor|plant|research)', 'tab:olive', dash=True, hatch = '....', edgecolour='gold'),
        ],
        folder_name = 'simple-terms'
    )
    return (TERMS_SIMPLE,)


@app.cell
def _(COS, TERMS_SIMPLE, run_pipeline):
    out_simple = run_pipeline(TERMS_SIMPLE, COS, reprocess=True, plot=True)

    return


@app.cell
def _(COS, TERMS_ALL, TERMS_SIMPLE, TERMS_SUMMARY, run_pipeline):
    out = run_pipeline(TERMS_ALL, COS, reprocess=True, plot=True)
    out_sum = run_pipeline(TERMS_SUMMARY, COS, reprocess=True, plot=True)
    out_simple = run_pipeline(TERMS_SIMPLE, COS, reprocess=True, plot=True)
    return (out,)


@app.cell
def _(out, plot_comparison, plt):
    plot_comparison(out['dfs'], suptitle='Technology mentions - normalised by word count')
    plt.show()
    return


@app.cell
def _(COS, TERMS_ALL, out, plot_comparison, plt):
    plot_comparison(out['dfs_unnormed'], terms=TERMS_ALL, cos=COS, normed_by=None,
                    suptitle='Technology mentions - unnormalised')
    plt.show()
    return


@app.cell
def _(COS, TERMS_ALL, out, plot_comparison, plt):
    plot_comparison(out['dfs_unnormed'], terms=TERMS_ALL, cos=COS, rank=True,
                    suptitle='Technology mention rankings')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
