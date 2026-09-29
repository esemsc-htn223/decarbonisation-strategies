import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo=marimo):
    mo.md(r"""
    # BERT testing
    """)
    return


@app.cell
def _():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    import datasets
    from tqdm.auto import tqdm
 
    dataset_name = "climatebert/climate_detection"
    tokenizer_name = "climatebert/distilroberta-base-climate-detector"
    model_name = "climatebert/renewable"
 
    # If you want to use your own data, simply load them as 🤗 Datasets dataset, see https://huggingface.co/docs/datasets/loading
    dataset = datasets.load_dataset(dataset_name, split="test")
 
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len=512)
 
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
 
    # See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
    for out in tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True)):
       print(out)
    return (dataset,)


@app.cell
def _(dataset):
    dataset._data
    return


@app.cell
def _():
    import pandas as pd
    _df = pd.read_csv('out/simple-terms/regex/mentions-unnormed/bp-an.csv').melt(
        id_vars=['year', 'filename'], var_name='term', value_name='count'
    )
    _df_meta = _df.loc[_df['term'].isin(['n_words', 'n_pages'])]
    _df = _df.loc[~_df['term'].isin(['n_words', 'n_pages'])]
    _df
    return (pd,)


@app.cell
def _(pd):
    pd.read_csv('out/simple-terms/regex/mentions-unnormed/bp-an.csv')
    return


if __name__ == "__main__":
    app.run()
