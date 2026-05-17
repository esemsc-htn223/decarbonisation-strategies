import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

def stage_analyse(target_date: str, model: str = "gpt-5.1", max_workers: int = 5):
    """Stage 3: Run LLM analysis on all prompts."""
    

    load_dotenv()
    client = OpenAI()
    data_dir = get_data_dir(target_date)

    # Collect articles to analyse
    to_analyze = []
    for article_dir in sorted(data_dir.iterdir()):
        if not article_dir.is_dir():
            continue

        prompt_file = article_dir / "prompt.md"
        response_file = article_dir / "response.md"

        if not prompt_file.exists() or response_file.exists():
            continue

        meta_file = article_dir / "meta.json"
        with open(meta_file) as f:
            article = Article(**json.load(f))

        to_analyze.append((article_dir, article, prompt_file.read_text()))

    if not to_analyze:
        print("No articles to analyze.")
        return

    print(f"Analyzing {len(to_analyze)} articles with {max_workers} workers...")

    def analyze_one(item):
        article_dir, article, prompt = item
        response_file = article_dir / "response.md"
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "medium"},
                text={"verbosity": "medium"},
            )
            result = response.output_text
            with open(response_file, 'w') as f:
                f.write(result)
            return (article.item_id, article.title[:50], len(result), None)
        except Exception as e:
            return (article.item_id, article.title[:50], 0, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_one, item): item for item in to_analyze}
        for future in as_completed(futures):
            item_id, title, chars, error = future.result()
            if error:
                print(f"  {item_id}: {title}... Error: {error}")
            else:
                print(f"  {item_id}: {title}... Done ({chars} chars)")

    print(f"\nAnalysis complete. Results in {data_dir}")