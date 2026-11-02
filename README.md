# CineMark: Movie Recommendation Benchmark

An APSP project comparing movie recommenders on the supplied movie catalogue and feedback data. The notebook runs the complete benchmark and displays its result tables and graphs.

## Run the project

Open a terminal in this folder and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Open `Markov chain CineMark.ipynb` in Jupyter or VS Code, select the project Python environment, and run all cells. The notebook runs `run_benchmark.py`, shows the result tables, and displays both comparison charts.

To run from a terminal instead:

```powershell
python run_benchmark.py
```

To try different train/test split seeds:

```powershell
python run_benchmark.py --seeds 3 11 42 87 101
```

Edit the seed list in the notebook's run cell to use different seeds there.

## What the benchmark does

1. Cleans movie IDs and feedback labels, removes duplicate and unmatched records, and reports data quality.
2. Splits each eligible user's positive feedback into training data and one held-out test movie. Previously rated movies are excluded from recommendations.
3. Prepares movie text with TF-IDF and numeric catalogue features with median imputation and standard scaling.
4. Compares popularity, content-based, item-item collaborative, and hybrid recommenders across five seeded splits.
5. Measures Hit Rate@5, Recall@5, MRR@5, and NDCG@5.
6. Separately trains and tests a Markov chain on generated, ordered sequences and compares it with popularity.

## Current results

Across the default five seeds (731 eligible test users per split), the hybrid recommender ranks first by mean NDCG@5 (0.0190), narrowly ahead of popularity (0.0187). Popularity has a slightly higher mean Hit Rate@5 (3.34% vs. 3.20%). These are descriptive results, not evidence that one model is universally better.

The synthetic Markov experiment reports 75.55% mean Hit Rate@5, compared with 21.39% for popularity, on generated next-event data. **This is not a result on real CineMark user histories.**

## Generated files

The benchmark saves only the current results under `outputs/`:

- `benchmark_data_quality.csv` — cleaning summary
- `seed_model_metrics.csv` — metrics for each model and seed
- `model_seed_summary.csv` — mean, variation, and ranking across seeds
- `recommendation_preview.csv` — sample recommendations with masked user labels
- `multi_seed_model_comparison.png` — real-feedback recommender graph
- `synthetic_markov_seed_metrics.csv` — synthetic Markov metrics by seed
- `synthetic_markov_summary.csv` — synthetic Markov summary
- `synthetic_markov_seed_comparison.png` — synthetic sequence graph

Old output files not listed above are not used by the current benchmark.

## Data limitation and privacy

The feedback data has no reliable timestamps or viewing order. Therefore, a Markov chain cannot be trained or evaluated on real user viewing sequences from these files. The Markov experiment is explicitly synthetic; real next-movie prediction would require authorized, privacy-safe, timestamped interaction data.

The ratings file contains user identifiers. Do not publish it or expose identifiers in reports, screenshots, or recommendation examples without data-owner approval.
