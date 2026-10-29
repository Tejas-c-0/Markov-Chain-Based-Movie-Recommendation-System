# CineMark APSP — Group A11
## Reproducible Python recommendation-system project

CineMark is a movie recommendation research project for Applied Statistics & Stochastic Processes (APSP), CAT 2.

This repository is designed so a group member or instructor can inspect the data-cleaning steps, run the models, reproduce the evaluation, and understand what the results do—and do not—show.

## 1. Project contents

```text
cinemark_apsp_group_a11/
├── CineMark_APSP_Group_A11.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── movies.csv
│   └── ratings.json
└── outputs/                  # generated when notebook runs
```

Only the two files needed for modelling are included. User-profile files are intentionally excluded because they contain personal information and password fields.

## 2. Requirements

- Python 3.10 or newer recommended
- pip
- Jupyter Notebook or JupyterLab
- Internet is not required after Python packages are installed

## 3. Setup

Open a terminal in this project folder.

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

In Jupyter, open `CineMark_APSP_Group_A11.ipynb` and choose **Kernel → Restart Kernel and Run All Cells**. Run all cells from top to bottom.

## 4. Input data format

### `data/ratings.json`

This file uses JSON Lines (NDJSON): each non-empty line is a separate JSON object. A record resembles:

```json
{
  "_id": "example-user-id",
  "rated": {
    "tt5286444": ["1"],
    "tt4434004": ["0"],
    "submit": ["submit"]
  }
}
```

The notebook:
- uses `_id` only as an internal join key,
- reads movie feedback from `rated`,
- ignores non-movie keys such as `submit`,
- accepts only labels `-1`, `0`, and `1`,
- removes duplicate user–movie pairs,
- removes feedback whose movie ID is absent from the catalogue.

Interpretation for this experiment:
- `1` = positive feedback,
- `0` = neutral/unspecified (not automatically a dislike),
- `-1` = negative feedback.

### `data/movies.csv`

Expected columns include `movie_id`, `name`, `description`, `genre`, `language`, `cast`, `director`, and `writer`. The notebook checks the required columns and builds text features from available movie metadata.

## 5. Pipeline: how it works

### Step A — Inspect and clean
The notebook checks input files, normalizes IDs, parses labels, drops non-movie form keys, removes duplicate pairs, and matches feedback IDs against the movie catalogue.

### Step B — Split into training and test data
The supplied feedback does not contain a reliable viewing timestamp or event order. The notebook therefore uses a **per-user random holdout of one positive movie** for each user with at least two positive movies. Remaining positive movies are training examples.

This is a valid simple retrieval test, but it is not a chronological “predict the next movie” test. The notebook uses a fixed random seed for repeatability.

### Step C — Popularity baseline
Count positive training feedback per movie and rank movies by that count. Exclude movies already present in each user's training history.

### Step D — Content-based TF–IDF
Combine movie text fields into one document per movie. TF–IDF converts the text into numeric vectors. For each user, average vectors for their training-positive movies to form a profile. Rank unseen movies by similarity to that profile.

### Step E — Evaluate
For each test user, check whether their hidden positive movie appears in the top five recommendations.

- **Hit Rate@5:** share of users whose held-out movie is in the top five.
- **Recall@5:** equals Hit Rate@5 here because each user has one hidden positive target.
- **MRR@5:** rewards a hit more when it appears near rank 1.
- **NDCG@5:** rewards higher-ranked hits with a logarithmic discount.

All popularity counts and user profiles are built from training data only, reducing test leakage.

### Step F — Synthetic Markov Chain simulation
A Markov Chain estimates the next item from the current item. To learn and test it correctly, the model needs an ordered sequence of events. The supplied ratings file does not provide reliable timestamps or viewing order.

The notebook therefore generates **synthetic sequences** with a deliberately defined transition matrix. It compares a Markov recommender with a popularity baseline and measures how close the learned transition matrix is to the known synthetic matrix.

**Never report synthetic Markov scores as real CineMark results.** The simulation validates the basic idea under controlled conditions; it does not establish real-world performance.

## 6. Outputs

After a successful run, the notebook creates the following files in `outputs/`:

| File | Purpose |
|---|---|
| `data_quality_summary.csv` | Cleaning and dataset counts |
| `model_evaluation_results.csv` | Main model comparison metrics |
| `popularity_user_metrics.csv` | Per-user popularity evaluation |
| `content_user_metrics.csv` | Per-user content evaluation |
| `model_comparison.png` | Real-data model comparison chart |
| `synthetic_markov_simulation.csv` | Synthetic Markov simulation measurements |
| `synthetic_markov_simulation.png` | Synthetic-only chart |

Use the values generated by the notebook for the final report. Results can change if the seed, data, or preprocessing changes.

## 7. Privacy and safe handling

The supplied user-profile JSON/CSV files contain sensitive fields such as emails, dates of birth, and password values or password hashes. They are not needed for this model and are deliberately not copied into this project.

- Do not upload user-profile files to a public Git repository.
- Do not include personal user IDs in public screenshots or the report.
- Use pseudonymous IDs and only the minimum fields required for research.
- If the supplied profile file is real and its passwords were exposed, notify the data owner and rotate any affected credentials.

## 8. Limitations to explain during the review

1. **No real sequential evaluation:** timestamps or reliable viewing order are missing.
2. **One holdout split:** results may vary with a different split; repeated splits or cross-validation would provide a stronger comparison.
3. **One hidden positive target per test user:** Hit Rate@5 and Recall@5 are equal under this setup.
4. **Sparse feedback:** many catalogue movies may have little or no feedback.
5. **Label meaning:** the notebook treats `0` as neutral/unspecified, not as a dislike.
6. **Metadata quality:** missing or noisy descriptions and cast/genre fields can weaken TF–IDF recommendations.
7. **Catalogue rating is not user feedback:** the movie metadata rating is not used as the recommendation target.

## 9. What to say to the instructor

> “CineMark recommends movies using the feedback labels in our dataset. We cleaned the movie IDs and labels, then compared a popularity baseline with a content-based model using movie descriptions and metadata. We evaluated both with a per-user holdout and top-five ranking metrics. The dataset has no reliable viewing timestamps, so we did not claim a real-data Markov result. Instead, we tested the Markov logic on synthetic sequences, where the event order is known. The next step is to collect timestamped, privacy-safe user events and evaluate Markov on future interactions.”

## 10. Troubleshooting

- **FileNotFoundError:** confirm `data/movies.csv` and `data/ratings.json` exist beside the notebook in the `data/` directory.
- **Missing package:** activate the virtual environment and run `pip install -r requirements.txt`.
- **Notebook uses the wrong Python:** in Jupyter, select the kernel from the environment where requirements were installed.
- **Different results:** confirm the same data files and `RANDOM_SEED = 42` are being used.
- **Large output or slow run:** the content similarity computation is manageable for the supplied 2,850-movie catalogue. Avoid adding unnecessary user-profile columns.

## 11. Reproducibility checklist

- [ ] Use only `movies.csv` and `ratings.json` as model inputs.
- [ ] Install packages from `requirements.txt`.
- [ ] Restart the kernel and run every cell in order.
- [ ] Confirm both real-data models have metric rows.
- [ ] Keep synthetic Markov results labelled as synthetic.
- [ ] Attach the generated CSVs/charts or copy their measured values into the report.
