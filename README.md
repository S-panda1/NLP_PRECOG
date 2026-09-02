# NLP_PRECOG: The Library of Babel

Can you tell Human-written text apart from AI-generated text — even when the AI is
explicitly mimicking a specific author's style? This project builds a labeled dataset
from two Project Gutenberg authors, engineers linguistic fingerprint features, trains
three tiers of detectors, interprets what they pick up on, and finally tries to evolve
an AI paragraph that fools the best detector.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

Create a `.env` file in this folder with:

```
GEMINI_API_KEY="your_key_here"
```

Optionally, add `GEMINI_API_KEY2` and `GEMINI_API_KEY3` from **separate** Google
accounts/projects. `task0.py` and `task4.py` round-robin across every key that's
present, each with its own independent daily/per-minute quota — so 3 keys roughly
triples total throughput. Keys from the same account/project share one quota pool and
won't help; see the note below.

Run everything from this folder (`NLP_PRECOG/`), not from `scripts/`:

```powershell
python .\scripts\task0.py
python .\scripts\task1.py
python .\scripts\task2.py
python .\scripts\task3.py
python .\scripts\task4.py
```

Each script is idempotent and reads/writes under `results/<task>/`, resolved from the
script's own location — so it works regardless of your current working directory.

## Pipeline

| Script | Produces |
|---|---|
| `task0.py` | `results/task0/dataset_raw.csv` — Human paragraphs (Melville, Chambers) + Gemini-generated AI_Standard and AI_Mimic paragraphs on the same topics. 500 paragraphs/class (250 per author). |
| `task1.py` | `results/task1/dataset_with_features.csv` plus lexical/statistical distinctness analysis (see below). |
| `task2.py` | Three detectors — Tier A (XGBoost on hand-crafted features), Tier B (FFNN on averaged spaCy word vectors), Tier C (RoBERTa + LoRA) — with classification reports for each. |
| `task3.py` | Integrated Gradients saliency report + error analysis on Human-labeled-as-AI misclassifications. |
| `task4.py` | Genetic-algorithm evasion log + the personal-writing test. |

## Task 1: proving the classes are distinct

Per-paragraph TTR/Hapax are noisy at ~150 words, so in addition to the per-row features
(used for training in Task 2), `task1.py` also computes:

- `lexical_richness_5000word_sample.csv` — TTR and Hapax Legomena ratio computed on a
  fixed 5,000-word sample per class, as specified.
- `feature_summary_by_class.csv` — mean/std of every feature, grouped by class.
- `feature_anova_pvalues.csv` — one-way ANOVA per feature across the three classes,
  to numerically back the claim that classes are distinguishable.
- `feature_distinctness_boxplots.png` — visual counterpart to the ANOVA table.

## Task 2: negative results

If any tier's accuracy is weak (e.g. near-baseline on an imbalanced split), don't just
report the number — check the per-class support in `tier_*_report.txt` first. A high
"accuracy" with very low recall on the minority class usually means the model is mostly
predicting the majority class. Document the imbalance and what it implies, rather than
treating the raw accuracy as a success metric.

## Task 4: the personal test

To run the personal-writing test, add a plain-text file at `data/my_essay.txt` (create
the `data/` file yourself — it isn't committed since it's personal writing), with
paragraphs separated by a blank line, then re-run `task4.py`. It will:

1. Score each paragraph's Human-probability under Tier C.
2. If the essay reads as "AI," suggest manual edits (vary rhythm, cut stock AI-vocabulary)
   — you make the edits yourself and re-run to see if the score improves.
3. If the essay reads as "Human," it also generates one Gemini-assisted "make this sound
   like an LLM" rewrite of your first paragraph as a demonstration — try doing the same
   rewrite by hand for the actual exercise.

Results are written to `results/task4/personal_test.txt`.

## Notes

- Two authors are used for the Human class: Herman Melville (*Moby-Dick*) and Robert W.
  Chambers (*The King in Yellow*), both public-domain texts in `data/`.
- `PARAGRAPHS_PER_AUTHOR` in `task0.py` controls dataset size (default 250 → 500/class
  total). Generating the full dataset makes ~1000 Gemini API calls; expect it to take a
  while and to consume API quota.
- `task0.py`/`task4.py` default to `gemini-3.5-flash-lite`, not `gemini-3.5-flash` — the
  latter has only a 20-requests/day free-tier quota on new projects, which isn't enough
  for bulk generation. If every configured key hits its daily quota, the script stops
  early and saves whatever it already generated; re-run after the quota resets (midnight
  Pacific time) to top up the dataset, or adjust `MODEL_NAME` if you have billing enabled
  and want a stronger model.
- **`task0.py` is resumable.** If it gets cut short by a daily-quota exhaustion, re-running
  it reuses the extracted topics (`extracted_topics.txt`) and every AI paragraph already
  saved in `dataset_raw.csv`, and only generates the remaining shortfall per
  author/class — it won't re-spend quota regenerating paragraphs you already have.
- Rate limits are enforced per Google Cloud **project**, not per API key (per Google's
  docs). `GEMINI_API_KEY2`/`GEMINI_API_KEY3` only help if they belong to different
  projects — verify on the [AI Studio API keys page](https://aistudio.google.com/apikey),
  which shows each key's project.
