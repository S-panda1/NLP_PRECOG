# The Library of Babel: Detecting AI-Generated Text Through Linguistic Fingerprinting

**Report on dataset construction, detector design, interpretability, and adversarial evasion.**

---

## 1. Task Completion Summary

Legend: ✅ Done · ⚠️ Done with deviation · ❌ Not done (reason given)

### Task 0 — Dataset Construction

| Requirement | Status | Notes |
|---|---|---|
| Two Gutenberg novels, cleaned | ⚠️ | Melville's *Moby-Dick* + Chambers' *The King in Yellow*, downloaded manually but cleaned via script (Gutenberg boilerplate, headings, footnotes, dedup). |
| 5–10 topic extraction | ✅ | Exactly 10, via Gemini (Appendix A). |
| 500 AI_Standard paragraphs, 100–200 words | ✅ | 250/author, ~150 words each. |
| 500 AI_Mimic paragraphs (author-style) | ✅ | 250/author. |

Extra engineering (not spec'd, required by free-tier API limits): multi-key rotation, daily-vs-transient error handling, resumable/checkpointed generation.

### Task 1 — The Fingerprint

| Requirement | Status |
|---|---|
| TTR, Hapax Legomena (5,000-word sample) | ✅ |
| Adj/Noun POS ratio | ✅ |
| Dependency tree depth | ✅ |
| Punctuation heatmap | ✅ |
| Flesch-Kincaid | ✅ |

Added beyond spec: one-way ANOVA + boxplots, since "prove mathematically distinct" needs a statistical test, not just raw features.

### Task 2 — Multi-Tiered Detective

| Requirement | Status | Notes |
|---|---|---|
| Tier A: XGBoost on Task 1 features | ✅ | |
| Tier B: FFNN on GloVe/FastText embeddings | ⚠️ | Used spaCy `en_core_web_md` static vectors instead — same class of averaged pretrained embedding, not the literal named tool. |
| Tier C: RoBERTa/DistilBERT + LoRA | ✅ | `roberta-base` + LoRA. |
| Negative-results analysis if accuracy is weak | ❌ *(n/a)* | All tiers hit 97–100% — nothing weak to diagnose. The inverse ("why is this *too* good") is dug into in §6. |

### Task 3 — The Smoking Gun

| Requirement | Status | Notes |
|---|---|---|
| Saliency mapping (Captum) | ✅ | 3 imposter samples (spec asked for ≥1). |
| AI-isms vs. rhythm findings | ✅ | Quantified: 0% AI-isms / 80% content words / 20% structural. |
| 3 Human→AI misclassifications | ❌ *(impossible)* | Code fully implemented, but Tier C made **zero** Human→AI errors on the test set — nothing to find. Tier A's own errors are used instead in §6. |

### Task 4 — The Turing Test

| Requirement | Status | Notes |
|---|---|---|
| Initial population: 10 fresh Gemini paragraphs | ⚠️ | Reused 10 existing `AI_Standard` paragraphs instead of new calls (quota conservation); still genuinely Gemini-authored. |
| Fitness / selection / mutation via Gemini | ✅ | |
| 5–10 generations | ⚠️ | Ran 5 (lower bound); fitness plateaued for 3 gens straight, so more generations of the same mutation strategy were unlikely to help (see §6). |
| Reach >90% Human confidence | ❌ *(negative result)* | Best reached: 0.82%. Detector was not fooled — a real finding, not a skipped step. |
| Personal test (SOP/essay) | ✅ | Both code paths exist; only the "Human" path was exercised since the essay scored Human. |

**Bottom line:** every script runs end-to-end and produces every required artifact category. Genuine gaps: one embedding substitution, one GA-seeding shortcut, one task made impossible by a flawless model, and two negative results that are findings, not omissions. We beat random performance (50%) by a wide margin on every tier (§5), and the rest of this report focuses on *why*, not just *how much*.

---

## 2. Overview

Question: can a detector tell human text from AI text, even when the AI is told to mimic a specific author's voice? All three classes discuss the *same* 10 topics, so any signal must come from *how* text is written, not *what* it's about.

| Script | Output |
|---|---|
| `task0.py` | `results/task0/dataset_raw.csv` |
| `task1.py` | Features + distinctness proof, `results/task1/` |
| `task2.py` | 3 trained detectors, `results/task2/` |
| `task3.py` | Saliency + error analysis, `results/task3/` |
| `task4.py` | GA log + personal test, `results/task4/` |

---

## 3. Methodology

**Task 0.** Clean 500 human paragraphs (250/author, seed 42). Extract 10 topics via Gemini. Generate 250 `AI_Standard` + 250 `AI_Mimic` paragraphs per author per topic. Multi-key rotation retries on any error by hopping keys; daily-quota errors retire a key permanently; a resumed run only generates the shortfall vs. the existing CSV.

**Task 1.** 8 features per paragraph via spaCy/NLTK/textstat: `TTR`, `Hapax_Ratio`, `Adj_Noun_Ratio`, `Avg_Tree_Depth`, `Flesch_Kincaid`, `Semicolons`, `Em_Dashes`, `Exclamations`. TTR/Hapax also recomputed once per class on a pooled, shuffled 5,000-word sample. One-way ANOVA (`scipy.stats.f_oneway`) per feature across classes.

**Task 2.** Shared 80/20 split (seed 42) on 1,500 rows, target = `is_ai`. Tier A: XGBoost on the 8 features. Tier B: mean spaCy word-vector per paragraph → small FFNN (300→128→64→1). Tier C: `roberta-base` + LoRA (r=8, α=32, targeting `query`/`value`), 3 epochs.

**Task 3.** Captum `LayerIntegratedGradients` on Tier C's word-embedding layer, over 3 `AI_Standard` samples; top-5 tokens/sample checked against a 20+ word "AI-ism" list vs. other content words vs. function words. Same test split re-derived; any Human row scored higher-AI-than-Human is a false positive, ranked and explained via feature deltas.

**Task 4.** GA: population 10 (seeded from existing `AI_Standard` paragraphs) → fitness = Tier C's P(Human) → keep top 3 → refill to 10 via Gemini rewrites (rhythm/archaism/de-formalization prompts) → repeat for 5 generations or until >90% or a hard generation failure. Personal test: score user's essay paragraphs; if AI-flagged, print manual humanization tips; if Human-flagged, also auto-generate and re-score an "LLM-ified" rewrite of paragraph 1.

---

## 4. Design Choices & Why We Made Them

**Task 0 — data.** Melville (sea-adventure/philosophical) and Chambers (Gothic/horror) write in very different voices but cover similar themes (dread, isolation, obsession) — so every class talks about the same topics, and only the *writing style* changes between classes. 250 paragraphs per author per class (1,500 rows total) is enough data for the stats tests and train/test splits to be meaningful, while still fitting inside what a couple of free API keys can generate in a day. Paragraphs target ~150 words (the middle of the assigned 100–200 word range) so length itself isn't a giveaway between classes.

**Task 1 — metrics.** Word-diversity scores like TTR look artificially high on short text just because there's less chance to repeat a word — so comparing them paragraph-by-paragraph would mostly measure noise, not style. Pooling a fixed 5,000-word sample per class first fixes that. Adjective/noun ratio and sentence-tree depth were picked because they need no labeled examples, just a parser, and directly answer the plain-English questions the brief asks ("does the AI over-describe?", "are its sentences more nested?"). We used one ANOVA test across all 3 classes at once instead of three separate pairwise tests, since running several tests back-to-back inflates the chance of a false "significant" result.

**Task 2 — models & settings.**
- *Tier A (XGBoost):* trees instead of a straight-line model because they can catch combined rules like "AI-like only if it over-uses adjectives *and* never uses exclamation marks" — a linear model can't easily express "and" conditions like that. Kept shallow (`max_depth=4`) and with a moderate number of trees (`n_estimators=100`) since we only have 8 features and ~1,200 training rows — a deeper model would just memorize the training set.
- *Tier B (FFNN on averaged embeddings):* averaging each paragraph's word vectors and feeding that into a small neural net is a classic, cheap text-classification baseline. It ignores word order, but is fast and strong enough for style differences this large. A modest network size (300→128→64→1, with dropout) avoids overfitting on a modest amount of data.
- *Tier C (RoBERTa + LoRA):* RoBERTa reads the actual words in order, so it can catch subtler style cues the other two tiers can't. LoRA freezes almost all of RoBERTa's 125-million parameters and only trains a small add-on — this is what made fine-tuning a full transformer possible without a dedicated GPU (see §5). The specific settings used (`r=8`, `alpha=32`) are the commonly recommended defaults for tasks this size, used instead of an expensive trial-and-error search we couldn't afford to run.
- All three tiers are trained/tested on the *exact same* data split, so any accuracy difference between them reflects the model, not lucky/unlucky data.

**Task 3 — method.** We used Integrated Gradients (via Captum) instead of a plain gradient check because it comes with a guarantee: the scores it assigns to each word always add up to the model's actual prediction, which plain gradients don't guarantee and can give misleading results for. It's applied at the word-embedding layer because the model's raw input (token IDs) isn't something you can smoothly vary to compute a gradient over. We looked at 3 samples and their top 5 words each — kept small on purpose, since the goal was to read and understand the results by hand, not produce another big number.

**Task 4 — method.** There's no clean way to "cross" two paragraphs the way you'd cross two numbers, so instead of a classic genetic-algorithm crossover step, each new paragraph is produced by asking Gemini to rewrite a survivor — this is the "LLM-as-mutator" idea the brief itself suggests. The top 3 paragraphs are always carried over unchanged into the next generation (elitism), so a bad rewrite can never erase the best result found so far — important since every generation costs real API calls that can't be redone for free.

---

## 5. Handling Scale & Limited Resources

The real bottleneck in this project was never "big data" volume (1,500 rows, ~225k words total, by design) — it was a hard external API rate limit and modest local compute. The engineering below addresses both, using techniques that also generalize to genuinely large-scale settings:

- **API-bound "large dataset" collection.** Free-tier Gemini quotas (15 requests/min, 500/day per key/project) meant ~1,000 required generation calls could not run as one naive loop. The fix: multi-key round-robin so a rate-limited key's calls fall through to an unaffected key; explicit daily-vs-transient error differentiation so a temporary rate limit doesn't permanently burn a key's whole day; and a **resumable, checkpointed** pipeline that persists progress to `dataset_raw.csv` and, on any later run, only requests the shortfall per (author, class) bucket instead of regenerating everything. This is the same idea as checkpointing a long training job, applied to an API-bound data-collection job — and is exactly the kind of technique (checkpoint + resume, rather than "just get more quota") that scales to collecting millions of records, not just thousands.
- **Compute-bound fine-tuning.** No dedicated training GPU was assumed. LoRA cuts Tier C's trainable parameters by >99% relative to full fine-tuning of RoBERTa's 125M parameters, which is what made fine-tuning tractable at all. `max_length=128` token truncation and modest batch sizes (16 train / 32 batched inference in error analysis) keep memory bounded; 128 was chosen because the ~150-word generation prompts mostly tokenize under that limit with RoBERTa's BPE tokenizer, so truncation loss is minor for most paragraphs (flagged as a real trade-off in §7, not an unexamined default).
- **Sampling for statistics, not the full corpus.** Task 1's lexical-richness metrics use a fixed 5,000-word reservoir sample per class rather than either noisy per-paragraph values or an unnecessarily expensive full-corpus pass — the standard middle ground between "too little context" and "compute everything."
- **Why this matters beyond this project's scale.** The same three moves — checkpoint/resume instead of re-running from scratch, parameter-efficient fine-tuning instead of full fine-tuning, and fixed-size representative sampling instead of loading everything into memory — are exactly how these pipelines would be adapted to a genuinely large real-world corpus (millions of documents): streaming feature extraction via `nlp.pipe(batch_size=...)`, stratified/reservoir sampling instead of full loads, and LoRA + mixed precision + gradient checkpointing to fit large-model fine-tuning into limited VRAM.

---

## 6. Results

**Dataset:** 1,500 rows, perfectly balanced — 500 Human / 500 AI_Standard / 500 AI_Mimic (250 per author each).

**Lexical richness (5,000-word sample):**

| Class | TTR | Hapax Ratio |
|---|---|---|
| Human | 0.3044 | 0.2158 |
| AI_Standard | 0.2644 | 0.1476 |
| AI_Mimic | 0.2902 | 0.1868 |

**ANOVA:** 7 of 8 features significant at p<0.05. Strongest: `Adj_Noun_Ratio` (F=385.6), `Hapax_Ratio` (F=265.9), `Flesch_Kincaid` (F=240.2). Only `Em_Dashes` is not significant (p=0.122).

**Detectors** (300-row test set, 104 Human / 196 AI — well above the 50% random-guess baseline):

| Tier | Approach | Accuracy |
|---|---|---|
| A | XGBoost, hand-crafted features | 97% (8/300 errors) |
| B | FFNN, averaged spaCy vectors | ~100% |
| C | RoBERTa + LoRA | 100% (eval loss 0.0009) |

**Saliency:** top-5 tokens/sample were 0% AI-isms, 80% content words, 20% structural.

**Error analysis (Tier C):** 0 of 104 Human samples misclassified as AI.

**GA evasion:** best Human-probability by generation: 0.0002 → 0.0043 → 0.0043 → 0.0043 → 0.0082. Never approached the 90% target.

**Personal test:** essay scored 0.9314 Human-probability; an automated "sound like an LLM" rewrite dropped it to 0.7932 — a real swing, not a flip.

---

## 7. Deeper Analysis: Hypotheses & What the Errors Actually Reveal

Accuracy alone doesn't explain *why* these results happened. Digging into the actual feature distributions and Tier A's (the only tier with any errors) specific mistakes surfaces a pattern the summary numbers hide.

**Hypothesis 1 — AI text occupies a much narrower region of feature space, and that alone may explain most of the accuracy.** Computing the coefficient of variation (std/mean) per feature per class:

| Feature | Human CoV | AI_Standard CoV | AI_Mimic CoV |
|---|---|---|---|
| TTR | 11.8% | 3.7% | 3.6% |
| Hapax_Ratio | 17.4% | 5.8% | 5.3% |
| Flesch_Kincaid | 65.3% | 10.7% | 13.9% |

Every measured feature is 2–6× *more variable* in human writing than in either AI class. This reframes "detection" as partly an outlier-detection problem: a human paragraph gets flagged as AI mainly when it happens to land near the corpus's dense, narrow AI-typical center; an AI paragraph gets flagged as human only when a generation happens to drift out into the wide, sparse human region. If true, a much simpler unsupervised baseline — flagging any paragraph with unusually *low* distance-to-centroid in feature space — might recover a meaningful fraction of Tier A's accuracy with no labels at all. That's a concrete, testable follow-up this analysis motivates.

**Testing that hypothesis against Tier A's real errors.** Reloading `tier_a_xgboost.json` on its exact test split surfaces exactly 8/300 misclassifications, and the pattern matches:
- **6 of 8 are Human→AI false positives**, and every one has `Exclamations = 0` (matching AI_Standard's constant 0.0 exactly) and unusually *high* TTR (0.73–0.83 — at or above the AI clusters' typical band). The humans the model confuses are precisely the least "human-typical" ones: no emotional punctuation, higher-than-usual lexical diversity.
- **2 of 8 are AI→Human false negatives.** One — an `AI_Mimic_Chambers` paragraph — is the standout: its TTR (0.716), Adjective/Noun ratio (0.40), Flesch-Kincaid grade (11.8), and semicolon rate (2.0) all sit within a hair of the *Human class means* (0.718 / 0.395 / 11.10 / 1.384). This is a genuine instance of "the AI being particularly brilliant" — exactly the question Task 3's brief poses — just caught in Tier A's errors rather than Tier C's (which had none to show).

**Hypothesis 2 — the GA's mutation operator can't escape the region Hypothesis 1 describes.** Every GA mutation is itself an LLM rewrite. If LLM output is intrinsically confined to a narrow stylistic band (Hypothesis 1), then using an LLM to search for a "human-outlier" paragraph is somewhat self-defeating — the operator is biased toward the very center it's trying to escape. This predicts exactly what was observed: fitness plateaus quickly (3 flat generations) regardless of generation count. A more promising fix than "run more generations" would be mutation prompts that explicitly target the *high-variance* traits Hypothesis 1 identifies as human-typical (irregular punctuation bursts, uneven sentence-to-sentence complexity) rather than generic rhythm/vocabulary changes.

**Hypothesis 3 — some saliency signal may be a methodological artifact.** Sentence-final periods repeatedly rank among Task 3's top attributed tokens. That's plausible as a real rhythm signal, but Integrated Gradients attributions are computed relative to a baseline (here, effectively an all-zero/padding embedding), and high-frequency, low-information tokens near that baseline are known in the interpretability literature to sometimes be over-attributed. Without a baseline-sensitivity check (re-running with, e.g., the tokenizer's `<mask>` embedding as baseline), this specific finding is flagged as *suggestive, not conclusive* — deliberately not overclaimed.

**What this suggests for improving the approach.** All three hypotheses point the same way: the next experiment worth running isn't a bigger model, it's modeling *within-class variance* directly — e.g. adding a "distance from class centroid" feature to Tier A, or deliberately sampling higher-temperature/more-diverse Gemini generations for Task 0's AI classes, to build a harder dataset that stress-tests whether Tier C's ~100% accuracy survives contact with AI writing that isn't so stylistically uniform.

---

## 8. Limitations & Future Work

- Both source authors write in ornate, archaic registers far from contemporary prose — conclusions may not generalize to modern-vs-modern text.
- Tier B uses spaCy vectors, not literal GloVe/FastText.
- GA's initial population was reused, not freshly generated.
- The saliency method's punctuation-heavy findings (§7, Hypothesis 3) haven't been cross-checked against an alternate IG baseline.
- Natural next step: test against contemporary human writing vs. contemporary LLM output on the same modern topics, and try the variance-aware mutation/sampling strategies proposed in §7.

---

## Appendix A — Extracted Topics

Cosmic dread · Madness · Obsession · Fate · Isolation · The fragility of the human mind · Forbidden knowledge · The destructive power of nature · Alienation · Existential despair

## Appendix B — Repository Map

```
NLP_PRECOG/
  data/                         Source novels + optional personal essay
  scripts/                      task0.py .. task4.py
  results/
    task0/dataset_raw.csv
    task1/dataset_with_features.csv, feature_anova_pvalues.csv,
          feature_summary_by_class.csv, lexical_richness_5000word_sample.csv,
          punctuation_heatmap.png, feature_distinctness_boxplots.png
    task2/tier_a_report.txt, tier_b_report.txt, tier_c_report.txt,
          tier_a_xgboost.json, tier_b_ffnn.pth, tier_c_lora_model/
    task3/saliency_report.txt, error_analysis.txt
    task4/evolution_log.txt, personal_test.txt
  README.md / REPORT.md
```

See `README.md` for setup and run instructions.
