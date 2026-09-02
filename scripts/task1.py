import random
from pathlib import Path

import pandas as pd
import spacy
import textstat
import nltk
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "task1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

nltk.download('punkt', quiet=True)
tqdm.pandas()

print("--- Task 1: Linguistic Fingerprinting ---")
print("Loading Spacy model...")
nlp = spacy.load('en_core_web_md')
df_all = pd.read_csv(PROJECT_ROOT / "results" / "task0" / "dataset_raw.csv")

def calc_tree_depth(token):
    if not list(token.children): return 1
    return 1 + max(calc_tree_depth(child) for child in token.children)

def extract_features(text):
    if not isinstance(text, str): return pd.Series([0]*8)
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    
    ttr = len(set(words)) / len(words) if words else 0
    fdist = FreqDist(words)
    hapax = len([w for w, c in fdist.items() if c == 1]) / len(words) if words else 0
    
    pos_counts = doc.count_by(spacy.attrs.POS)
    adj_count = pos_counts.get(doc.vocab.strings.add('ADJ'), 0)
    noun_count = pos_counts.get(doc.vocab.strings.add('NOUN'), 0)
    adj_noun_ratio = adj_count / noun_count if noun_count > 0 else 0
    
    tree_depths = [calc_tree_depth(sent.root) for sent in doc.sents]
    avg_tree_depth = sum(tree_depths) / len(tree_depths) if tree_depths else 0
    
    flesch = textstat.flesch_kincaid_grade(text)
    punct = {'semicolon': text.count(';'), 'em_dash': text.count('—') + text.count('--'), 'exclamation': text.count('!')}
    
    return pd.Series([ttr, hapax, adj_noun_ratio, avg_tree_depth, flesch, punct['semicolon'], punct['em_dash'], punct['exclamation']])

feature_cols = ['TTR', 'Hapax_Ratio', 'Adj_Noun_Ratio', 'Avg_Tree_Depth', 'Flesch_Kincaid', 'Semicolons', 'Em_Dashes', 'Exclamations']

print("Extracting features (this may take a while)...")
df_all[feature_cols] = df_all['text'].progress_apply(extract_features)

# Save the dataset with features
df_all.to_csv(RESULTS_DIR / "dataset_with_features.csv", index=False)
# Plot and save Punctuation Heatmap
punct_df = df_all.groupby('class_id')[['Semicolons', 'Em_Dashes', 'Exclamations']].mean()
plt.figure(figsize=(8, 4))
sns.heatmap(punct_df, annot=True, cmap="YlGnBu", yticklabels=['Human', 'AI Standard', 'AI Mimic'])
plt.title("Punctuation Density by Text Class")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "punctuation_heatmap.png")
plt.close()

CLASS_NAMES = {0: 'Human', 1: 'AI_Standard', 2: 'AI_Mimic'}
df_all['class_name'] = df_all['class_id'].map(CLASS_NAMES)

# --- Lexical richness on a fixed 5,000-word sample per class (as specified) ---
# Per-row TTR/Hapax above are noisy at paragraph length (~150 words); this recomputes
# both metrics on a much larger, class-level sample for a statistically meaningful figure.
def lexical_richness_on_sample(texts, sample_size=5000, seed=42):
    rng = random.Random(seed)
    texts_list = list(texts)
    rng.shuffle(texts_list)
    words = []
    for t in texts_list:
        if not isinstance(t, str):
            continue
        words.extend(w.lower() for w in t.split() if w.isalpha())
        if len(words) >= sample_size:
            break
    sample = words[:sample_size]
    if not sample:
        return {'TTR_5000': 0.0, 'Hapax_Ratio_5000': 0.0, 'sample_size': 0}
    fdist = FreqDist(sample)
    ttr = len(set(sample)) / len(sample)
    hapax = len([w for w, c in fdist.items() if c == 1]) / len(sample)
    return {'TTR_5000': ttr, 'Hapax_Ratio_5000': hapax, 'sample_size': len(sample)}

print("Computing lexical richness on 5,000-word samples per class...")
richness_rows = []
for cid, name in CLASS_NAMES.items():
    row = lexical_richness_on_sample(df_all[df_all['class_id'] == cid]['text'])
    row['class_id'], row['class_name'] = cid, name
    richness_rows.append(row)
richness_df = pd.DataFrame(richness_rows)[['class_id', 'class_name', 'sample_size', 'TTR_5000', 'Hapax_Ratio_5000']]
richness_df.to_csv(RESULTS_DIR / "lexical_richness_5000word_sample.csv", index=False)
print(richness_df.to_string(index=False))

# --- Statistical distinctness: per-feature summary stats + one-way ANOVA across classes ---
print("Running one-way ANOVA per feature to test class distinctness...")
summary_stats = df_all.groupby('class_name')[feature_cols].agg(['mean', 'std']).round(4)
summary_stats = summary_stats.reindex(['Human', 'AI_Standard', 'AI_Mimic'])
summary_stats.to_csv(RESULTS_DIR / "feature_summary_by_class.csv")

anova_rows = []
for col in feature_cols:
    groups = [df_all[df_all['class_id'] == cid][col].dropna() for cid in sorted(df_all['class_id'].unique())]
    f_stat, p_val = stats.f_oneway(*groups)
    anova_rows.append({'feature': col, 'F_statistic': f_stat, 'p_value': p_val, 'significant_p<0.05': p_val < 0.05})
anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(RESULTS_DIR / "feature_anova_pvalues.csv", index=False)
print(anova_df.to_string(index=False))

# Boxplots for every feature, grouped by class, to visually back up the ANOVA table.
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, col in zip(axes.flatten(), feature_cols):
    sns.boxplot(data=df_all, x='class_name', y=col, ax=ax, order=['Human', 'AI_Standard', 'AI_Mimic'])
    ax.set_title(col)
    ax.set_xlabel('')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_distinctness_boxplots.png")
plt.close()

print(f"Task 1 Complete. Dataset and analysis artifacts saved in {RESULTS_DIR}")