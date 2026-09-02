from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from peft import PeftModel
from captum.attr import LayerIntegratedGradients

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "task3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("--- Task 3: Interpretability (Captum) ---")

df = pd.read_csv(PROJECT_ROOT / "results" / "task1" / "dataset_with_features.csv").dropna(subset=['text'])
feature_cols = ['TTR', 'Hapax_Ratio', 'Adj_Noun_Ratio', 'Avg_Tree_Depth', 'Flesch_Kincaid', 'Semicolons', 'Em_Dashes', 'Exclamations']

# Recreate the exact same train/test split used in task2.py (same seed, same source
# dataframe) so the "test set" analyzed here matches what the model was evaluated on.
y = df['is_ai']
text_train, text_test, y_text_train, y_text_test = train_test_split(df['text'], y, test_size=0.2, random_state=42)

# Load Tier C Model
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model = PeftModel.from_pretrained(base_model, PROJECT_ROOT / "results" / "task2" / "tier_c_lora_model")
model.eval()
model.to('cpu')

def custom_forward_func(inputs, attention_mask):
    return model(inputs, attention_mask=attention_mask).logits[:, 1]

lig = LayerIntegratedGradients(custom_forward_func, model.base_model.model.roberta.embeddings.word_embeddings)

def predict_batch(texts, batch_size=32):
    """Returns (human_prob, ai_prob, predicted_label) for each text."""
    results = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        for p in probs:
            human_p, ai_p = p[0].item(), p[1].item()
            results.append((human_p, ai_p, 1 if ai_p > human_p else 0))
    return results

# --- Saliency Mapping: which words push the model toward "AI"? ---
AI_ISM_WORDS = {
    'delve', 'tapestry', 'testament', 'navigate', 'underscore', 'realm', 'vibrant',
    'intricate', 'moreover', 'furthermore', 'robust', 'holistic', 'multifaceted',
    'nuanced', 'embark', 'foster', 'crucial', 'pivotal', 'myriad', 'intertwine',
    'profound', 'ultimately', 'landscape',
}
SPECIAL_TOKENS = {'<s>', '</s>', '<pad>'}

imposter_samples = df[df['class_id'] == 1]['text'].head(3).tolist()

saliency_report_lines = []
ai_ism_hits_total = 0
content_word_hits_total = 0
top_token_count_total = 0

for sample_idx, sample_imposter in enumerate(imposter_samples, start=1):
    inputs = tokenizer(sample_imposter, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Calculating Integrated Gradients for sample {sample_idx}/{len(imposter_samples)}...")
    attributions, delta = lig.attribute(
        inputs=input_ids, additional_forward_args=(attention_mask,), n_steps=50, return_convergence_delta=True
    )
    attributions_sum = attributions.sum(dim=-1).squeeze(0)
    attributions_sum = attributions_sum / torch.norm(attributions_sum)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    word_importances = sorted(zip(tokens, attributions_sum.detach().numpy()), key=lambda x: x[1], reverse=True)
    top_words = [(w.replace('Ġ', ''), s) for w, s in word_importances if w not in SPECIAL_TOKENS and s > 0.05]

    saliency_report_lines.append(f"=== Sample {sample_idx} ===")
    saliency_report_lines.append("Analyzed Text:\n" + sample_imposter)
    saliency_report_lines.append("\nTop 'AI' Signaling Tokens:")
    for word, score in top_words:
        saliency_report_lines.append(f"{word}: {score:.4f}")

    for word, _ in top_words[:5]:
        top_token_count_total += 1
        clean = word.lower().strip('.,!?;:')
        if clean in AI_ISM_WORDS:
            ai_ism_hits_total += 1
        elif clean.isalpha() and len(clean) > 3:
            content_word_hits_total += 1
    saliency_report_lines.append("")

# --- Findings: is the signal lexical ("AI-isms") or structural (rhythm/punctuation)? ---
findings = ["=== Findings ==="]
if top_token_count_total == 0:
    findings.append("No tokens crossed the attribution threshold; inconclusive.")
else:
    ai_ism_frac = ai_ism_hits_total / top_token_count_total
    content_frac = content_word_hits_total / top_token_count_total
    rhythm_frac = 1 - ai_ism_frac - content_frac
    findings.append(
        f"Across the top-5 signaling tokens per sample: {ai_ism_frac:.0%} were known 'AI-ism' words "
        f"(e.g. delve, tapestry, testament), {content_frac:.0%} were other content words, and the "
        f"remaining {rhythm_frac:.0%} were function words/punctuation (rhythm/structure signal)."
    )
    if ai_ism_frac >= 0.3:
        findings.append(
            "This suggests the model leans on specific lexical 'tells' (stock AI vocabulary) rather than "
            "purely on sentence rhythm."
        )
    elif rhythm_frac >= 0.5:
        findings.append(
            "This suggests the model leans more on sentence rhythm/structure (function words, punctuation) "
            "than on specific vocabulary choices."
        )
    else:
        findings.append("This suggests a mixed signal: no single mechanism (lexical vs. rhythmic) dominates.")

with open(RESULTS_DIR / "saliency_report.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(saliency_report_lines))
    f.write("\n\n" + "\n".join(findings) + "\n")
print("\n".join(findings))

# --- Error Analysis: find Human samples the model mislabeled as AI ---
print("Running error analysis on the held-out test set...")
test_df = df.loc[text_test.index].copy()
predictions = predict_batch(test_df['text'].tolist())
test_df['human_prob'] = [p[0] for p in predictions]
test_df['ai_prob'] = [p[1] for p in predictions]
test_df['predicted_label'] = [p[2] for p in predictions]

false_positives = test_df[(test_df['is_ai'] == 0) & (test_df['predicted_label'] == 1)]
false_positives = false_positives.sort_values('ai_prob', ascending=False).head(3)

human_means = df[df['class_id'] == 0][feature_cols].mean()

error_lines = [f"Found {len(false_positives)} Human sample(s) misclassified as AI (out of {len(test_df[test_df['is_ai'] == 0])} Human samples in the test set).\n"]

for rank, (_, row) in enumerate(false_positives.iterrows(), start=1):
    error_lines.append(f"=== Misclassified Sample {rank} (author: {row.get('author', 'unknown')}) ===")
    error_lines.append(f"Model confidence: AI={row['ai_prob']:.4f}, Human={row['human_prob']:.4f}")
    error_lines.append("Text:\n" + str(row['text']))
    error_lines.append("\nFeature comparison vs. Human class average:")
    notes = []
    for col in feature_cols:
        diff = row[col] - human_means[col]
        direction = "above" if diff > 0 else "below"
        error_lines.append(f"  {col}: {row[col]:.3f} ({direction} Human avg {human_means[col]:.3f})")
        if col == 'TTR' and diff < -0.05:
            notes.append("unusually low lexical diversity (repetitive vocabulary)")
        if col == 'Hapax_Ratio' and diff < -0.05:
            notes.append("few one-off words, i.e. repetitive word choice")
        if col == 'Avg_Tree_Depth' and diff > 1:
            notes.append("unusually complex/nested sentence structure, atypically 'polished'")
        if col == 'Adj_Noun_Ratio' and diff > 0.1:
            notes.append("heavier adjective use than typical human text (over-describing)")
    if notes:
        error_lines.append("\nLikely explanation: " + "; ".join(notes) + ".")
    else:
        error_lines.append("\nLikely explanation: no single feature stands out; the misclassification may stem "
                            "from subtler stylistic overlap not captured by these 8 features.")
    error_lines.append("")

with open(RESULTS_DIR / "error_analysis.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(error_lines))

print(f"Task 3 Complete. Saliency report and error analysis saved in {RESULTS_DIR}")
