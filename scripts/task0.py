import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import google.generativeai as genai
from tqdm.auto import tqdm
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "task0"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 500 paragraphs per class, split evenly across the two authors.
PARAGRAPHS_PER_AUTHOR = 250


# "gemini-3.5-flash-lite" is the high-throughput sibling with a much higher free RPD;
MODEL_NAME = 'gemini-3.5-flash-lite'

random.seed(42)
load_dotenv(PROJECT_ROOT / ".env")

# 1. Initialization & Local Files
# Multiple keys (from separate Google accounts/projects) are round-robined so each
# key's own daily/per-minute quota is used independently -- effectively multiplying
# total throughput by the number of keys available.
API_KEYS = [
    v for v in (
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("GEMINI_API_KEY2"),
        os.environ.get("GEMINI_API_KEY3"),
    ) if v
]
if not API_KEYS:
    raise ValueError("No API key found. Please ensure your .env file contains GEMINI_API_KEY.")
print(f"Loaded {len(API_KEYS)} Gemini API key(s) for round-robin generation.")

# ~13 calls/min per key,under the model's free-tier RPM cap
GEMINI_CALL_DELAY = 4.5 / len(API_KEYS)

class KeyPool:
    """Round-robins across multiple API keys, dropping any that hit their daily quota."""
    def __init__(self, keys):
        self.keys = list(keys)
        self.exhausted = [False] * len(self.keys)
        self.idx = 0

    def get_active_key(self):
        for _ in range(len(self.keys)):
            if not self.exhausted[self.idx]:
                return self.keys[self.idx]
            self.idx = (self.idx + 1) % len(self.keys)
        return None

    def advance(self):
        self.idx = (self.idx + 1) % len(self.keys)

    def mark_current_exhausted(self):
        self.exhausted[self.idx] = True

key_pool = KeyPool(API_KEYS)
genai.configure(api_key=key_pool.get_active_key())
gemini_model = genai.GenerativeModel(MODEL_NAME)

class DailyQuotaExceededError(Exception):
    """Raised when every available API key has exhausted its per-day request quota, Retrying within the same run cannot fix this."""


def generate_with_retry(prompt, max_cycles=3, base_delay=3.0):
    """Tries every active key once per cycle -- any error (daily quota, per-minute
    rate limit, etc.) immediately rotates to the next key instead of retrying the
    same one, since a different key/project is likely unaffected. Only backs off
    (with growing delay) once a full cycle through all active keys has failed."""
    for cycle in range(max_cycles):
        n_active_at_cycle_start = sum(1 for exhausted in key_pool.exhausted if not exhausted)
        for _ in range(max(n_active_at_cycle_start, 1)):
            key = key_pool.get_active_key()
            if key is None:
                raise DailyQuotaExceededError(
                    f"All {len(key_pool.keys)} API key(s) have exhausted their daily quota for '{MODEL_NAME}'."
                )
            genai.configure(api_key=key)
            try:
                result = gemini_model.generate_content(prompt).text.strip()
                key_pool.advance()
                return result
            except Exception as e:
                err_str = str(e)
                if 'PerDay' in err_str:
                    print(f"Key #{key_pool.idx + 1}/{len(key_pool.keys)} exhausted its daily quota. Dropping it from rotation.")
                    key_pool.mark_current_exhausted()
                    key_pool.advance()
                else:
                    print(f"API Error on key #{key_pool.idx + 1}/{len(key_pool.keys)}: {e}")
                    key_pool.advance()
        if cycle < max_cycles - 1:
            delay = base_delay * (2 ** cycle)
            print(f"All active keys failed this round. Backing off {delay:.0f}s before retrying...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to generate content after {max_cycles} full cycles across all active keys.")

LOCAL_FILES = {
    "Melville": DATA_DIR / "moby dick.txt",
    "Chambers": DATA_DIR / "the king in yellow.txt",
}

CHAPTER_HEADING_RE = re.compile(r'^\s*(CHAPTER|BOOK|PART|VOLUME)\b[\s\divxlcIVXLC.:—-]*$', re.IGNORECASE | re.MULTILINE)
ILLUSTRATION_RE = re.compile(r'\[Illustration[^\]]*\]', re.IGNORECASE)
FOOTNOTE_RE = re.compile(r'\[\d+\]')
WHITESPACE_RE = re.compile(r'[ \t]+')

def load_and_clean_local_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx + len(start_marker):end_idx]

    # Strip residual Gutenberg boilerplate, chapter headings, illustration/footnote markers.
    text = CHAPTER_HEADING_RE.sub('', text)
    text = ILLUSTRATION_RE.sub('', text)
    text = FOOTNOTE_RE.sub('', text)

    seen = set()
    paragraphs = []
    for raw_para in text.split('\n\n'):
        para = WHITESPACE_RE.sub(' ', raw_para.replace('\n', ' ')).strip()
        if len(para.split()) <= 50:
            continue
        if 'project gutenberg' in para.lower():
            continue
        if para in seen:
            continue
        seen.add(para)
        paragraphs.append(para)
    return paragraphs

print("--- Task 0: Data Generation ---")
human_data = []
for author, filepath in LOCAL_FILES.items():
    try:
        paras = load_and_clean_local_txt(filepath)
        sampled = random.sample(paras, min(PARAGRAPHS_PER_AUTHOR, len(paras)))
        for p in sampled:
            human_data.append({"text": p, "author": author, "label": "Human", "class_id": 0})
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'")

df_human = pd.DataFrame(human_data)
print(f"Collected {len(df_human)} human paragraphs.")

# Resume support: if a previous run got quota-cut-short, reuse whatever AI paragraphs, it has already saved instead of regenerating (and re-spending quota on) them from scratch.
DATASET_PATH = RESULTS_DIR / "dataset_raw.csv"
if DATASET_PATH.exists():
    existing_df = pd.read_csv(DATASET_PATH).dropna(subset=['text'])
    print(f"Found existing dataset ({len(existing_df)} rows) at {DATASET_PATH.name}; resuming AI generation.")
else:
    existing_df = pd.DataFrame(columns=["text", "author", "label", "class_id"])

def existing_rows_for(label, author):
    subset = existing_df[(existing_df['label'] == label) & (existing_df['author'] == author)]
    return subset[["text", "author", "label", "class_id"]].to_dict("records")

# 2. Topic Extraction
# Reused verbatim on resumed runs, re-extracting could shuffle the topic list and skew the per-topic paragraph counts already generated in a previous, quota-cut-short run.
topics_path = RESULTS_DIR / "extracted_topics.txt"
if topics_path.exists():
    topics = [line.strip() for line in topics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Reusing {len(topics)} previously extracted topic(s) from {topics_path.name}.")
else:
    extraction_prompt = """
Analyze the overarching themes of classic literature, specifically focusing on the cosmic dread and madness of Robert W. Chambers' 'The King in Yellow' and the epic, obsessive struggle of man vs. nature in Herman Melville's 'Moby-Dick'. 
Provide exactly 10 core topics/themes, formatted as a simple comma-separated list.
"""
    try:
        topics_response = generate_with_retry(extraction_prompt)
    except (DailyQuotaExceededError, RuntimeError) as e:
        raise SystemExit(
            f"Could not extract topics using '{MODEL_NAME}' across all {len(API_KEYS)} key(s). "
            f"If this is a daily quota issue, wait for the reset (midnight Pacific time) and re-run.\n{e}"
        )
    topics = [t.strip() for t in topics_response.replace('.', '').split(',')]
    with open(topics_path, "w") as f:
        f.write("\n".join(topics))
    print("Topics extracted and saved.")

# 3. Text Generation
def generate_ai_texts(topics, author, mimic=False, count=250, existing_texts=None):
    generated = list(existing_texts or [])
    if len(generated) >= count:
        print(f"Already have {len(generated)}/{count} paragraph(s) for {author} (Mimic={mimic}); skipping generation.")
        return generated[:count]
    remaining = count - len(generated)
    # Ceiling division so every last paragraph gets a slot even when remaining isn't
    # evenly divisible by the topic count; the len(generated) >= count checks below
    # stop generation at exactly `count` rather than overshooting.
    per_topic = max(1, -(-remaining // len(topics))) if len(topics) > 0 else remaining
    quota_hit = False
    desc = f"Generating {author} (Mimic={mimic}) [{len(generated)}/{count} already done]"
    for topic in tqdm(topics, desc=desc):
        if quota_hit or len(generated) >= count:
            break
        for _ in range(per_topic):
            if len(generated) >= count:
                break
            if mimic:
                prompt = f"Write a 150-word paragraph about '{topic}' written strictly in the unique literary style, vocabulary, and pacing of {author}. Do not mention the author's name."
                label, class_id = f"AI_Mimic_{author}", 2
            else:
                prompt = f"Write an informative 150-word paragraph exploring the topic of '{topic}'."
                label, class_id = "AI_Standard", 1
            try:
                text = generate_with_retry(prompt)
                generated.append({"text": text, "author": author, "label": label, "class_id": class_id})
            except DailyQuotaExceededError:
                print(
                    f"\nDaily free-tier quota for '{MODEL_NAME}' exhausted. Stopping early with "
                    f"{len(generated)}/{count} paragraph(s) collected for {author} (Mimic={mimic}). "
                    "Re-run this script after the reset (midnight Pacific time) to top up the dataset.\n"
                )
                quota_hit = True
                break
            except Exception as e:
                print(f"API Error (giving up on this paragraph): {e}")
            time.sleep(GEMINI_CALL_DELAY)
    return generated

# 250 paragraphs per author, per class -> 500 paragraphs per class across both authors.
df_ai_base_melville = pd.DataFrame(generate_ai_texts(
    topics, "Melville", mimic=False, count=PARAGRAPHS_PER_AUTHOR,
    existing_texts=existing_rows_for("AI_Standard", "Melville")))
df_ai_mimic_melville = pd.DataFrame(generate_ai_texts(
    topics, "Melville", mimic=True, count=PARAGRAPHS_PER_AUTHOR,
    existing_texts=existing_rows_for("AI_Mimic_Melville", "Melville")))
df_ai_base_chambers = pd.DataFrame(generate_ai_texts(
    topics, "Chambers", mimic=False, count=PARAGRAPHS_PER_AUTHOR,
    existing_texts=existing_rows_for("AI_Standard", "Chambers")))
df_ai_mimic_chambers = pd.DataFrame(generate_ai_texts(
    topics, "Chambers", mimic=True, count=PARAGRAPHS_PER_AUTHOR,
    existing_texts=existing_rows_for("AI_Mimic_Chambers", "Chambers")))

df_all = pd.concat([df_human, df_ai_base_melville, df_ai_mimic_melville, df_ai_base_chambers, df_ai_mimic_chambers], ignore_index=True)
df_all['is_ai'] = df_all['class_id'].apply(lambda x: 1 if x > 0 else 0)

# Save final dataset
df_all.to_csv(RESULTS_DIR / "dataset_raw.csv", index=False)
print(f"Task 0 Complete. Dataset saved to {RESULTS_DIR / 'dataset_raw.csv'}")