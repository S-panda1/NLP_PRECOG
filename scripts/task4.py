import os
import random
import time
from pathlib import Path

import torch
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "task4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

print("--- Task 4: The Turing Test (Genetic Algorithm) ---")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = 'gemini-3.1-flash-lite'

# Multiple keys (from separate Google accounts/projects) are round-robined so each
# key's own daily/per-minute quota is used independently.
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

df = pd.read_csv(PROJECT_ROOT / "results" / "task1" / "dataset_with_features.csv").dropna(subset=['text'])
initial_population = df[(df['class_id'] == 1)]['text'].head(10).tolist()

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model = PeftModel.from_pretrained(base_model, PROJECT_ROOT / "results" / "task2" / "tier_c_lora_model").to(DEVICE)
model.eval()

def get_human_probability(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)
    return probs[0][0].item() # Prob of class 0 (Human)

def mutate_text(text):
    strategies = [
        "Rewrite this paragraph to slightly alter the sentence rhythm. Introduce a minor grammatical quirk.",
        "Rewrite this paragraph using a more archaic vocabulary and vary the sentence length drastically.",
        "Paraphrase this text to sound more colloquial and less perfectly structured. Remove words like 'tapestry' or 'delve'."
    ]
    prompt = f"{random.choice(strategies)}\n\nText: {text}"
    return generate_with_retry(prompt)

with open(RESULTS_DIR / "evolution_log.txt", "w", encoding='utf-8') as log_file:
    log_file.write("Starting Genetic Algorithm Evasion\n==============================\n")
    
    population = initial_population.copy()
    generations = 5
    
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        
        scored_population = [(get_human_probability(t), t) for t in population]
        scored_population.sort(key=lambda x: x[0], reverse=True)
        survivors = [text for fitness, text in scored_population[:3]]
        
        best_score = scored_population[0][0]
        msg = f"Gen {gen+1} - Top Human Prob: {best_score:.4f}\n"
        print(msg.strip())
        log_file.write(msg)
        
        if best_score > 0.90:
            msg = "Success: Bypassed detector with >90% confidence.\n"
            print(msg.strip())
            log_file.write(msg)
            break
            
        population = survivors.copy()
        quota_hit = False
        while len(population) < 10:
            try:
                population.append(mutate_text(random.choice(survivors)))
            except (DailyQuotaExceededError, RuntimeError) as e:
                msg = f"Gemini generation failed mid-mutation, stopping GA early: {e}\n"
                print(msg.strip())
                log_file.write(msg)
                quota_hit = True
                break
        if quota_hit:
            break

    log_file.write("\n\n--- EVOLVED MASTERPIECE ---\n")
    log_file.write(scored_population[0][1])

print(f"Task 4 Complete. Evolution logs and final text saved to {RESULTS_DIR}")

# --- The Personal Test ---
print("\n--- The Personal Test ---")
personal_essay_path = PROJECT_ROOT / "data" / "my_essay.txt"
personal_log_path = RESULTS_DIR / "personal_test.txt"

with open(personal_log_path, "w", encoding='utf-8') as pf:
    pf.write("Personal Test: Detector applied to your own writing\n")
    pf.write("=" * 55 + "\n\n")

    if not personal_essay_path.exists():
        msg = (
            f"No personal essay found at {personal_essay_path}.\n"
            "To run this test: add a plain-text file there (e.g. your SOP or a recent essay), "
            "with paragraphs separated by a blank line, then re-run this script.\n"
        )
        print(msg)
        pf.write(msg)
    else:
        with open(personal_essay_path, 'r', encoding='utf-8') as ef:
            essay_text = ef.read()
        paragraphs = [p.strip() for p in essay_text.split('\n\n') if p.strip()]
        pf.write(f"Loaded {len(paragraphs)} paragraph(s) from {personal_essay_path.name}\n\n")

        scores = []
        for i, para in enumerate(paragraphs, start=1):
            human_prob = get_human_probability(para)
            scores.append(human_prob)
            pf.write(f"Paragraph {i} - Human probability: {human_prob:.4f} (AI probability: {1 - human_prob:.4f})\n")

        avg_score = sum(scores) / len(scores) if scores else 0
        pf.write(f"\nAverage Human probability across essay: {avg_score:.4f}\n\n")

        if avg_score < 0.5:
            pf.write(
                "Verdict: the detector leans 'AI' on your own writing.\n"
                "Manual next steps (do this yourself, then re-run to check the new score):\n"
                "  - Vary sentence length/rhythm; avoid uniformly 'balanced' sentence structures.\n"
                "  - Cut stock phrasing (e.g. 'delve', 'tapestry', 'moreover', 'furthermore').\n"
                "  - Keep small idiosyncrasies/imperfections typical of your own voice.\n"
            )
        else:
            pf.write(
                "Verdict: the detector leans 'Human' on your own writing.\n"
                "Try the reverse experiment: manually rewrite one paragraph to sound like an LLM "
                "(overly helpful, structured, repetitive, hedged) and re-run this script to see if you "
                "can fool your own detector.\n"
            )

        # Automated proxy for "rewrite to sound like an LLM" (true manual rewriting is on you;
        # this just demonstrates the effect using Gemini as a stand-in mutator).
        if paragraphs:
            llmify_prompt = (
                "Rewrite the following paragraph to sound like a typical AI assistant response: "
                "overly helpful, highly structured, repetitive, and hedged. Keep the same core meaning.\n\n"
                f"Text: {paragraphs[0]}"
            )
            try:
                llmified = generate_with_retry(llmify_prompt)
                llmified_score = get_human_probability(llmified)
                pf.write("\n--- Automated 'sound like an LLM' rewrite of paragraph 1 (Gemini-assisted) ---\n")
                pf.write(llmified + "\n")
                pf.write(f"Human probability after LLM-ification: {llmified_score:.4f} "
                         f"(was {scores[0]:.4f} before)\n")
            except Exception as e:
                pf.write(f"\nCould not generate LLM-ified rewrite: {e}\n")

print(f"Personal test results saved to {personal_log_path}")