from pathlib import Path

import torch
import numpy as np
import pandas as pd
import spacy
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, set_seed
from peft import get_peft_model, LoraConfig, TaskType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "task2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Task 2: Detectors (Using {DEVICE}) ---")

df = pd.read_csv(PROJECT_ROOT / "results" / "task1" / "dataset_with_features.csv").dropna(subset=['text'])
feature_cols = ['TTR', 'Hapax_Ratio', 'Adj_Noun_Ratio', 'Avg_Tree_Depth', 'Flesch_Kincaid', 'Semicolons', 'Em_Dashes', 'Exclamations']

X = df[feature_cols]
y = df['is_ai']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
text_train, text_test, y_text_train, y_text_test = train_test_split(df['text'], y, test_size=0.2, random_state=42)

# --- Tier A: XGBoost ---
tier_a = xgb.XGBClassifier(n_estimators=100, max_depth=4)
tier_a.fit(X_train, y_train)
preds_a = tier_a.predict(X_test)
with open(RESULTS_DIR / "tier_a_report.txt", "w") as f:
    f.write("Tier A (XGBoost) Classification Report:\n")
    f.write(classification_report(y_test, preds_a))
tier_a.save_model(str(RESULTS_DIR / "tier_a_xgboost.json"))

# --- Tier B: FFNN ---
# Uses spaCy's en_core_web_md pretrained static word vectors (300d, GloVe-style vectors
# trained on Common Crawl) as the averaged-embedding representation for each paragraph.
nlp = spacy.load('en_core_web_md')
def get_avg_embedding(text): return nlp(text).vector

X_emb_train = torch.tensor(np.vstack(text_train.apply(get_avg_embedding).values), dtype=torch.float32).to(DEVICE)
X_emb_test = torch.tensor(np.vstack(text_test.apply(get_avg_embedding).values), dtype=torch.float32).to(DEVICE)
y_ten_train = torch.tensor(y_text_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

tier_b = nn.Sequential(
    nn.Linear(300, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1), nn.Sigmoid()
).to(DEVICE)

optimizer = torch.optim.Adam(tier_b.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

tier_b.train()
for epoch in range(100):
    optimizer.zero_grad()
    loss = loss_fn(tier_b(X_emb_train), y_ten_train)
    loss.backward()
    optimizer.step()

tier_b.eval()
with torch.no_grad():
    b_preds = (tier_b(X_emb_test) > 0.5).float().cpu().numpy()
with open(RESULTS_DIR / "tier_b_report.txt", "w") as f:
    f.write("Tier B (FFNN) Classification Report:\n")
    f.write(classification_report(y_test, b_preds))
torch.save(tier_b.state_dict(), RESULTS_DIR / "tier_b_ffnn.pth")

# --- Tier C: RoBERTa + LoRA ---
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"])
tier_c = get_peft_model(base_model, lora_config)

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

train_encodings = tokenizer(text_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(text_test.tolist(), truncation=True, padding=True, max_length=128)

trainer = Trainer(
    model=tier_c,
    args=TrainingArguments(
        output_dir=str(RESULTS_DIR / "lora_checkpoints"),
        num_train_epochs=3, 
        per_device_train_batch_size=16,
        logging_dir=str(RESULTS_DIR / "logs"),
    ),
    train_dataset=TextDataset(train_encodings, y_text_train.tolist()),
    eval_dataset=TextDataset(test_encodings, y_text_test.tolist())
)

print("Training Tier C...")
trainer.train()
eval_results = trainer.evaluate()

test_predictions = trainer.predict(TextDataset(test_encodings, y_text_test.tolist()))
preds_c = np.argmax(test_predictions.predictions, axis=1)
report_c = classification_report(y_text_test, preds_c)

with open(RESULTS_DIR / "tier_c_report.txt", "w") as f:
    f.write("Tier C (RoBERTa LoRA) Classification Report:\n")
    f.write(report_c)
    f.write(f"\n\nEval Results:\n{eval_results}")
tier_c.save_pretrained(RESULTS_DIR / "tier_c_lora_model")

print(f"Task 2 Complete. Models and reports saved in {RESULTS_DIR}")