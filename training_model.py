import os, io, csv, json, random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification

# ----- Reproducibility -----
SEED = 2018
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ----- Import CSV -----

# filename = "train_textos_turisticos"
# filename = "tweets_with_annotations" 
filename = "reviews_filmaffinity_with_annotations"
# filename = "sentiment_analysis_dataset_with_annotations"

df = pd.read_csv(f"./datasets/full datasets/{filename}.csv")
df = df[["text", "normalized_sentiment"]]

print(df.head())


train = df.sample(frac=0.8, random_state=SEED)
test = df.drop(train.index)
print(f"Train size: {len(train)} | Test size: {len(test)}")

test.to_csv(f'./datasets/testing/{filename}_test.csv', index=False)
train.to_csv(f'./datasets/training/{filename}_train.csv', index=False)

df = train

print("✅ Data loaded.")

# ----- Tokenize -----
model_id = "dccuchile/bert-base-spanish-wwm-uncased" 
tokenizer = BertTokenizer.from_pretrained(model_id)
enc = tokenizer(
    df["text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
input_ids = enc["input_ids"]
attention_masks = enc["attention_mask"]

# ----- Labels -----
codes, uniques = pd.factorize(df["normalized_sentiment"].astype(str).str.strip())
labels = torch.tensor(codes, dtype=torch.long)
num_labels = len(uniques)

# ----- Split -----
idx = np.arange(len(df))
train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=SEED, stratify=codes)
train_ds = TensorDataset(input_ids[train_idx], attention_masks[train_idx], labels[train_idx])
val_ds   = TensorDataset(input_ids[val_idx],   attention_masks[val_idx],   labels[val_idx])

train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=8)
val_loader   = DataLoader(val_ds,   sampler=SequentialSampler(val_ds),  batch_size=8)

# ----- Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

def evaluate(model, data_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in data_loader:
            b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
            out = model(b_input_ids, attention_mask=b_mask, labels=b_labels)
            logits = out.logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == b_labels).sum().item()
            total += b_labels.size(0)
    return correct / max(total, 1)

print("✅ Model and Data ready, starting training...")
# ----- Train (tiny dataset → few epochs) -----
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    print("Epoch ", epoch+1, " of ", EPOCHS)
    for step, batch in enumerate(train_loader):
        print("\t Step: ", step, " of ", len(train_loader))
        b_input_ids, b_mask, b_labels = [t.to(device) for t in batch]
        optimizer.zero_grad()
        out = model(b_input_ids, attention_mask=b_mask, labels=b_labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss={running_loss/len(train_loader):.4f} | val_acc={val_acc:.3f}")

# ----- Save -----
save_dir = "polarity_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
with open(os.path.join(save_dir, "labels.json"), "w") as f:
    json.dump(uniques.tolist(), f)
print(f"Saved to: {os.path.abspath(save_dir)}")
