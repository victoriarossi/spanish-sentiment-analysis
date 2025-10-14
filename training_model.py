import os, json, random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification


SEED = 2018
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----- Import CSV -----

# filename = "train_textos_turisticos_with_annotations"
# filename = "tweets_with_annotations" 
# filename = "reviews_filmaffinity_with_annotations"
filename = "sentiment_analysis_dataset_with_annotations"

df = pd.read_csv(f"./datasets/full datasets/{filename}.csv")
df = df[["text", "normalized_sentiment"]]

# print(df.head())


train = df.sample(frac=0.8, random_state=SEED)
test = df.drop(train.index)
print(f"Train size: {len(train)} | Test size: {len(test)}")

until = len(filename) - len("_with_annotations")
test.to_csv(f'./datasets/testing/{filename[:until]}_test.csv', index=False)
train.to_csv(f'./datasets/training/{filename[:until]}_train.csv', index=False)

df = train

print("Data loaded.")

# ----- Tokenize -----
model_id = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

enc = tokenizer(
    df["text"].tolist(), 
    padding="max_length", 
    truncation=True, 
    max_length=128, 
    return_tensors="pt"
)
input_ids, attention_masks = enc["input_ids"], enc["attention_mask"]

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
device = torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
use_mps = torch.backends.mps.is_available()
# scaler = torch.cuda.amp.GradScaler(enabled=not use_mps and torch.cuda.is_available())

def evaluate(m, loader):
    m.eval(); correct=total=0; loss_sum=0.0; n=0
    with torch.no_grad():
        for x, msk, y in loader:
            x, msk, y = x.to(device), msk.to(device), y.to(device)
            out = m(x, attention_mask=msk, labels=y)
            pred = out.logits.argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
            loss_sum += out.loss.item(); n += 1
    return (correct/max(total,1)), (loss_sum/max(n,1))


print("Model and Data ready, starting training...")
# ----- Train until convergence -----
PATIENCE = 3
MAX_EPOCHS = 20
best = float("inf")
no_improve = 0

for epoch in range(1, MAX_EPOCHS+1):
    model.train(); run_loss=0.0
    total_steps = len(train_loader)
    for step, (x, msk, y) in enumerate(train_loader, 1):
        x, msk, y = x.to(device), msk.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, attention_mask=msk, labels=y)
        loss = out.loss
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        print(f"\tStep: {step} of {total_steps}")
    val_acc, val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch} | loss={run_loss/len(train_loader):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
    if val_loss < best - 1e-4: 
        best = val_loss; 
        no_improve = 0
    else: 
        no_improve += 1
    if no_improve >= PATIENCE: 
        break

# ----- Save -----
save_dir = "polarity_model"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
with open(os.path.join(save_dir, "labels.json"), "w") as f:
    json.dump(uniques.tolist(), f)
print(os.path.abspath(save_dir))