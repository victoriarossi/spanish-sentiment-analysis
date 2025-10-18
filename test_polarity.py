# python3 test_polarity.py \
#  --model_dir polarity_model \
#  --csv ./datasets/testing/sentiment_analysis_dataset_test.csv \
#  --text_col text \
#  --label_col normalized_sentiment \
#  --out_csv ./predictions/sentiment_analysis_predictions.csv


#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification

def load_labels(model_dir):
    path = os.path.join(model_dir, "labels.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            classes = json.load(f)
        return [str(x) for x in classes]
    return None  # fall back to indices if missing

def build_dataloader(tokenizer, texts, max_len=128, batch_size=16):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    return DataLoader(ds, sampler=SequentialSampler(ds), batch_size=batch_size)

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    # Apple Silicon (Metal)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"
    return torch.device("cpu"), "CPU"

def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved mDeBERTa classifier on a CSV.")
    ap.add_argument("--model_dir", default="polarity_model", help="Path to directory with saved model & tokenizer")
    ap.add_argument("--csv", default="train_textos_turisticos.cleaned.csv", help="Path to input CSV")
    ap.add_argument("--text_col", default="text", help="Column with input text")
    ap.add_argument("--label_col", default="label", help="(Optional) column with gold labels for metrics")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--out_csv", default="predictions.csv", help="Where to save row-level predictions")
    ap.add_argument("--seed", type=int, default=2018)
    args = ap.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model + tokenizer
    print("Loading model and tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    class_names = load_labels(args.model_dir)  # may be None
    num_labels = model.config.num_labels
    if class_names is None:
        class_names = [str(i) for i in range(num_labels)]
    print(f"Classes: {class_names}")

    device, dev_name = pick_device()
    model.to(device)
    model.eval()
    print(f"Using device: {dev_name}")

    # Read data
    print(f"Reading input CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in {args.csv}")
    texts = df[args.text_col].astype(str).tolist()
    print(f"Loaded {len(texts)} rows.")

    # Optional labels
    has_labels = args.label_col in df.columns
    if has_labels:
        print(f"Found gold label column: '{args.label_col}'")
    else:
        print("No gold label column found; proceeding without gold labels.")

    # DataLoader
    loader = build_dataloader(tokenizer, texts, max_len=args.max_length, batch_size=args.batch_size)
    print(f"Prepared dataloader with {len(loader)} batches (batch_size={args.batch_size}).")

    # Inference
    print("Starting inference...")
    all_logits = []
    total_batches = len(loader)
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            b_input_ids, b_mask = [t.to(device) for t in batch]
            out = model(b_input_ids, attention_mask=b_mask)
            all_logits.append(out.logits.detach().cpu())
            if i % 10 == 0 or i == total_batches:
                print(f"Processed {i}/{total_batches} batches ({(i/total_batches)*100:.1f}%).")
    print("Inference complete.")

    # ---- Make sure this block is OUTSIDE the loop ----
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    pred_labels = [class_names[i] for i in preds]

    # Print predictions row by row
    print("\nPredictions:\n")
    for i, text in enumerate(texts):
        gold_label = df[args.label_col].iloc[i] if has_labels else "N/A"
        print("LABEL: ", df[args.label_col].iloc[i])
        print(f"Text: {text}")
        print(f"Gold: {gold_label}")
        print(f"Pred: {pred_labels[i]}")
        print("-" * 40)

    # Save per-row predictions to CSV
    out_df = pd.DataFrame({
        args.text_col: texts,
        "pred_label": pred_labels,
        "pred_id": preds,
        "pred_confidence": probs.max(axis=1)
    })
    if has_labels:
        out_df["gold_label"] = df[args.label_col].astype(str).str.strip().values

    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"\nSaved predictions to: {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    main()
