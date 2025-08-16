import os, numpy as np, pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import GroupShuffleSplit
import evaluate

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data_prepared"
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)

def load_df():
    df = pd.read_csv(DATA / "all_examples.csv")
    df["text"] = df["context"].fillna("").astype(str)
    return df

def split_by_conversation(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["conversation_id"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

def to_ds(train_df, val_df, tokenizer, label_col):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    td = Dataset.from_pandas(train_df[["text", label_col]].rename(columns={label_col:"label"}), preserve_index=False)
    vd = Dataset.from_pandas(val_df[["text", label_col]].rename(columns={label_col:"label"}), preserve_index=False)
    td = td.map(tok, batched=True)
    vd = vd.map(tok, batched=True)
    td.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    vd.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return DatasetDict(train=td, validation=vd)

def train_one(model_name, df, label_col, out_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_df, val_df = split_by_conversation(df)
    ds = to_ds(train_df, val_df, tokenizer, label_col)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"macro_f1": metric.compute(predictions=preds, references=labels, average="macro")["f1"]}
    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(out_dir))

def main():
    df = load_df()
    base_model = os.environ.get("HF_MODEL", "distilbert-base-uncased")
    train_one(base_model, df, "label_mi", MODELS / "hf_mistake_identification")
    train_one(base_model, df, "label_pg", MODELS / "hf_providing_guidance")
    print("HF models saved under models/")

if __name__ == "__main__":
    main()
