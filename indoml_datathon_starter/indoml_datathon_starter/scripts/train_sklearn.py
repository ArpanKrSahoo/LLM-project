import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from joblib import dump

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data_prepared"
MODELS = HERE / "models"
MODELS.mkdir(exist_ok=True)

def load_data():
    path = DATA / "all_examples.csv"
    if not path.exists():
        raise SystemExit("Run scripts/prepare_data.py first.")
    df = pd.read_csv(path)
    df["text"] = df["context"].fillna("").astype(str)
    return df

def split_by_conversation(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["conversation_id"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

def train_task(df, label_col, model_name):
    train_df, val_df = split_by_conversation(df)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=100000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipe.fit(train_df["text"], train_df[label_col])
    val_pred = pipe.predict(val_df["text"])
    macro_f1 = f1_score(val_df[label_col], val_pred, average="macro")
    acc = accuracy_score(val_df[label_col], val_pred)
    print(f"[{label_col}] Macro-F1={macro_f1:.4f}  Acc={acc:.4f}")
    print(classification_report(val_df[label_col], val_pred, digits=4))
    dump(pipe, MODELS / f"{model_name}.joblib")
    return pipe

def main():
    df = load_data()
    print(f"Loaded {len(df)} examples.")
    train_task(df, "label_mi", "sk_mistake_identification")
    train_task(df, "label_pg", "sk_providing_guidance")
    print(f"Models saved to: {MODELS}")

if __name__ == "__main__":
    main()
