import argparse
import pandas as pd
from pathlib import Path
from joblib import load

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data_prepared"
MODELS = HERE / "models"

def main(out_path):
    all_csv = DATA / "all_examples.csv"
    if not all_csv.exists():
        raise SystemExit("Run scripts/prepare_data.py first.")
    df = pd.read_csv(all_csv)
    # Prefer an explicit test/public split if present; otherwise sample 20% as a demo.
    if "split" in df.columns:
        mask = df["split"].astype(str).str.lower().isin({"test","public","dev","validation"})
        test_df = df[mask].copy()
        if len(test_df) == 0:
            test_df = df.sample(frac=0.2, random_state=42)
    else:
        test_df = df.sample(frac=0.2, random_state=42)

    mi_model = load(MODELS / "sk_mistake_identification.joblib")
    pg_model = load(MODELS / "sk_providing_guidance.joblib")

    test_df["pred_mi"] = mi_model.predict(test_df["context"].fillna(""))
    test_df["pred_pg"] = pg_model.predict(test_df["context"].fillna(""))

    sub = pd.DataFrame({
        "uid": test_df["uid"],
        "Mistake_Identification": test_df["pred_mi"],
        "Providing_Guidance": test_df["pred_pg"],
    })
    sub.to_csv(out_path, index=False)
    print(f"Wrote submission file to {out_path}")
    print(sub.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(HERE / "submit.csv"))
    args = parser.parse_args()
    main(args.out)
