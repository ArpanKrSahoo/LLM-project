import os, json, glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent.parent
DATA_ROOT = HERE / "IndoML_Datathon"
OUT_DIR = HERE / "data_prepared"
OUT_DIR.mkdir(exist_ok=True)

LABEL_MAP = {"Yes": 2, "To some extent": 1, "No": 0}

def read_json_any(path):
    # Read either a JSON array/object or JSON Lines (one JSON per line).
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        # try JSON lines first
        if "\n" in text and text.lstrip().startswith("{") and "\n{" in text:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        # otherwise standard json
        return json.loads(text)

def normalize_label(v):
    if not isinstance(v, str):
        return None
    v = v.strip()
    # reduce long forms like "Yes (and the answer is correct)"
    if v.startswith("Yes"):
        v2 = "Yes"
    elif v.startswith("No"):
        v2 = "No"
    elif v.lower().startswith("to some extent"):
        v2 = "To some extent"
    else:
        v2 = v
    return LABEL_MAP.get(v2)

def flatten_item(obj):
    cid = obj.get("conversation_id") or obj.get("id") or obj.get("conversationId")
    convo = obj.get("conversation_history") or obj.get("Conversation_History") or obj.get("history") or ""
    split = obj.get("Split") or obj.get("split") or "train"
    data_name = obj.get("Data") or obj.get("data") or ""
    topic = obj.get("Topic") or obj.get("topic") or ""
    gt = obj.get("Ground_Truth_Solution") or obj.get("ground_truth_solution") or ""
    responses = obj.get("anno_llm_responses") or obj.get("responses") or {}
    rows = []
    for model, payload in responses.items():
        if not isinstance(payload, dict):
            continue
        response = payload.get("response") or payload.get("text") or ""
        ann = payload.get("annotation") or {}
        mi = normalize_label(ann.get("Mistake_Identification"))
        pg = normalize_label(ann.get("Providing_Guidance"))
        if mi is None or pg is None:
            continue
        context = f"[CONTEXT]\n{convo}\n[TUTOR]\n{response}"
        uid = f"{cid}::{model}"
        rows.append({
            "uid": uid,
            "conversation_id": cid,
            "model": model,
            "split": split,
            "data": data_name,
            "topic": topic,
            "context": context,
            "response": response,
            "label_mi": mi,
            "label_pg": pg,
        })
    return rows

def main():
    if not DATA_ROOT.exists():
        raise SystemExit(f"Could not find {DATA_ROOT}. Please place the IndoML_Datathon folder at project root.")
    files = []
    for ext in ("*.json", "*.jsonl"):
        files.extend(glob.glob(str(DATA_ROOT / "**" / ext), recursive=True))
    if not files:
        raise SystemExit(f"No .json/.jsonl files found under {DATA_ROOT}.")
    all_rows = []
    for fp in tqdm(files, desc="Reading JSON datasets"):
        try:
            data = read_json_any(fp)
            if isinstance(data, dict):
                data = [data]
            for obj in data:
                all_rows.extend(flatten_item(obj))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["uid"])
    df.to_csv(OUT_DIR / "all_examples.csv", index=False)
    for s in sorted(df["split"].dropna().unique()):
        dfx = df[df["split"] == s]
        dfx.to_csv(OUT_DIR / f"split_{s}.csv", index=False)
    print(f"Saved {len(df)} examples to {OUT_DIR/'all_examples.csv'}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
