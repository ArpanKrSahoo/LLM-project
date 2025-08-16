# IndoML Datathon 2025 — Starter Kit (LLM Tutor Evaluation)

This starter kit helps you train baseline models **from scratch** on the official dataset:
**`UnifyingAITutorEvaluation/IndoML_Datathon`** (GitHub).

> Put the downloaded `IndoML_Datathon/` folder at the project root (same level as this file)
> so paths look like:
>
> ```text
> indoml_datathon_starter/
>   scripts/
>   requirements.txt
>   IndoML_Datathon/        <-- you place this here
> ```

## What you will build
Two simple classifiers that evaluate a tutor's reply given the dialogue context:
1. **Mistake Identification** — does the tutor correctly spot the student's mistake? *(Yes / To some extent / No)*
2. **Providing Guidance** — does the tutor give helpful guidance? *(Yes / To some extent / No)*

We train fast, reliable baselines using **TF–IDF + Logistic Regression** (no GPU required). A Hugging Face fine-tuning script is also provided if you want to try a transformer later.

## Quickstart

```bash
# 0) (optional) create a venv and activate it
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 1) install deps
pip install -r requirements.txt

# 2) prepare the data -> creates CSVs under data_prepared/
python scripts/prepare_data.py

# 3) train both tasks (saves models under models/ and prints Macro-F1)
python scripts/train_sklearn.py

# 4) run predictions for the held-out test/public split and write a submission file
python scripts/predict.py --out submit.csv
```

> ⚠️ **Submission format varies by platform.** This kit writes `submit.csv` with columns:
> `uid, Mistake_Identification, Providing_Guidance` where `uid = conversation_id::model_name`.
> Check the Datathon/CodaBench page and adjust `scripts/predict.py` if a different header is required.

## How the data is read

Each conversation JSON looks like (simplified):

```json
{
  "conversation_id": "2895106109",
  "conversation_history": "Tutor: ...\nStudent: ...",
  "Data": "Bridge",
  "Split": "train",
  "Topic": "4.5A.Geometric Lines",
  "ground_truth_solution": "Not Available",
  "anno_llm_responses": {
    "GPT4": {
      "response": "Good try, but ...",
      "annotation": {
        "Mistake_Identification": "Yes",
        "Providing_Guidance": "To some extent"
      }
    },
    "... other tutors ...": { ... }
  }
}
```

We **flatten** this into rows, one per tutor response, with columns:
`conversation_id, model, split, context, response, label_mi, label_pg`.

Labels are normalized to integers:
- `Yes -> 2`, `To some extent -> 1`, `No -> 0`

## Tips
- Always split **by `conversation_id`** to avoid leakage.
- Start with the sklearn baseline; move to the HF script once everything works.
- Use **Macro F1** as the main metric (matches the Datathon spec).

Good luck — and have fun! 🎉
