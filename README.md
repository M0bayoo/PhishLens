# PhishLens

A real-time phishing detection Chrome extension using a two-phase hybrid
pipeline (semantic language model + structural Random Forest classifier),
with SHAP-based explainability, a dual-mode (simple/expert) verdict
interface, and an optional active-blocking confirmation gate.

MSc Dissertation Project — Leeds Beckett University (CRN-19236)
Author: John Oluwatobi Ogunbayo (C77628782)

## Project Structure

```
PhishLens/
├── data/
│   ├── raw/            # Raw downloaded datasets (not committed — see .gitignore)
│   └── processed/       # Cleaned, merged, feature-extracted datasets
├── src/
│   ├── data_collection/ # Scripts to download and explore PhiUSIIL, PhishTank, Tranco
│   └── feature_extraction/ # Modular feature extraction pipeline (5 categories)
├── notebooks/           # Exploratory analysis notebooks
├── models/               # Trained models (ONNX exports — not committed)
├── docs/                 # Methodology and supporting documentation
└── requirements.txt
```

## Dataset Sources

| Source | Type | Role |
|---|---|---|
| PhiUSIIL | Phishing + Legitimate | Historical base (235,795 URLs, ~56 pre-extracted features) |
| PhishTank | Phishing only | Recent verified phishing URLs (live API) |
| Tranco Top 1M | Legitimate only | Recent diverse legitimate URLs (weekly list) |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline Stages

1. **Data Collection** (`src/data_collection/`)
   - `01_download_data.py` — downloads all three sources
   - `02_explore_data.py` — inspects schema, row counts, label balance

2. **Feature Extraction** (`src/feature_extraction/`) — *in progress*
   - URL-lexical features
   - Host/DNS features
   - TLS certificate features
   - Rendered page features
   - Behavioural (redirect depth) features

3. **Model Training** — *planned*
4. **Score Fusion & SHAP** — *planned*
5. **Chrome Extension** — *planned*

See `docs/` for the full methodology document.
