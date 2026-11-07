# Streamlit review screening app

## Run locally

Use Python 3.13 to match the saved LightGBM model environment. From the project root:

```powershell
python -m pip install -r Streamlit_App/requirements.txt
python tools/make_demo_batch.py
python tools/verify_chapter5_pipeline.py
streamlit run Streamlit_App/app.py
```

At startup, the app downloads the NLTK corpora used by the project and checks the stopword-list hash. This is needed to reproduce the existing cleaning and lemmatization pipeline. The first startup therefore needs outbound access to the NLTK data host.

## Deploy on Streamlit Community Cloud

This folder is self-contained for deployment. Publish only the contents of `Streamlit_App` to a dedicated GitHub repository; do not publish the project-level `data`, `Notebooks`, or `Report_Docs` folders with the app. In Community Cloud, select `app.py` as the entry point and choose Python 3.13. Keep this folder's `requirements.txt` at the repository root. The `models` directory must contain the verified LightGBM bundle and `chapter3_transformers.py` must remain beside the app so the serialized pipeline can load its custom classes.

Community Cloud may hibernate an app after inactivity. This deployment is for demonstration and does not provide an always-on availability guarantee.

## Data integrity and interpretation

- `data/demo_batch_reviews.xlsx` is selected deterministically from 50 actual records in `data/raw/data.csv`; its target labels are excluded. `data/demo_batch_manifest.json` records source line numbers and hashes.
- Preset reviews are actual records looked up by `review_id` in the cleaned dataset, not by treating the ID as a raw-file row number. Preset 12844 is used for the uncertain-boundary case because its saved P(CG)=0.4993. Review 20176 is excluded from that role because its saved P(CG)=0.3604.
- The project-level preset verification script compares raw-text preprocessing and model predictions with existing Chapter 2 and Chapter 4 outputs. It does not retrain the model or create evaluation records.
- Batch demo data are for checking application workflow, not measuring model performance.
- The app reports TreeSHAP contribution values via LightGBM's `pred_contrib`; these explain model contributions and are not causal effects.
