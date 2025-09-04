# FAERS NLP→SQL — Query Only

A minimal Streamlit app that takes a FAERS natural-language question and returns **only the generated SQL** (no database execution).

## Setup
```bash
pip install -r requirements.txt
```
Add your OpenAI key and schema catalog path to `.streamlit/secrets.toml`.

## Run
```bash
streamlit run app.py
```
Type a question → get SQL. Validator errors (e.g., non-SELECT statements) are shown inline.
