# Mini NURA — Healthcare RAG Assistant

A document-grounded Q&A assistant that answers questions using only the provided WHO fact sheets. It refuses questions not covered by the documents and shows which source it used.

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env             # then edit .env with your key

# 4. Add WHO fact sheets
# Paste content from https://www.who.int/news-room/fact-sheets into:
#   docs/diabetes.txt
#   docs/hypertension.txt
#   docs/tuberculosis.txt
#   docs/depression.txt
#   docs/cancer.txt

# 5. Launch notebook
jupyter notebook mini_nura.ipynb
```

## Running the notebook

- **First run:** execute all cells top to bottom. Cell 4 builds the ChromaDB index (takes ~30s).
- **Subsequent runs:** Cell 4 detects the existing index and skips re-embedding. Jump straight to Cell 7 for Q&A.
- **Reset index:** delete the `chroma_db/` folder and re-run Cell 4.

## Project structure

```
mini-nura/
├── .venv/              # virtual environment (gitignored)
├── .env                # ANTHROPIC_API_KEY (gitignored)
├── .gitignore
├── requirements.txt
├── README.md
├── docs/               # WHO fact sheets as .txt files
│   ├── diabetes.txt
│   ├── hypertension.txt
│   ├── tuberculosis.txt
│   ├── depression.txt
│   └── cancer.txt
├── chroma_db/          # persisted vector index (gitignored, auto-generated)
└── mini_nura.ipynb     # main notebook
```
