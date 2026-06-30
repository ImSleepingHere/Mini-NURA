# Mini NURA — Healthcare RAG Assistant

A document-grounded Q&A assistant that answers questions using only the provided WHO fact sheets. It refuses questions not covered by the documents and shows which source it used.

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key to a .env file (you can make one)

# 4. Add your documents
Can only be a .txt document. To add a source make sure that the first line in the document be SOURCE: your-link-here

# 5. Launch notebook
jupyter notebook mini_nura.ipynb
```
## FastAPI setup
```bash
# 1. Open terminal in vscode
# 2. type the following:
uvicorn app:app --reload
# 3. go to http://127.0.0.1:8000/ and it should be there
```

## Running the notebook

- **First run:** execute all cells top to bottom. Cell 4 builds the ChromaDB index (takes ~30s).
- **Subsequent runs:** Cell 4 detects the existing index and skips re-embedding. Jump straight to Cell 7 for Q&A.
- **Reset index:** delete the `chroma_db/` folder and re-run Cell 4.

## Project structure

```
mini-nura/
├── requirements.txt
├── README.md
├── docs/               # WHO fact sheets as .txt files
│   ├── asthma.txt
│   ├── botulism.txt
│   ├── cancer.txt
│   ├── hantavirus.txt
│   ├── tetanus.txt
│   └── cancer.txt
├── chroma_db/          # persisted vector index (gitignored, auto-generated)
└── mini_nura.ipynb     # main notebook
```
