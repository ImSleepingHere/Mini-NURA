# Mini NURA — Healthcare RAG Assistant

A document-grounded Q&A assistant that answers questions using only the provided WHO fact sheets. It refuses questions not covered by the documents and shows which source it used.

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/Scripts/activate

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

## Sample Data

- The data was from WHO's Fact Sheets. The documents includes only what was written in the fact sheet.

## AI Usage
- Used AI for the RAG pipeline
- FastAPI backend and HTML front end
- Document upload
- Out-of-scope logic

## What I changed/fixed
- Changed the APIs multiple times due to errors (some gemini models don't have a RPM)
- Fixed an error where the code was in an infinte for loop in cell 3 (chunk_text function)
- Cleaning the coding, and fixing errors that were produced
- Testing use cases

## Improvement and Extra Ideas
- Creating a front end for the project instead of using the notebook
- Live document upload (on the spot RAG updates)
- Adding the source with each answer