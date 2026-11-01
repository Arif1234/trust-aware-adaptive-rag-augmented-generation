# Trust-Aware Adaptive RAG

A command-line Retrieval-Augmented Generation (RAG) prototype. It retrieves records from a local FAISS index, estimates context trust, generates an answer with Llama, and compares answer claims with the retrieved context.

## Pipeline

1. **Indexing:** `src/create_embeddings.py` loads `data/source_info.jsonl`, converts each record's `source_info` field to text, embeds it with `sentence-transformers/all-MiniLM-L6-v2`, and writes a FAISS index and document store.
2. **Retrieval:** `src/adaptive_retrieval.py` searches the index and chooses 3, 5, or 10 records using a distance-based heuristic.
3. **Trust scoring:** `src/trust_scoring.py` combines semantic consistency with a source reliability weight. `main.py` currently passes `source_type="academic"`.
4. **Generation:** `src/llama_rag.py` asks `meta-llama/Llama-3.1-8B` to answer using the retrieved context.
5. **Claim checking:** `src/claim_extractor.py` splits the generated answer into sentences. `src/verifier.py` compares each sentence with a combined embedding of the retrieved context.

## Project Structure

```text
.
├── data/
│   ├── response.jsonl
│   └── source_info.jsonl
├── faiss_store/
│   ├── docs.pkl             # Generated document text
│   └── ragtruth.index       # Generated FAISS index
├── main.py                  # Interactive RAG pipeline
├── requirements.txt
└── src/
    ├── adaptive_retrieval.py
    ├── claim_extractor.py
    ├── create_embeddings.py
    ├── evaluate.py
    ├── hallucination_detector.py
    ├── llama_rag.py
    ├── load_data.py
    ├── trust_scoring.py
    └── verifier.py
```

Run commands from the project root. The scripts use paths such as `data/source_info.jsonl` and `faiss_store/ragtruth.index` relative to the current working directory.

## Requirements

- Python and the packages listed in `requirements.txt`
- Internet access for the first download of the embedding model and Llama weights
- Hugging Face access to `meta-llama/Llama-3.1-8B`
- Sufficient memory and storage for the 8B-parameter language model and downloaded model files

The Llama model is gated. Accept its access terms on Hugging Face before authenticating. On first use, the embedding model and language model are downloaded and cached locally.

## Installation

Create and activate a virtual environment from the project root. This avoids installing packages into Homebrew's externally managed Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Authenticate with Hugging Face if the gated model requires it:

```bash
hf auth login
```

## Build the Search Index

The index must be built before starting `main.py`. The build process reads `data/source_info.jsonl` and creates `faiss_store/ragtruth.index` and `faiss_store/docs.pkl`.

```bash
mkdir -p faiss_store
python -m src.create_embeddings
```

Run this command from the project root with the virtual environment activated. Building the index embeds every `source_info` record and may take time, depending on dataset size and hardware. Rebuild the index after changing `data/source_info.jsonl`.

## Run the Application

```bash
python main.py
```

Enter a question when prompted. The application prints the retrieved records, trust score, generated answer, claim-level scores, and a summary of supported versus unsupported claims.

## Input Data

Both files in `data/` are JSON Lines files: each line is a separate JSON object.

- `source_info.jsonl` is used to build the retrieval index. Each record must contain a `source_info` field.
- `response.jsonl` is loaded by `src/load_data.py`, but it is not used by the current indexing and question-answering path.

Structured values in `source_info` are serialized as JSON text before embedding. The document text is stored in `faiss_store/docs.pkl`, in the same order as vectors in the FAISS index.

## Troubleshooting

- **`externally-managed-environment`:** Activate `.venv` before installing packages. Avoid `--break-system-packages` for this project.
- **FAISS cannot open `ragtruth.index`:** Build the index first, and run commands from the project root. Confirm that both `ragtruth.index` and `docs.pkl` exist in `faiss_store/`.
- **Hugging Face access or authentication error:** Confirm that access to `meta-llama/Llama-3.1-8B` has been granted and authenticate with `hf auth login`.
- **Slow startup or high memory use:** The embedding and Llama models are loaded when their modules are imported. Model downloads and loading can take a while; Llama 3.1 8B may exceed available memory on some machines.
- **Import errors for `src`:** Run from the project root. Build the index with `python -m src.create_embeddings` rather than executing the module file directly.

## Limitations

Trust scores and claim verification are similarity-based heuristics, not proof of factual correctness or entailment. The verifier embeds all retrieved documents as one evidence string and uses a fixed cosine-similarity threshold of `0.50`. The source type is currently hard-coded as `academic`, so the reliability weight may not reflect the actual provenance of every record. Review generated answers and verification results before relying on them.