import os
import pickle
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PDF_FOLDER = "knowledge_base"

INDEX_PATH = "faiss_store/rag.index"

DOCS_PATH = "faiss_store/docs.pkl"

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def chunk_text(
        text,
        chunk_size=1000,
        overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += (
            chunk_size
            - overlap
        )

    return chunks


documents = []

for file in os.listdir(PDF_FOLDER):

    if not file.endswith(".pdf"):
        continue

    path = os.path.join(
        PDF_FOLDER,
        file
    )

    print(
        f"Reading {file}"
    )

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = (
            page.extract_text()
            or ""
        )

        text += page_text

    chunks = chunk_text(text)

    for chunk in chunks:

        documents.append({
            "source": file,
            "content": chunk
        })


print(
    f"Total Chunks: "
    f"{len(documents)}"
)

texts = [
    doc["content"]
    for doc in documents
]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(
    embeddings.astype(
        "float32"
    )
)

faiss.write_index(
    index,
    INDEX_PATH
)

with open(
        DOCS_PATH,
        "wb"
) as f:

    pickle.dump(
        documents,
        f
    )

print(
    "\nFAISS Index Saved"
)

print(
    f"Chunks Indexed: "
    f"{len(documents)}"
)