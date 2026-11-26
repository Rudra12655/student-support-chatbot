from pathlib import Path
import importlib

try:
    chromadb = importlib.import_module("chromadb")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'chromadb' package is required. Install it with: pip install chromadb"
    ) from exc

try:
    sentence_transformers = importlib.import_module("sentence_transformers")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'sentence-transformers' package is required. Install it with: pip install sentence-transformers"
    ) from exc

SentenceTransformer = sentence_transformers.SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "student_support_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_markdown_chunks(data_dir: Path) -> list[dict]:
    """Read each .md file in data/ and split it into clean sections/chunks."""
    chunks = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for md_file in sorted(data_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        raw_sections = text.split("\n## ")
        for idx, section in enumerate(raw_sections):
            section = section.strip()
            if not section:
                continue
            if idx > 0 and not section.startswith("## "):
                section = "## " + section

            chunks.append(
                {
                    "id": f"{md_file.stem}_{idx}",
                    "text": section,
                    "source": md_file.name,
                }
            )

    return chunks


def build_vector_db():
    """Embed chunks with SentenceTransformer and store them in ChromaDB."""
    chunks = load_markdown_chunks(DATA_DIR)
    if not chunks:
        raise ValueError(f"No markdown content found in {DATA_DIR}")

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Encoding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(DB_DIR))
    existing_names = [col.name for col in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )

    print(f"Saved {len(chunks)} chunks to ChromaDB at '{DB_DIR}'.")
    return collection, model


def test_search(collection, model, query: str, top_k: int = 2):
    """Run a sample semantic search against the stored collection."""
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    print(f"\nQuery: '{query}'")
    print("-" * 50)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source", "unknown")
        preview = " ".join(doc.split())[:220]
        print(f"{rank}. [{source}] {preview}...")


if __name__ == "__main__":
    col, emb_model = build_vector_db()
    test_search(col, emb_model, "What are the weekend batch timings and recordings?")
    test_search(col, emb_model, "How do I qualify for placement assistance?")
    test_search(col, emb_model, "Can I retake the assessment test?")