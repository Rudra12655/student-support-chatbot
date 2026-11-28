import os  # [MODIFIED] Added for environment variables and path handling
from pathlib import Path
import re
import sys  # [MODIFIED] Added to detect PyInstaller frozen mode
import threading  # [MODIFIED] Added to launch browser once server starts
import webbrowser  # [MODIFIED] Added to open local chatbot URL automatically

from flask import Flask, jsonify, render_template, request  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

# [MODIFIED] Optional heavy ML/vector packages for lightweight hosting (e.g., PythonAnywhere free tier)
try:
    import chromadb  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    chromadb = None
    SentenceTransformer = None
    VECTOR_SEARCH_AVAILABLE = False


# [MODIFIED] Resolve BASE_DIR using sys._MEIPASS when running inside a PyInstaller .exe
def get_base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "student_support_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# [MODIFIED] Explicitly point Flask to bundled templates and static folders
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

model = None
collection = None

# [MODIFIED] Initialize model and ChromaDB only when available and chroma_db exists
if VECTOR_SEARCH_AVAILABLE and DB_DIR.exists():
    try:
        print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as exc:
        print(f"Vector search disabled due to initialization error: {exc}")
        model = None
        collection = None

# --- CUSTOMIZED ANSWERS START ---
CUSTOM_ANSWERS: dict[str, tuple[list[str], str]] = {
    "greetings": (
        ["hi", "hello", "hey", "hola", "namaste", "good morning", "good afternoon", "good evening"],
        "Hey there! Ask me anything about courses, schedules, assessments, or placements.",
    ),
    "courses": (
        ["course", "courses", "couses", "program", "programs", "track", "syllabus", "topics", "capstone", "learn"],
        "Our main program is the 20-week Data Science & Applied AI Track. No prior coding is required before Week 1—it starts with Python basics and moves through NumPy, pandas, SQL, data visualization, statistics, machine learning, embeddings/vector search, RAG basics, and a capstone project.",
    ),
    "schedules": (
        ["schedule", "schedules", "timing", "timings", "time", "class", "classes", "batch", "batches", "weekend", "weekday", "days"],
        "Classes run in flexible weekday and weekend batches depending on your cohort. Live sessions cover core topics and hands-on practice, with recordings and materials shared so you can stay on track if you miss a slot.",
    ),
    "assessments": (
        ["assessment", "assessments", "test", "tests", "exam", "exams", "quiz", "quizzes", "grade", "grades", "grading", "pass", "retake", "project"],
        "You'll be evaluated through periodic tests/quizzes, hands-on assignments, and your end-to-end capstone project. If you miss a test or need another attempt, follow the assessment guidelines or reach out to your mentor/support team for retake rules.",
    ),
    "placements": (
        ["placement", "placements", "job", "jobs", "career", "careers", "interview", "interviews", "resume", "portfolio", "hiring", "eligibility"],
        "Placement support helps you prepare your resume and portfolio, practice with mock interviews, and connect with hiring opportunities once you meet the course completion and eligibility criteria.",
    ),
}


def get_custom_answer(query: str) -> str | None:
    q = query.lower().strip()
    for keywords, answer in CUSTOM_ANSWERS.values():
        if any(re.search(rf"\b{re.escape(word)}\b", q) for word in keywords):
            return answer
    return None
# --- CUSTOMIZED ANSWERS END ---


def search_kb(query: str, top_k: int = 3) -> list[dict]:
    # [MODIFIED] Safely return empty matches when vector search is not active
    if model is None or collection is None or collection.count() == 0:
        return []

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []
    for doc, meta, dist in zip(docs, metas, distances):
        matches.append(
            {
                "text": doc.strip(),
                "source": meta.get("source", "unknown"),
                "distance": round(float(dist), 4) if dist is not None else None,
            }
        )
    return matches


def _clean_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"^[-*]\s+", "", cleaned)
    return cleaned.strip()


def format_reply(matches: list[dict]) -> str:
    if not matches:
        return "I couldn't find an answer for that right now. Try asking about courses, schedules, assessments, or placements."

    parts = []
    total_chars = 0

    for m in matches:
        lines = [_clean_line(line) for line in m.get("text", "").splitlines()]
        cleaned_chunk = " ".join(line for line in lines if line)
        if len(cleaned_chunk) < 25:
            continue

        parts.append(cleaned_chunk)
        total_chars += len(cleaned_chunk)

        if total_chars > 450:
            break

    if not parts:
        return "I couldn't find a detailed answer for that. Try asking about courses, schedules, assessments, or placements."

    return "\n\n".join(parts)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    custom = get_custom_answer(message)
    if custom:
        return jsonify({"reply": custom, "matches": []})

    matches = search_kb(message, top_k=3)
    reply = format_reply(matches)

    return jsonify(
        {
            "reply": reply,
            "matches": matches,
        }
    )


# [MODIFIED] Helper to open the local app in the default browser once running
def _open_browser(url: str) -> None:
    webbrowser.open(url)


# [MODIFIED] Single clean startup block with debug=False and use_reloader=False
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.5, _open_browser, args=[url]).start()

    app.run(host=host, port=port, debug=False, use_reloader=False)