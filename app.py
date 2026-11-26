from pathlib import Path
import re
from flask import Flask, jsonify, render_template, request  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

import chromadb  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "student_support_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

app = Flask(__name__)

print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

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
    if collection.count() == 0:
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)