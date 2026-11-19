# [imports]
# Loads helpers used to read local markdown docs and search them with TF-IDF vectors.
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).with_name("docs")
MIN_SCORE = 0.25

_vectorizer: TfidfVectorizer | None = None
_matrix = None
_chunks: list[dict] | None = None


# [is_heading_only]
# True when a chunk is only a markdown title, which is noise in search results.
def is_heading_only(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) == 1 and lines[0].startswith("#")


# [load_chunks]
# Reads every markdown file in docs/ and splits it into searchable chunks.
# 1. Skip work when the docs folder is missing.
# 2. Read each markdown file.
# 3. Split the file on blank lines into chunks with a source name.
# 4. Skip title-only chunks so tool output stays useful.
def load_chunks() -> list[dict]:
    # [load_chunks-1] Skip work when the docs folder is missing.
    if not DOCS_DIR.is_dir():
        return []

    chunks: list[dict] = []

    # [load_chunks-2] Read each markdown file.
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")

        # [load_chunks-3] Split the file on blank lines into chunks with a source name.
        for part in text.split("\n\n"):
            part = part.strip()
            if not part:
                continue

            # [load_chunks-4] Skip title-only chunks so tool output stays useful.
            if is_heading_only(part):
                continue
            chunks.append({"source": path.name, "text": part})
    return chunks


# [build_index]
# Builds a TF-IDF vector index for all doc chunks and keeps it in memory.
# 1. Reuse a built index when it already exists.
# 2. Fit a TF-IDF vectorizer on the chunk texts.
# 3. Store the chunk list, vectorizer, and matrix together.
def build_index() -> tuple[list[dict], TfidfVectorizer, object]:
    global _chunks, _vectorizer, _matrix

    # [build_index-1] Reuse a built index when it already exists.
    if _chunks is not None and _vectorizer is not None and _matrix is not None:
        return _chunks, _vectorizer, _matrix

    chunks = load_chunks()
    if not chunks:
        _chunks = []
        _vectorizer = TfidfVectorizer()
        _matrix = _vectorizer.fit_transform([""])
        return _chunks, _vectorizer, _matrix

    # [build_index-2] Fit a TF-IDF vectorizer on the chunk texts.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    # [build_index-3] Store the chunk list, vectorizer, and matrix together.
    _chunks = chunks
    _vectorizer = vectorizer
    _matrix = matrix
    return _chunks, _vectorizer, _matrix


# [search_local_docs]
# Finds the best matching doc chunks for a query using TF-IDF cosine similarity.
# 1. Build or reuse the in-memory vector index.
# 2. Turn the user query into a TF-IDF vector.
# 3. Score each chunk with cosine similarity.
# 4. Return a clear miss message when nothing is similar enough.
# 5. Return the top matching snippets with their source filenames.
def search_local_docs(
    query: str,
    limit: int = 1,
    min_score: float = MIN_SCORE,
) -> str:
    # [search_local_docs-1] Build or reuse the in-memory vector index.
    chunks, vectorizer, matrix = build_index()
    if not chunks:
        return "No matching docs found."

    cleaned_query = query.strip()
    if not cleaned_query:
        return "No matching docs found."

    # [search_local_docs-2] Turn the user query into a TF-IDF vector.
    query_vector = vectorizer.transform([cleaned_query])

    # [search_local_docs-3] Score each chunk with cosine similarity.
    scores = cosine_similarity(query_vector, matrix)[0]
    ranked = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # [search_local_docs-4] Return a clear miss message when nothing is similar enough.
    best_hits = [
        (float(score), chunks[index])
        for index, score in ranked
        if float(score) >= min_score
    ][:limit]
    if not best_hits:
        return "No matching docs found."

    # [search_local_docs-5] Return the top matching snippets with their source filenames.
    return "\n\n".join(
        f"[{chunk['source']}] {chunk['text']}" for _score, chunk in best_hits
    )
