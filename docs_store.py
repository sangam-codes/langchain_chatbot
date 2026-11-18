# [imports]
# Loads helpers used to read local markdown docs and search their chunks.
from pathlib import Path

DOCS_DIR = Path(__file__).with_name("docs")


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


# [search_local_docs]
# Finds the best matching doc chunks for a query using simple word overlap.
# 1. Build a list of meaningful query words.
# 2. Score each chunk by how many query words it contains.
# 3. Return a clear miss message when nothing matches.
# 4. Return the top matching snippets with their source filenames.
def search_local_docs(query: str, limit: int = 1) -> str:
    # [search_local_docs-1] Build a list of meaningful query words.
    words = [word.lower() for word in query.split() if len(word) > 2]
    if not words:
        return "No matching docs found."

    # [search_local_docs-2] Score each chunk by how many query words it contains.
    scored: list[tuple[int, dict]] = []
    for chunk in load_chunks():
        haystack = chunk["text"].lower()
        score = sum(1 for word in words if word in haystack)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    # [search_local_docs-3] Return a clear miss message when nothing matches.
    if not scored:
        return "No matching docs found."

    # [search_local_docs-4] Return the top matching snippets with their source filenames.
    lines = []
    for _score, chunk in scored[:limit]:
        lines.append(f"[{chunk['source']}] {chunk['text']}")
    return "\n\n".join(lines)
