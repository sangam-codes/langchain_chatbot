# [imports]
# Loads helpers used to read and write lasting facts on disk.
import json
from pathlib import Path

MEMORY_PATH = Path(__file__).with_name(".memory.json")


# [load_memory]
# Reads saved facts from the local JSON file.
# 1. Return an empty dict when the file is missing.
# 2. Return the parsed facts from disk.
def load_memory() -> dict:
    # [load_memory-1] Return an empty dict when the file is missing.
    if not MEMORY_PATH.exists():
        return {}

    # [load_memory-2] Return the parsed facts from disk.
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


# [save_memory]
# Writes the full facts dict to the local JSON file.
def save_memory(data: dict) -> None:
    MEMORY_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# [set_fact]
# Saves one lasting fact and confirms what was stored.
# 1. Reject empty keys or values.
# 2. Store the fact and write the file.
def set_fact(key: str, value: str) -> str:
    cleaned_key = key.strip().lower()
    cleaned_value = value.strip()

    # [set_fact-1] Reject empty keys or values.
    if not cleaned_key or not cleaned_value:
        return "Could not save that fact. Provide both a key and a value."

    # [set_fact-2] Store the fact and write the file.
    data = load_memory()
    data[cleaned_key] = cleaned_value
    save_memory(data)
    return f"Saved {cleaned_key}={cleaned_value}"


# [get_facts]
# Returns all saved lasting facts as plain text.
# 1. Say when nothing is saved yet.
# 2. Format each saved fact on its own line.
def get_facts() -> str:
    data = load_memory()

    # [get_facts-1] Say when nothing is saved yet.
    if not data:
        return "No saved facts yet."

    # [get_facts-2] Format each saved fact on its own line.
    return "\n".join(f"{key}: {value}" for key, value in sorted(data.items()))
