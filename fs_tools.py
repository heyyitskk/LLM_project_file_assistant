"""
fs_tools.py
-----------
Core file-system tools for the LLM-Powered File System Assistant.

Each function returns a structured dict so it can be safely serialized
and handed back to an LLM as a "tool result".

Supported document types for reading: .pdf, .txt, .docx
"""

import os
import re
from datetime import datetime
from pathlib import Path

# Optional third-party parsers. We import lazily / defensively so the
# module still loads even if one of them isn't installed, and we give
# a clear error message if a user tries to read that file type.
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx  # python-docx
except ImportError:
    docx = None


# ---------------------------------------------------------------------------
# 1. read_file
# ---------------------------------------------------------------------------
def read_file(filepath: str) -> dict:
    """
    Read a resume file (PDF, TXT, or DOCX) and extract its text content.

    Args:
        filepath: Path to the file to read.

    Returns:
        dict with keys:
            success (bool)
            filepath (str)
            content (str)      - extracted text, "" on failure
            metadata (dict)    - file_size_bytes, extension, modified_date,
                                  num_characters, num_words
            error (str | None)
    """
    result = {
        "success": False,
        "filepath": filepath,
        "content": "",
        "metadata": {},
        "error": None,
    }

    path = Path(filepath)

    if not path.exists():
        result["error"] = f"File not found: {filepath}"
        return result

    if not path.is_file():
        result["error"] = f"Path is not a file: {filepath}"
        return result

    extension = path.suffix.lower()

    try:
        if extension == ".txt":
            content = path.read_text(encoding="utf-8", errors="replace")

        elif extension == ".pdf":
            if PdfReader is None:
                result["error"] = (
                    "pypdf is not installed. Run: pip install pypdf"
                )
                return result
            reader = PdfReader(str(path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            content = "\n".join(pages_text)

        elif extension == ".docx":
            if docx is None:
                result["error"] = (
                    "python-docx is not installed. Run: pip install python-docx"
                )
                return result
            document = docx.Document(str(path))
            content = "\n".join(p.text for p in document.paragraphs)

        else:
            result["error"] = (
                f"Unsupported file extension '{extension}'. "
                f"Supported types: .pdf, .txt, .docx"
            )
            return result

        stat = path.stat()
        result["success"] = True
        result["content"] = content
        result["metadata"] = {
            "file_size_bytes": stat.st_size,
            "extension": extension,
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "num_characters": len(content),
            "num_words": len(content.split()),
        }

    except Exception as e:
        result["error"] = f"Failed to read file: {e}"

    return result


# ---------------------------------------------------------------------------
# 2. list_files
# ---------------------------------------------------------------------------
def list_files(directory: str, extension: str = None) -> list:
    """
    List files in a directory, optionally filtered by extension.

    Args:
        directory: Directory path to scan.
        extension: Optional extension filter, e.g. ".pdf" or "pdf".

    Returns:
        list of dicts, one per file, each containing:
            name, path, size_bytes, modified_date, extension
        Returns a list with a single error dict if the directory is invalid.
    """
    dir_path = Path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        return [{"error": f"Directory not found: {directory}"}]

    if extension and not extension.startswith("."):
        extension = f".{extension}"

    files_info = []
    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue
        if extension and entry.suffix.lower() != extension.lower():
            continue

        stat = entry.stat()
        files_info.append(
            {
                "name": entry.name,
                "path": str(entry),
                "size_bytes": stat.st_size,
                "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": entry.suffix.lower(),
            }
        )

    return files_info


# ---------------------------------------------------------------------------
# 3. write_file
# ---------------------------------------------------------------------------
def write_file(filepath: str, content: str) -> dict:
    """
    Write text content to a file, creating parent directories if needed.

    Args:
        filepath: Destination path.
        content: Text content to write.

    Returns:
        dict with keys: success, filepath, bytes_written, error
    """
    result = {
        "success": False,
        "filepath": filepath,
        "bytes_written": 0,
        "error": None,
    }

    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        result["success"] = True
        result["bytes_written"] = len(content.encode("utf-8"))
    except Exception as e:
        result["error"] = f"Failed to write file: {e}"

    return result


# ---------------------------------------------------------------------------
# 4. search_in_file
# ---------------------------------------------------------------------------
def search_in_file(filepath: str, keyword: str, context_chars: int = 60) -> dict:
    """
    Case-insensitively search for a keyword inside a file's content and
    return each match along with surrounding context.

    Args:
        filepath: File to search (uses read_file internally, so PDF/TXT/DOCX
                   are all supported).
        keyword: Keyword or phrase to search for.
        context_chars: Number of characters of context to include on each
                        side of a match.

    Returns:
        dict with keys:
            success (bool)
            filepath (str)
            keyword (str)
            match_count (int)
            matches (list[dict])  - each with 'context' and 'position'
            error (str | None)
    """
    result = {
        "success": False,
        "filepath": filepath,
        "keyword": keyword,
        "match_count": 0,
        "matches": [],
        "error": None,
    }

    read_result = read_file(filepath)
    if not read_result["success"]:
        result["error"] = read_result["error"]
        return result

    content = read_result["content"]
    pattern = re.escape(keyword)

    matches = list(re.finditer(pattern, content, flags=re.IGNORECASE))

    for m in matches:
        start = max(0, m.start() - context_chars)
        end = min(len(content), m.end() + context_chars)
        snippet = content[start:end].replace("\n", " ").strip()
        result["matches"].append(
            {
                "position": m.start(),
                "context": f"...{snippet}..." if start > 0 or end < len(content) else snippet,
            }
        )

    result["success"] = True
    result["match_count"] = len(matches)
    return result


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Listing .txt files in ./resumes:")
    for f in list_files("resumes", extension=".txt"):
        print(" ", f)

    print("\nReading resumes/resume_john_doe.txt:")
    r = read_file("resumes/resume_john_doe.txt")
    print(" success:", r["success"], "| words:", r["metadata"].get("num_words"))

    print("\nSearching for 'python' in resumes/resume_john_doe.txt:")
    s = search_in_file("resumes/resume_john_doe.txt", "python")
    print(" matches found:", s["match_count"])
    for match in s["matches"]:
        print("   ", match["context"])
