"""
fs_tools.py
-----------
Core file-system tools for the LLM-Powered File System Assistant.

Design principles (important for reliability when an LLM is calling these
as tools):
  - Every function returns a structured dict (never raises for expected/
    predictable problems) so a caller — human or LLM — always gets a
    well-formed result back, never a stack trace.
  - Every dict always has the same keys regardless of success/failure, so
    downstream code doesn't need to guess which keys exist.
  - "Nothing found" (empty directory, zero search matches) is treated as a
    *successful* result with an explanatory message, not an error --
    finding nothing is a valid outcome, not a failure.
  - Genuine problems (missing file, unsupported format, bad permissions,
    corrupt document) are treated as errors with a specific, actionable
    message rather than a generic "something went wrong".

Supported document types for reading: .pdf, .txt, .docx
"""

import os
import re
from datetime import datetime
from pathlib import Path

# Optional third-party parsers. We import lazily / defensively so the
# module still loads even if one of them isn't installed, and we can give
# a clear "please pip install X" error message the first time a user
# actually tries to read that file type, rather than crashing on import.
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

# File extensions read_file() knows how to handle. Used both for the
# actual parsing logic and to generate a helpful error message when an
# unsupported type is requested.
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


# ---------------------------------------------------------------------------
# 1. read_file
# ---------------------------------------------------------------------------
def read_file(filepath: str) -> dict:
    """
    Read a resume file (PDF, TXT, or DOCX) and extract its text content.

    Handles these edge cases explicitly (rather than letting them raise):
      - empty / non-string filepath
      - file does not exist
      - path exists but is a directory, not a file
      - unsupported file extension
      - required parser library not installed (pypdf / python-docx)
      - file exists but is empty (0 bytes)
      - PDF/DOCX opens fine but no text could be extracted (e.g. a
        scanned/image-only PDF with no OCR layer) -- this is reported as
        a *warning*, not an error, since the read technically succeeded
      - corrupted or unreadable document content
      - permission errors

    Args:
        filepath: Path to the file to read.

    Returns:
        dict with keys:
            success (bool)
            filepath (str)
            content (str)      - extracted text, "" if unavailable
            metadata (dict)    - file_size_bytes, extension, modified_date,
                                  num_characters, num_words (empty dict on
                                  failure)
            warning (str|None) - set when the read succeeded but the
                                  result may not be what the caller expects
                                  (e.g. empty text extracted from a PDF)
            error (str | None) - set when the read failed outright
    """
    result = {
        "success": False,
        "filepath": filepath,
        "content": "",
        "metadata": {},
        "warning": None,
        "error": None,
    }

    # --- Edge case: bad input type / empty string -------------------------
    if not filepath or not isinstance(filepath, str):
        result["error"] = "No filepath provided (expected a non-empty string)."
        return result

    path = Path(filepath)

    # --- Edge case: file doesn't exist -------------------------------------
    if not path.exists():
        result["error"] = f"File not found: '{filepath}'. Check the path and try again."
        return result

    # --- Edge case: path exists but is a directory, not a file -------------
    if path.is_dir():
        result["error"] = (
            f"'{filepath}' is a directory, not a file. "
            f"Use list_files() to see the files inside it."
        )
        return result

    if not path.is_file():
        # Covers unusual cases: symlinks to nowhere, sockets, device files, etc.
        result["error"] = f"'{filepath}' exists but is not a regular file."
        return result

    extension = path.suffix.lower()

    # --- Edge case: unsupported file extension ------------------------------
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        result["error"] = (
            f"Unsupported file extension '{extension or '(none)'}' for '{filepath}'. "
            f"Supported types: {supported}."
        )
        return result

    # --- Edge case: empty file (0 bytes) ------------------------------------
    # Still "successful" -- an empty file isn't an error, just worth flagging
    # so the caller doesn't mistake it for a parsing failure.
    if path.stat().st_size == 0:
        result["success"] = True
        result["content"] = ""
        result["warning"] = "File is empty (0 bytes)."
        result["metadata"] = _build_metadata(path, extension, "")
        return result

    try:
        if extension == ".txt":
            # errors="replace" avoids raising on files with unexpected/
            # invalid byte sequences -- we'd rather return best-effort text
            # with replacement characters than fail the whole read.
            content = path.read_text(encoding="utf-8", errors="replace")

        elif extension == ".pdf":
            if PdfReader is None:
                result["error"] = (
                    "Cannot read PDF: the 'pypdf' library is not installed. "
                    "Run: pip install pypdf"
                )
                return result
            try:
                reader = PdfReader(str(path))
            except Exception as e:
                result["error"] = f"Failed to open '{filepath}' as a PDF: {e}"
                return result

            if len(reader.pages) == 0:
                result["error"] = f"'{filepath}' is a PDF with no pages."
                return result

            pages_text = [page.extract_text() or "" for page in reader.pages]
            content = "\n".join(pages_text)

            if not content.strip():
                # PDF opened and parsed without error, but no extractable
                # text -- most likely a scanned/image-only PDF that would
                # need OCR. This is a warning, not a hard failure.
                result["warning"] = (
                    "No extractable text found in this PDF. It may be a "
                    "scanned/image-only document that requires OCR."
                )

        elif extension == ".docx":
            if docx is None:
                result["error"] = (
                    "Cannot read DOCX: the 'python-docx' library is not installed. "
                    "Run: pip install python-docx"
                )
                return result
            try:
                document = docx.Document(str(path))
            except Exception as e:
                result["error"] = f"Failed to open '{filepath}' as a DOCX: {e}"
                return result

            content = "\n".join(p.text for p in document.paragraphs)

            if not content.strip():
                result["warning"] = "This DOCX file has no readable paragraph text."

        result["success"] = True
        result["content"] = content
        result["metadata"] = _build_metadata(path, extension, content)

    except PermissionError:
        result["error"] = f"Permission denied when reading '{filepath}'."
    except Exception as e:
        # Catch-all for anything unexpected (corrupt file, encoding issue
        # we didn't anticipate, etc.) -- still returns a structured result
        # instead of letting the exception propagate to the caller.
        result["error"] = f"Failed to read '{filepath}': {e}"

    return result


def _build_metadata(path: Path, extension: str, content: str) -> dict:
    """Small helper to build the metadata dict, shared by all read_file branches."""
    stat = path.stat()
    return {
        "file_size_bytes": stat.st_size,
        "extension": extension,
        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "num_characters": len(content),
        "num_words": len(content.split()),
    }


# ---------------------------------------------------------------------------
# 2. list_files
# ---------------------------------------------------------------------------
def list_files(directory: str, extension: str = None) -> dict:
    """
    List files in a directory, optionally filtered by extension.

    Handles these edge cases explicitly:
      - empty / non-string directory argument
      - directory does not exist
      - path exists but is a file, not a directory
      - permission errors while scanning
      - directory exists but is empty
      - directory has files, but none match the requested extension
        (these last two are reported as *successful, empty* results with
        a clear message -- not as errors)

    Args:
        directory: Directory path to scan.
        extension: Optional extension filter, e.g. ".pdf" or "pdf".

    Returns:
        dict with keys:
            success (bool)
            directory (str)
            extension (str | None)   - normalized filter that was applied
            files (list[dict])       - each with name, path, size_bytes,
                                        modified_date, extension
            count (int)              - len(files), for convenience
            message (str | None)     - explains empty results; None otherwise
            error (str | None)
    """
    result = {
        "success": False,
        "directory": directory,
        "extension": extension,
        "files": [],
        "count": 0,
        "message": None,
        "error": None,
    }

    # --- Edge case: bad input type / empty string ---------------------------
    if not directory or not isinstance(directory, str):
        result["error"] = "No directory provided (expected a non-empty string)."
        return result

    dir_path = Path(directory)

    # --- Edge case: directory doesn't exist ----------------------------------
    if not dir_path.exists():
        result["error"] = f"Directory not found: '{directory}'."
        return result

    # --- Edge case: path exists but is a file, not a directory --------------
    if not dir_path.is_dir():
        result["error"] = (
            f"'{directory}' is a file, not a directory. Did you mean to call "
            f"read_file() instead?"
        )
        return result

    # Normalize the extension filter so both ".pdf" and "pdf" work the same way.
    normalized_extension = extension
    if normalized_extension and not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"
    result["extension"] = normalized_extension

    try:
        entries = sorted(dir_path.iterdir())
    except PermissionError:
        result["error"] = f"Permission denied when listing '{directory}'."
        return result

    saw_any_file = False
    files_info = []
    for entry in entries:
        if not entry.is_file():
            continue
        saw_any_file = True
        if normalized_extension and entry.suffix.lower() != normalized_extension.lower():
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

    result["success"] = True
    result["files"] = files_info
    result["count"] = len(files_info)

    # --- Edge case: empty results -- explain *why* it's empty ---------------
    if not files_info:
        if not saw_any_file:
            result["message"] = f"Directory '{directory}' contains no files."
        else:
            result["message"] = (
                f"No files with extension '{normalized_extension}' found in "
                f"'{directory}'."
            )

    return result


# ---------------------------------------------------------------------------
# 3. write_file
# ---------------------------------------------------------------------------
def write_file(filepath: str, content: str) -> dict:
    """
    Write text content to a file, creating parent directories if needed.

    Handles these edge cases explicitly:
      - empty / non-string filepath
      - non-string content (coerced to str rather than failing, with a
        warning so the caller knows a conversion happened)
      - destination's parent directory doesn't exist yet (created)
      - target path already exists (overwrite is flagged, not silent)
      - permission errors / disk full / other OS-level write failures

    Args:
        filepath: Destination path.
        content: Text content to write.

    Returns:
        dict with keys: success, filepath, bytes_written, overwritten,
        warning, error
    """
    result = {
        "success": False,
        "filepath": filepath,
        "bytes_written": 0,
        "overwritten": False,
        "warning": None,
        "error": None,
    }

    # --- Edge case: bad input type / empty string ----------------------------
    if not filepath or not isinstance(filepath, str):
        result["error"] = "No filepath provided (expected a non-empty string)."
        return result

    # --- Edge case: non-string content -----------------------------------
    if not isinstance(content, str):
        result["warning"] = (
            f"content was type {type(content).__name__}, not str; converted "
            f"with str() before writing."
        )
        content = str(content)

    try:
        path = Path(filepath)

        # --- Edge case: writing to an existing directory path -------------
        if path.exists() and path.is_dir():
            result["error"] = f"'{filepath}' is an existing directory; cannot write a file there."
            return result

        result["overwritten"] = path.exists()

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        result["success"] = True
        result["bytes_written"] = len(content.encode("utf-8"))

    except PermissionError:
        result["error"] = f"Permission denied when writing to '{filepath}'."
    except OSError as e:
        # Covers disk-full, path-too-long, invalid filename characters, etc.
        result["error"] = f"OS error while writing '{filepath}': {e}"
    except Exception as e:
        result["error"] = f"Failed to write '{filepath}': {e}"

    return result


# ---------------------------------------------------------------------------
# 4. search_in_file
# ---------------------------------------------------------------------------
def search_in_file(filepath: str, keyword: str, context_chars: int = 60) -> dict:
    """
    Case-insensitively search for a keyword inside a file's content and
    return each match along with surrounding context.

    Handles these edge cases explicitly:
      - empty / non-string keyword
      - negative or absurd context_chars values
      - the underlying read_file() call failing (missing file, unsupported
        format, etc.) -- the specific error is passed through unchanged
      - zero matches found -- reported as a *successful* search with an
        explanatory message, not an error, since "not found" is a valid
        and expected outcome of a search

    Args:
        filepath: File to search (uses read_file internally, so PDF/TXT/DOCX
                   are all supported).
        keyword: Keyword or phrase to search for.
        context_chars: Number of characters of context to include on each
                        side of a match. Negative values are treated as 0.

    Returns:
        dict with keys:
            success (bool)
            filepath (str)
            keyword (str)
            match_count (int)
            matches (list[dict])  - each with 'context' and 'position'
            message (str | None)  - explains a zero-match result; None otherwise
            error (str | None)
    """
    result = {
        "success": False,
        "filepath": filepath,
        "keyword": keyword,
        "match_count": 0,
        "matches": [],
        "message": None,
        "error": None,
    }

    # --- Edge case: bad input type / empty keyword ---------------------------
    if not keyword or not isinstance(keyword, str):
        result["error"] = "No keyword provided (expected a non-empty string)."
        return result

    # --- Edge case: invalid context_chars -------------------------------
    if not isinstance(context_chars, int) or context_chars < 0:
        context_chars = 60

    # Delegate file access/parsing to read_file() so PDF/TXT/DOCX and all of
    # its edge-case handling (missing file, unsupported type, empty file,
    # etc.) is handled in exactly one place instead of being duplicated here.
    read_result = read_file(filepath)
    if not read_result["success"]:
        result["error"] = read_result["error"]
        return result

    content = read_result["content"]

    # --- Edge case: file read fine but has no content to search -----------
    if not content:
        result["success"] = True
        result["message"] = f"'{filepath}' has no text content to search."
        return result

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

    # --- Edge case: zero matches -- successful search, just nothing found ---
    if not matches:
        result["message"] = f"No matches found for '{keyword}' in '{filepath}'."

    return result


# ---------------------------------------------------------------------------
# Manual smoke test / example usage.
# Run directly (`python fs_tools.py`) to exercise all four tools, including
# a couple of deliberate edge cases, against the sample resumes/ folder.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Listing .txt files in ./resumes:")
    listing = list_files("resumes", extension=".txt")
    print(" success:", listing["success"], "| count:", listing["count"], "| message:", listing["message"])

    print("\nReading resumes/resume_john_doe.txt:")
    r = read_file("resumes/resume_john_doe.txt")
    print(" success:", r["success"], "| words:", r["metadata"].get("num_words"), "| warning:", r["warning"])

    print("\nSearching for 'python' in resumes/resume_john_doe.txt:")
    s = search_in_file("resumes/resume_john_doe.txt", "python")
    print(" matches found:", s["match_count"])
    for match in s["matches"]:
        print("   ", match["context"])

    print("\n--- Edge case checks ---")

    print("Reading a missing file:")
    r = read_file("resumes/does_not_exist.pdf")
    print(" success:", r["success"], "| error:", r["error"])

    print("Reading an unsupported extension:")
    r = read_file("fs_tools.py")
    print(" success:", r["success"], "| error:", r["error"])

    print("Listing a nonexistent directory:")
    listing = list_files("does_not_exist_dir")
    print(" success:", listing["success"], "| error:", listing["error"])

    print("Searching for a keyword with zero matches:")
    s = search_in_file("resumes/resume_john_doe.txt", "zzz_not_present_zzz")
    print(" success:", s["success"], "| match_count:", s["match_count"], "| message:", s["message"])
