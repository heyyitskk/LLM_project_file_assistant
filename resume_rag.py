"""
resume_rag.py
--------------
Part A: RAG System Setup for resume profile matching.

Pipeline:
    1. Load resumes (PDF/TXT/DOCX) using fs_tools.py (Milestone 1 tools)
    2. Chunk each resume into section-aware chunks (Summary, Skills,
       Experience, Education) so retrieval preserves semantic boundaries
    3. Generate embeddings with a local HuggingFace sentence-transformers
       model (no API key needed, no per-call cost)
    4. Extract structured metadata (Name, Skills, Experience Years,
       Education) per resume
    5. Store chunks + embeddings + metadata in a persistent ChromaDB
       collection

Usage:
    python resume_rag.py                 # (re)builds the vector index
    python resume_rag.py --resumes-dir resumes --db-dir chroma_db
"""

import argparse
import re
import time
from pathlib import Path

import fs_tools

# chromadb / sentence-transformers are imported lazily inside build_index()
# so that chunk_resume() and extract_metadata() can be unit-tested (and
# reused, e.g. by job_matcher.py) without requiring those heavier deps
# to be installed just to exercise the pure text-processing logic.

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, 384-dim
COLLECTION_NAME = "resumes"

SECTION_HEADERS = ["SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION"]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_resume(text: str) -> list:
    """
    Split resume text into section-aware chunks. Falls back to a single
    whole-document chunk if no recognizable section headers are found.

    Returns a list of dicts: {"section": str, "text": str}
    """
    lines = text.splitlines()
    chunks = []
    current_section = "HEADER"
    current_lines = []

    def flush():
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({"section": current_section, "text": content})

    for line in lines:
        stripped = line.strip().upper()
        if stripped in SECTION_HEADERS:
            flush()
            current_section = stripped
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not chunks:
        chunks = [{"section": "FULL", "text": text.strip()}]

    return chunks


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
YEAR_RANGE_RE = re.compile(r"(\d{4})\s*-\s*(Present|\d{4})", re.IGNORECASE)
EXPLICIT_YEARS_RE = re.compile(r"(\d+)\+?\s*years?", re.IGNORECASE)


def extract_metadata(text: str, chunks: list, filename: str) -> dict:
    """
    Extract Name, Skills, Experience Years, and Education from resume text.
    Rule-based (fast, free, deterministic) — relies on the consistent
    section structure produced by chunk_resume.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0] if lines else filename

    section_text = {c["section"]: c["text"] for c in chunks}

    # Skills: comma-separated line in the SKILLS section
    skills_raw = section_text.get("SKILLS", "")
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

    # Education: first line of EDUCATION section
    education = section_text.get("EDUCATION", "").splitlines()
    education = education[0].strip() if education else ""

    # Experience years: prefer an explicit "N years" mention in SUMMARY,
    # else estimate from year ranges found anywhere in the document.
    summary_text = section_text.get("SUMMARY", "")
    explicit = EXPLICIT_YEARS_RE.search(summary_text)
    if explicit:
        years = int(explicit.group(1))
    else:
        years = 0
        current_year = time.localtime().tm_year
        for start, end in YEAR_RANGE_RE.findall(text):
            end_year = current_year if end.lower() == "present" else int(end)
            years = max(years, end_year - int(start))

    return {
        "name": name,
        "skills": skills,
        "skills_str": skills_raw,  # Chroma metadata must be flat scalars
        "experience_years": years,
        "education": education,
        "source_file": filename,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_index(resumes_dir: str = "resumes", db_dir: str = "chroma_db") -> dict:
    """
    Full Part A pipeline: load -> chunk -> embed -> extract metadata ->
    store in ChromaDB. Returns summary stats.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    t0 = time.time()

    # list_files() returns a structured dict (see fs_tools.py), not a bare
    # list -- surface its error clearly rather than silently proceeding
    # with zero files if e.g. the directory path is wrong.
    listing = fs_tools.list_files(resumes_dir)
    if not listing["success"]:
        raise RuntimeError(f"Could not list resumes in '{resumes_dir}': {listing['error']}")
    if listing["count"] == 0:
        raise RuntimeError(f"No files found in '{resumes_dir}/': {listing['message']}")
    files = listing["files"]

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=db_dir)
    # Fresh build each run so re-running the script doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    all_docs, all_metadatas, all_ids = [], [], []
    resumes_processed = 0
    skipped_files = []

    for f in files:
        read_result = fs_tools.read_file(f["path"])
        if not read_result["success"]:
            # Don't let one bad file (corrupt PDF, unsupported type that
            # slipped into the folder, etc.) abort the whole indexing run --
            # skip it, note why, and keep going.
            print(f"  [skip] {f['name']}: {read_result['error']}")
            skipped_files.append({"name": f["name"], "reason": read_result["error"]})
            continue

        text = read_result["content"]

        # --- Edge case: file read "successfully" but has no usable text ---
        # (0-byte file, or a scanned/image-only PDF read_file() already
        # flagged via its "warning" field). Indexing an empty chunk would
        # just waste a slot in the vector DB and never be a useful match,
        # so skip it and report why, same as a hard read failure.
        if not text.strip():
            reason = read_result.get("warning") or "File has no extractable text content."
            print(f"  [skip] {f['name']}: {reason}")
            skipped_files.append({"name": f["name"], "reason": reason})
            continue

        chunks = chunk_resume(text)
        metadata = extract_metadata(text, chunks, f["name"])
        resumes_processed += 1

        for i, chunk in enumerate(chunks):
            all_docs.append(chunk["text"])
            all_metadatas.append(
                {
                    "source_file": f["name"],
                    "filepath": f["path"],
                    "section": chunk["section"],
                    "name": metadata["name"],
                    "skills_str": metadata["skills_str"],
                    "experience_years": metadata["experience_years"],
                    "education": metadata["education"],
                }
            )
            all_ids.append(f"{f['name']}::{chunk['section']}::{i}")

    # --- Edge case: every file failed to read (all unsupported/corrupt) ---
    if resumes_processed == 0:
        raise RuntimeError(
            f"Found {len(files)} file(s) in '{resumes_dir}/' but none could "
            f"be read successfully. First error: "
            f"{skipped_files[0]['reason'] if skipped_files else 'unknown'}"
        )

    embeddings = model.encode(all_docs, show_progress_bar=False).tolist()

    collection.add(
        documents=all_docs,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    elapsed = time.time() - t0
    stats = {
        "resumes_processed": resumes_processed,
        "resumes_skipped": len(skipped_files),
        "skipped_files": skipped_files,
        "total_chunks": len(all_docs),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "db_dir": db_dir,
        "build_time_seconds": round(elapsed, 2),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build the resume RAG index.")
    parser.add_argument("--resumes-dir", default="resumes")
    parser.add_argument("--db-dir", default="chroma_db")
    args = parser.parse_args()

    print(f"Building resume index from {args.resumes_dir}/ ...")
    stats = build_index(args.resumes_dir, args.db_dir)

    print("\nDone.")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
