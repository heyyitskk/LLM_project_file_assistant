"""
job_matcher.py
---------------
Part B: Job Matching Engine.

Given a job description, this:
    1. Embeds the JD with the same model used to build the index
    2. Retrieves top-K similar resume chunks via semantic search (ChromaDB)
    3. Boosts/re-ranks with keyword matching for critical skills (hybrid
       search) so an exact skill mention isn't lost to embedding noise
    4. Aggregates chunk-level hits back up to one score per candidate
    5. Scores each candidate 0-100 with reasoning about which sections
       matched
    6. Filters out candidates who fail explicit must-have requirements
       (e.g. "5+ years Python")
    7. Emits the result in the required JSON schema

Usage:
    python job_matcher.py --jd job_descriptions/jd_backend_python.txt
    python job_matcher.py --jd-text "Senior Python backend engineer, 5+ years Python, AWS"
    python job_matcher.py --jd job_descriptions/jd_ml_engineer.txt --must-have "3+ years Python"
    python job_matcher.py --jd job_descriptions/jd_backend_python.txt --llm   # richer reasoning via OpenRouter
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

from resume_rag import EMBEDDING_MODEL_NAME, COLLECTION_NAME

# chromadb / sentence-transformers are imported lazily inside
# match_resumes() so the pure scoring/parsing logic here can be
# unit-tested without those heavier deps installed.

TOP_K = 10

# A short list of common tech/skill keywords we boost on for hybrid search.
# (In a fuller system this could be extracted dynamically from the JD via
# an LLM call; kept static + JD-driven here to stay fast and free.)
CRITICAL_SKILL_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9+.#]*(?:\s[A-Za-z][A-Za-z0-9+.#]*){0,2})\b"
)

MUST_HAVE_YEARS_RE = re.compile(r"(\d+)\+?\s*years?\s*(?:of\s*)?([A-Za-z0-9+.# ]+)", re.IGNORECASE)


def extract_jd_keywords(jd_text: str) -> list:
    """
    Pull a rough set of candidate 'critical skill' keywords out of a JD's
    Requirements section by looking at capitalized/technical tokens.
    Simple heuristic, not exhaustive -- good enough for keyword boosting.
    """
    known_skills = [
        "Python", "Java", "JavaScript", "TypeScript", "React", "Django",
        "FastAPI", "PostgreSQL", "SQL", "Redis", "Docker", "Kubernetes",
        "AWS", "Azure", "Terraform", "TensorFlow", "PyTorch", "Pandas",
        "scikit-learn", "NLP", "Machine Learning", "REST APIs", "CI/CD",
        "Jest", "Bash", "Swift", "Kotlin", "Figma", "Agile", "Spark",
        "Airflow", "Salesforce", "Tableau", "Selenium",
    ]
    found = [s for s in known_skills if s.lower() in jd_text.lower()]
    return found


def parse_must_haves(must_have_args: list) -> list:
    """
    Parse must-have requirement strings like '5+ years Python' into
    structured checks: {"min_years": 5, "skill": "python"}.
    Plain skill strings like 'AWS' are treated as a required-skill check.
    """
    parsed = []
    for req in must_have_args:
        m = MUST_HAVE_YEARS_RE.search(req)
        if m:
            parsed.append(
                {
                    "raw": req,
                    "min_years": int(m.group(1)),
                    "skill": m.group(2).strip().lower(),
                }
            )
        else:
            parsed.append({"raw": req, "min_years": None, "skill": req.strip().lower()})
    return parsed


def candidate_passes_must_haves(metadata: dict, must_haves: list) -> bool:
    if not must_haves:
        return True
    skills_str = metadata.get("skills_str", "").lower()
    exp_years = metadata.get("experience_years", 0)

    for req in must_haves:
        skill_ok = req["skill"] in skills_str if req["skill"] else True
        years_ok = exp_years >= req["min_years"] if req["min_years"] is not None else True
        if not (skill_ok and years_ok):
            return False
    return True


def score_candidate(chunk_hits: list, jd_keywords: list) -> tuple:
    """
    Aggregate chunk-level semantic similarity hits for one candidate into
    a single 0-100 score, plus a reasoning string and matched-skills list.

    chunk_hits: list of dicts {"section", "similarity", "text"}
    """
    # Semantic component: best similarity per section, averaged, scaled to 0-100
    best_by_section = {}
    for hit in chunk_hits:
        sec = hit["section"]
        if sec not in best_by_section or hit["similarity"] > best_by_section[sec]:
            best_by_section[sec] = hit["similarity"]

    semantic_avg = sum(best_by_section.values()) / len(best_by_section)
    semantic_score = max(0, min(100, semantic_avg * 100))

    # Keyword component: bonus for each critical JD skill mentioned in
    # any retrieved chunk text for this candidate
    full_text = " ".join(h["text"] for h in chunk_hits).lower()
    matched_skills = [kw for kw in jd_keywords if kw.lower() in full_text]
    keyword_bonus = min(20, len(matched_skills) * 4)  # up to +20

    final_score = round(min(100, semantic_score * 0.8 + keyword_bonus), 1)

    matched_sections = sorted(best_by_section.keys())
    reasoning = (
        f"Matched on {', '.join(matched_sections).lower()} section(s); "
        f"semantic similarity ~{round(semantic_avg, 2)}. "
    )
    if matched_skills:
        reasoning += f"Confirmed critical skills: {', '.join(matched_skills)}."
    else:
        reasoning += "No exact critical-skill keyword matches, relying on semantic similarity."

    return final_score, reasoning, matched_skills


def enhance_reasoning_with_llm(jd_text: str, candidate_name: str, reasoning: str) -> str:
    """
    Optional: use OpenRouter to rewrite the rule-based reasoning into a
    more natural sentence. Falls back silently to the rule-based reasoning
    if no API key is set or the call fails.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return reasoning

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
        )
        model = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4.5")
        prompt = (
            f"Job description:\n{jd_text}\n\n"
            f"Candidate: {candidate_name}\n"
            f"Raw match notes: {reasoning}\n\n"
            "In one short sentence, explain why this candidate is a good "
            "or partial match for the job, based only on the notes above."
        )
        response = client.chat.completions.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return reasoning


def match_resumes(
    jd_text: str,
    must_haves: list = None,
    db_dir: str = "chroma_db",
    top_k: int = TOP_K,
    use_llm: bool = False,
) -> dict:
    import chromadb
    from sentence_transformers import SentenceTransformer

    must_haves = must_haves or []
    parsed_must_haves = parse_must_haves(must_haves)
    jd_keywords = extract_jd_keywords(jd_text)

    t0 = time.time()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_collection(COLLECTION_NAME)

    jd_embedding = model.encode([jd_text]).tolist()

    # Over-fetch chunks (several chunks can belong to the same candidate)
    # then aggregate to candidate level before taking the true top-K.
    raw_results = collection.query(
        query_embeddings=jd_embedding,
        n_results=min(60, collection.count()),
    )

    candidates = defaultdict(list)  # source_file -> list of chunk hits
    candidate_meta = {}

    docs = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        similarity = 1 - dist  # cosine distance -> similarity
        source = meta["source_file"]
        candidates[source].append(
            {"section": meta["section"], "similarity": similarity, "text": doc}
        )
        candidate_meta[source] = meta

    scored = []
    for source, hits in candidates.items():
        meta = candidate_meta[source]

        if not candidate_passes_must_haves(meta, parsed_must_haves):
            continue

        score, reasoning, matched_skills = score_candidate(hits, jd_keywords)

        if use_llm:
            reasoning = enhance_reasoning_with_llm(jd_text, meta["name"], reasoning)

        relevant_excerpts = [
            h["text"][:200] for h in sorted(hits, key=lambda h: -h["similarity"])[:2]
        ]

        scored.append(
            {
                "candidate_name": meta["name"],
                "resume_path": meta["filepath"],
                "match_score": score,
                "matched_skills": matched_skills,
                "relevant_excerpts": relevant_excerpts,
                "reasoning": reasoning,
            }
        )

    scored.sort(key=lambda c: -c["match_score"])
    top_matches = scored[:top_k]

    elapsed = time.time() - t0

    return {
        "job_description": jd_text,
        "top_matches": top_matches,
        "_meta": {
            "candidates_considered": len(candidates),
            "candidates_after_filter": len(scored),
            "must_have_filters": must_haves,
            "latency_seconds": round(elapsed, 3),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Match resumes to a job description.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jd", help="Path to a job description text file.")
    group.add_argument("--jd-text", help="Job description text given directly.")
    parser.add_argument(
        "--must-have",
        action="append",
        default=[],
        help="Must-have requirement, e.g. '5+ years Python'. Repeatable.",
    )
    parser.add_argument("--db-dir", default="chroma_db")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--llm", action="store_true",
        help="Use OpenRouter to rewrite reasoning text (requires OPENROUTER_API_KEY).",
    )
    args = parser.parse_args()

    if args.jd:
        with open(args.jd, "r", encoding="utf-8") as f:
            jd_text = f.read()
    else:
        jd_text = args.jd_text

    result = match_resumes(
        jd_text,
        must_haves=args.must_have,
        db_dir=args.db_dir,
        top_k=args.top_k,
        use_llm=args.llm,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
