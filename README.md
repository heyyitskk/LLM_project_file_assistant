# LLM-Powered File System Assistant

A file-system assistant that lets you manage and query resume files
(PDF, TXT, DOCX) using natural language, via LLM tool calling.

## Project Structure

```
llm_file_assistant/
├── fs_tools.py            # Part A: core file system tools
├── llm_file_assistant.py  # Part B: LLM integration (tool calling)
├── requirements.txt
├── README.md
├── resumes/                # 7 sample resumes (.txt, .docx, .pdf)
└── output/                 # generated summaries land here
```

## Part A — Core File System Tools (`fs_tools.py`)

| Function | Signature | Purpose |
|---|---|---|
| `read_file` | `(filepath: str) -> dict` | Reads PDF/TXT/DOCX, returns text + metadata |
| `list_files` | `(directory: str, extension: str = None) -> list` | Lists files, optional extension filter |
| `write_file` | `(filepath: str, content: str) -> dict` | Writes content, creates dirs as needed |
| `search_in_file` | `(filepath: str, keyword: str) -> dict` | Case-insensitive keyword search with context |

Every function returns a structured dict/list (never raises on expected
errors like a missing file) so it's safe to hand back to an LLM as a tool
result.

You can test the core tools without any API key:

```bash
python fs_tools.py
```

## Part B — LLM Integration (`llm_file_assistant.py`)

Wires the four tools above into an LLM's tool-calling API via
**[OpenRouter](https://openrouter.ai)** — a single API that can route to
Claude, GPT, Gemini, Llama, and many other models (including free-tier
ones) using the standard OpenAI function-calling format. The model
decides which tool(s) to call based on your natural-language request, the
script executes them against `fs_tools.py`, and feeds the results back to
the model to produce a final answer.

Because OpenRouter is OpenAI-API-compatible, this script also works
unmodified against OpenAI directly, or any other OpenAI-compatible
provider (Groq, Together, a local vLLM server, etc.) — just change
`LLM_BASE_URL` / the API key env var.

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Get a key at [openrouter.ai/keys](https://openrouter.ai/keys) (free
   tier available) and set it:

   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

3. (Optional) Choose a model.
   Browse options at [openrouter.ai/models](https://openrouter.ai/models)
   — look for `:free` suffixed models if you want a $0 option:

   ```bash
   export LLM_MODEL="meta-llama/llama-3.3-70b-instruct:free"
   ```

4. Run the assistant:

   ```bash
   python llm_file_assistant.py
   ```

### Using a different provider entirely

To point this at plain OpenAI or another OpenAI-compatible host instead
of OpenRouter:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
export OPENROUTER_API_KEY="your-openai-key"   # the script reads this var regardless of provider
python llm_file_assistant.py
```

### Example queries

```
You: List all resumes in the resumes folder
You: Read all resumes in the resumes folder
You: Find resumes mentioning Python experience
You: Create a summary file for resume_john_doe.txt
```

On each turn, the console prints which tool(s) the model called and with
what arguments.

## Sample Data

`resumes/` contains 7 dummy resumes covering all three supported formats:

- `.txt`: resume_john_doe, resume_jane_smith, resume_carlos_ramirez, resume_amina_khan, resume_lena_muller
- `.docx`: resume_priya_nair
- `.pdf`: resume_tom_baker

## Error Handling

- Missing files, unsupported extensions, and missing optional dependencies
  (`pypdf` / `python-docx`) are all caught and reported in the `error`
  field of the returned dict rather than raising exceptions.
- `write_file` auto-creates any missing parent directories.

---

## RAG-Based Profile Matching (this milestone)

Builds on the file-system tools above to add a full RAG pipeline that
matches resumes against job descriptions.

```
resume_rag.py           # Part A: chunk, embed, extract metadata, index into ChromaDB
job_matcher.py           # Part B: semantic + hybrid search, scoring, must-have filtering
generate_dataset.py      # one-off script that generated the sample resumes/JDs below
RAG_experiments.ipynb    # chunking comparison, Recall@K, latency analysis
resumes/                 # 32 diverse sample resumes (.txt/.pdf/.docx)
job_descriptions/        # 6 sample job descriptions
chroma_db/                # created on first run — persistent vector store
```

### Design choices

- **Embeddings**: local `sentence-transformers` model (`all-MiniLM-L6-v2`) —
  free, no API key, runs on CPU. Swap `EMBEDDING_MODEL_NAME` in
  `resume_rag.py` for a larger model if you want higher recall at the cost
  of speed.
- **Vector DB**: ChromaDB in persistent local mode (`chroma_db/` folder) —
  no external service/account needed.
- **Chunking**: section-aware — splits each resume on its `SUMMARY` /
  `SKILLS` / `EXPERIENCE` / `EDUCATION` headers so each chunk stays a
  coherent unit, instead of naive fixed-size splitting.
- **Metadata extraction**: rule-based (regex/section parsing) rather than
  an LLM call — deterministic, free, and fast to run over the whole
  dataset. `job_matcher.py --llm` optionally uses your OpenRouter key to
  rewrite the *reasoning* text more naturally at query time, but scoring
  itself never depends on an LLM call.
- **Hybrid search**: semantic similarity (80% weight) + a keyword bonus
  (up to +20) for JD-critical skills literally present in the resume, so
  an exact skill match isn't lost to embedding noise.

### Setup & run

```bash
pip install -r requirements.txt

# 1. Build the vector index (run once, or whenever resumes/ changes)
python resume_rag.py

# 2. Match a job description against the indexed resumes
python job_matcher.py --jd job_descriptions/jd_backend_python.txt

# ...with a must-have filter
python job_matcher.py --jd job_descriptions/jd_backend_python.txt --must-have "5+ years Python"

# ...or pass JD text directly instead of a file
python job_matcher.py --jd-text "Senior Python backend engineer, AWS, Docker"

# ...with LLM-enhanced reasoning text (needs OPENROUTER_API_KEY, see above)
python job_matcher.py --jd job_descriptions/jd_backend_python.txt --llm
```

Output matches the required schema:

```json
{
  "job_description": "...",
  "top_matches": [
    {
      "candidate_name": "John Doe",
      "resume_path": "resumes/resume_john_doe.txt",
      "match_score": 92.4,
      "matched_skills": ["Python", "Docker", "AWS"],
      "relevant_excerpts": ["..."],
      "reasoning": "Matched on skills, experience section(s)..."
    }
  ]
}
```

### Notebook

`RAG_experiments.ipynb` builds the index, then evaluates:

1. Section-aware vs. naive fixed-size chunking
2. Recall@K (K = 1, 3, 5, 10, 15) against a small hand-labeled ground truth
3. Query latency
4. Must-have filtering sanity check

Run it with:

```bash
jupyter notebook RAG_experiments.ipynb
```

### Dataset

`resumes/` has 32 resumes across `.txt`, `.pdf`, and `.docx`, covering 15
different roles (Backend, Data Science, Frontend, DevOps, ML, PM, QA, Data
Engineering, Mobile, Security, UX, Full Stack, Cloud, Marketing, Sales)
with varied experience levels (0–12 years). `job_descriptions/` has 6 JDs
spanning those same role families. Regenerate/expand with:

```bash
python generate_dataset.py
```

---

## Notes

- Generated summaries are written under `output/` by default (the LLM is
  instructed to do this in its system prompt), but you can direct it to
  write anywhere by specifying a path in your request.
