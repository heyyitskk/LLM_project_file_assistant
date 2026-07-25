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

## Notes

- Generated summaries are written under `output/` by default (the LLM is
  instructed to do this in its system prompt), but you can direct it to
  write anywhere by specifying a path in your request.
