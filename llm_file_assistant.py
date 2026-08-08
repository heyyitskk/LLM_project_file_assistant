"""
llm_file_assistant.py
----------------------
Wires fs_tools.py up to an LLM using OpenRouter (an OpenAI-compatible API
that can route to Claude, GPT, Llama, Gemini, and many other models with
a single key), so a user can issue natural-language commands like:

    "List all resumes in the resumes folder"
    "Find resumes mentioning Python experience"
    "Create a summary file for resume_john_doe.pdf"
Setup:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY="your-openrouter-key-here"
    python llm_file_assistant.py

Get a free OpenRouter key at: https://openrouter.ai/keys
Browse available models at:   https://openrouter.ai/models
(Many models, e.g. some Llama/Gemini variants, have a free tier.)
"""

import json
import os
import sys

from openai import OpenAI

import fs_tools

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# Any OpenRouter model slug works here, e.g.:
#   "anthropic/claude-sonnet-4.5"
#   "openai/gpt-4o-mini"
#   "google/gemini-2.5-flash"
#   "meta-llama/llama-3.3-70b-instruct:free"   (free tier)
MODEL = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")


# Tool schema (OpenAI function-calling format, used by OpenRouter too)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a resume file (PDF, TXT, or DOCX) and extract its "
                "text content along with metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files in a directory, optionally filtered by "
                "extension (e.g. '.pdf'). Returns file name, size, and "
                "modified date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list.",
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional extension filter, e.g. '.pdf'.",
                    },
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file, creating parent "
                "directories if they don't exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Destination file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": (
                "Case-insensitively search a file's content for a keyword "
                "and return matches with surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "File to search.",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword or phrase to search for.",
                    },
                },
                "required": ["filepath", "keyword"],
            },
        },
    },
]

# Maps tool name -> actual python function in fs_tools.py
TOOL_FUNCTIONS = {
    "read_file": fs_tools.read_file,
    "list_files": fs_tools.list_files,
    "write_file": fs_tools.write_file,
    "search_in_file": fs_tools.search_in_file,
}

SYSTEM_PROMPT = """\
You are a file-system assistant that helps users work with resume files
(PDF, TXT, DOCX) using the tools provided to you.

- Use list_files to discover what's available before assuming file names.
- Use read_file to read resume content.
- Use search_in_file to find keywords/skills across resumes.
- Use write_file to save summaries or new content, always under the
  ./output directory unless the user specifies otherwise.
- Always explain what you found in plain language after calling tools,
  don't just dump raw tool output.
"""


def execute_tool(tool_name: str, tool_input: dict):
    """Dispatch a tool call to the matching function in fs_tools.py."""
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return func(**tool_input)


def run_conversation(client: OpenAI, user_message: str, messages: list) -> list:
    """
    Send a user message, let the model call tools as needed (looping until
    it produces a final text answer), and return the updated message
    history.
    """
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1500,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message

        # Show any text the model produced this turn
        if message.content and message.content.strip():
            print(f"\nAssistant: {message.content}")

        if choice.finish_reason != "tool_calls" or not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            break

        # Model wants to call one or more tools
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tc in message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            print(f"  [tool call] {tool_name}({json.dumps(tool_input)})")
            output = execute_tool(tool_name, tool_input)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(output, default=str),
                }
            )

    return messages


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Please set the OPENROUTER_API_KEY environment variable.")
        print("Get a free key at: https://openrouter.ai/keys")
        sys.exit(1)

    client = OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        # Optional but recommended by OpenRouter for attribution/rankings:
        default_headers={
            "HTTP-Referer": "https://github.com/your-username/llm-file-assistant",
            "X-Title": "LLM File System Assistant",
        },
    )
    messages = []

    print("LLM File System Assistant (type 'quit' to exit)")
    print(f"Provider: {BASE_URL}")
    print(f"Model:    {MODEL}")
    print("Try: 'List all resumes in the resumes folder'")
    print("Try: 'Find resumes mentioning Python experience'")
    print("Try: 'Create a summary file for resume_john_doe.txt'\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        messages = run_conversation(client, user_input, messages)


if __name__ == "__main__":
    main()
