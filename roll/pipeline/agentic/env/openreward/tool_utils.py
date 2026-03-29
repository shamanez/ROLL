"""Reusable utilities for OpenReward tool call parsing and system prompt building.

Supports Qwen3.5's **native** tool-call format (``<function=name><parameter=key>...``)
as well as the JSON fallback (``{"name": ..., "arguments": {...}}``).
"""
import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Hardcoded clean tool definitions for kanishk/EndlessTerminals.
# TODO: Delete this once the model reliably distinguishes tools from their
#       raw OpenReward schemas alone. These exist solely to give the base
#       Qwen3.5-2B clearer guidance (especially create_file vs str_replace).
# ---------------------------------------------------------------------------

QWEN_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to run in container",
                    },
                    "description": {
                        "type": "string",
                        "description": "Why I'm running this command",
                    },
                },
                "required": ["command", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": (
                "Create a NEW file at path with the given file_text content. "
                "Use this when the file does not exist yet. "
                "Do NOT use str_replace to create files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to create",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                    "description": {
                        "type": "string",
                        "description": "Why I'm creating this file",
                    },
                },
                "required": ["path", "file_text", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Edit an EXISTING file by replacing an exact substring. "
                "Requires old_str (the exact text to find) and new_str (the replacement). "
                "The file must already exist. Do NOT use this to create new files — use create_file instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "String to replace (must be unique in file)",
                    },
                    "new_str": {
                        "type": "string",
                        "default": "",
                        "description": "String to replace with (empty to delete)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Why I'm making this edit",
                    },
                },
                "required": ["path", "old_str", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Call this when you have fully completed the task and all requirements are met. "
                "Do not call any other tools after this."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": "View file contents or directory listings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory",
                    },
                    "view_range": {
                        "anyOf": [
                            {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "prefixItems": [{"type": "integer"}, {"type": "integer"}],
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "description": (
                            "Optional line range for text files. "
                            "Format: [start_line, end_line] where lines are indexed starting at 1. "
                            "Use [start_line, -1] to view from start_line to end."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Why I need to view this",
                    },
                },
                "required": ["path", "description"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool-spec conversion: OpenReward spec → Qwen chat-template dict
# ---------------------------------------------------------------------------

def openreward_spec_to_qwen_tool(spec: Any) -> Dict[str, Any]:
    """Convert an OpenReward tool spec to the dict format expected by
    ``tokenizer.apply_chat_template(tools=[...])``.

    Falls back to the hardcoded QWEN_TOOLS definition if the tool name
    is known, otherwise converts the raw spec directly.
    """
    hardcoded = {t["function"]["name"]: t for t in QWEN_TOOLS}
    if spec.name in hardcoded:
        return hardcoded[spec.name]
    # Fallback for unknown tools
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema or {"type": "object", "properties": {}, "required": []},
        },
    }


# ---------------------------------------------------------------------------
# Tool-call parsing: Qwen native XML + JSON fallback
# ---------------------------------------------------------------------------

# Regex for Qwen3.5 native format: <function=name>...<parameter=key>\nvalue\n</parameter>...
_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[^>]+)>(?P<body>.*?)</function>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<key>[^>]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from model output.

    Supports two formats:

    1. **Qwen3.5 native** (preferred)::

        <tool_call>
        <function=bash>
        <parameter=command>ls</parameter>
        <parameter=description>list files</parameter>
        </function>
        </tool_call>

    2. **JSON fallback** (cookbook style)::

        <tool_call>
        {"name": "bash", "arguments": {"command": "ls", "description": "list files"}}
        </tool_call>

    Args:
        text: Raw model output.

    Returns:
        ``None`` if no ``<tool_call>`` found.
        ``{"type": "success", "name": str, "arguments": dict}`` on success.
        ``{"type": "error", "error": str}`` on parse failure.
    """
    start_tag = "<tool_call>"
    si = text.find(start_tag)
    if si == -1:
        return None

    end_tag = "</tool_call>"
    ei = text.find(end_tag, si)
    inner = text[si + len(start_tag):ei].strip() if ei != -1 else text[si + len(start_tag):].strip()

    if not inner:
        return {"type": "error", "error": "empty tool call block"}

    # --- Try Qwen native XML format first ---
    func_match = _FUNCTION_RE.search(inner)
    if func_match:
        name = func_match.group("name").strip()
        body = func_match.group("body")
        arguments: Dict[str, str] = {}
        for param_match in _PARAMETER_RE.finditer(body):
            key = param_match.group("key").strip()
            value = param_match.group("value").strip()
            arguments[key] = value
        return {"type": "success", "name": name, "arguments": arguments}

    # --- Fallback: JSON format ---
    try:
        data = json.loads(inner)
        if not isinstance(data, dict):
            return {"type": "error", "error": f"parsed value is not a dict: {type(data).__name__}"}
        name = data.get("name")
        if not name:
            return {"type": "error", "error": "missing 'name' field in tool call"}
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            return {"type": "error", "error": f"arguments is not a dict: {type(args).__name__}"}
        return {"type": "success", "name": name, "arguments": args}
    except (json.JSONDecodeError, KeyError) as exc:
        return {"type": "error", "error": str(exc)}


def reduce_rewards(rewards: List[float], method: str) -> float:
    """Reduce a list of per-step rewards to a single scalar."""
    if not rewards:
        return 0.0
    if method == "sum":
        return sum(rewards)
    elif method == "mean":
        return sum(rewards) / len(rewards)
    elif method == "max":
        return max(rewards)
    elif method == "min":
        return min(rewards)
    raise ValueError(f"Unknown reward reduction method: {method!r}")
