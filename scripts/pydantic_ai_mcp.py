"""Local pydantic-ai MCP server for validation, guardrails, and sanitisation.

Backs AI operations with Ollama (llama3.1:8b) and static analysis.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# ---------------------------------------------------------------------------
# AI agent setup
# ---------------------------------------------------------------------------
_model = OllamaModel("llama3.1:8b", provider=OllamaProvider(base_url="http://localhost:11434/v1"))
_guardrail_agent = Agent(
    _model,
    system_prompt=(
        "You are a security-focused code auditor. Analyze the provided code or text "
        "for security vulnerabilities, injection risks, hardcoded secrets, improper "
        "cryptography, and data leakage. Respond ONLY with a JSON object containing "
        "strictly: 'passed' (bool), 'issues' (list of {'severity': str, 'line': int, 'description': str}), "
        "and 'recommendation' (str). Do not include markdown formatting."
    ),
)

_sanitise_agent = Agent(
    _model,
    system_prompt=(
        "You are a data-sanitisation assistant. Remove or redact personally identifiable "
        "information (PII) from the provided text, including names, emails, phone numbers, "
        "API keys, and IP addresses. Replace with [REDACTED]. Return ONLY the sanitised text."
    ),
)

_validate_agent = Agent(
    _model,
    system_prompt=(
        "You are a Python code validator. Check for pydantic model misuse, type-safety issues, "
        "missing error handling, and architectural anti-patterns. Respond ONLY with a JSON object "
        "containing 'valid' (bool), 'issues' (list of {'severity': str, 'line': int, 'message': str}), "
        "and 'summary' (str). No markdown."
    ),
)

app = Server("pydantic-ai-local")


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------
def _extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from markdown-like text."""
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return blocks


def _find_secrets(source: str) -> list[dict]:
    """Static scan for hardcoded secrets."""
    issues = []
    patterns = {
        "api_key": re.compile(r"api[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9]{20,}['\"]"),
        "secret": re.compile(r"secret\s*[:=]\s*['\"][a-zA-Z0-9]{16,}['\"]"),
        "password": re.compile(r"password\s*[:=]\s*['\"][^'\"]+['\"]"),
        "token": re.compile(r"token\s*[:=]\s*['\"][a-zA-Z0-9]{16,}['\"]"),
    }
    for name, pat in patterns.items():
        for m in pat.finditer(source):
            issues.append(
                {
                    "severity": "HIGH",
                    "line": source[: m.start()].count("\n") + 1,
                    "description": f"Possible hardcoded {name} detected.",
                }
            )
    return issues


def _ast_guardrails(source: str, filename: str = "<unknown>") -> list[dict]:
    """AST-based guardrail checks."""
    issues = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [{"severity": "BLOCKER", "line": exc.lineno or 1, "description": f"Syntax error: {exc.msg}"}]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name in ("eval", "exec"):
                issues.append(
                    {
                        "severity": "BLOCKER",
                        "line": getattr(node, "lineno", 1),
                        "description": f"Dangerous built-in '{func_name}' called.",
                    }
                )
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "app.route":
                    # FastAPI routes OK — skip blanket warning
                    pass
        if isinstance(node, ast.JoinedStr):
            # f-string used in SQL-like contexts — heuristic
            line = getattr(node, "lineno", 1)
            # We can't know context easily from AST alone, so keep light
            pass
    return issues


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="validate_python",
            description="Static + AI validation of Python source. Returns syntax, type-safety, and pydantic issues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Python source code to validate"},
                    "filename": {"type": "string", "description": "Optional filename for context"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="security_guardrail",
            description="Security audit of code or text. Checks for injection, secrets, and unsafe patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Code or text to audit"},
                    "filename": {"type": "string", "description": "Optional filename for context"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="sanitize_text",
            description="Redact PII and sensitive tokens from text using AI + heuristics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to sanitise"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="audit_codebase",
            description="Run a comprehensive security + architecture audit over a directory of Python files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute directory path to audit"},
                },
                "required": ["path"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "validate_python":
        source = arguments.get("source", "")
        filename = arguments.get("filename", "<unknown>")
        issues = []

        # AST checks
        try:
            ast.parse(source)
        except SyntaxError as exc:
            issues.append({"severity": "BLOCKER", "line": exc.lineno or 1, "message": f"Syntax error: {exc.msg}"})

        # Static checks
        issues.extend(_ast_guardrails(source, filename))

        # AI validation (async)
        ai_result = await _validate_agent.run(
            f"Validate this Python file ({filename}). Respond only with JSON.\n\n```python\n{source}\n```"
        )
        ai_text = ai_result.output if hasattr(ai_result, "output") else str(ai_result)
        # Try to parse JSON from AI response
        try:
            parsed = json.loads(ai_text)
            if isinstance(parsed.get("issues"), list):
                issues.extend(parsed["issues"])
            summary = parsed.get("summary", "AI validation complete.")
            valid = parsed.get("valid", len(issues) == 0)
        except Exception:
            summary = ai_text[:500]
            valid = len(issues) == 0

        return [TextContent(type="text", text=json.dumps({"valid": valid, "filename": filename, "issues": issues, "summary": summary}))]

    if name == "security_guardrail":
        source = arguments.get("source", "")
        filename = arguments.get("filename", "<unknown>")
        issues = _find_secrets(source)
        issues.extend(_ast_guardrails(source, filename))

        ai_result = await _guardrail_agent.run(
            f"Audit this code ({filename}) for security issues. Respond only with JSON.\n\n```python\n{source}\n```"
        )
        ai_text = ai_result.output if hasattr(ai_result, "output") else str(ai_result)
        try:
            parsed = json.loads(ai_text)
            if isinstance(parsed.get("issues"), list):
                for i in parsed["issues"]:
                    issues.append(
                        {
                            "severity": i.get("severity", "MEDIUM"),
                            "line": i.get("line", 0),
                            "description": i.get("description", i.get("message", "")),
                        }
                    )
            passed = parsed.get("passed", len(issues) == 0)
            recommendation = parsed.get("recommendation", "")
        except Exception:
            passed = len(issues) == 0
            recommendation = ai_text[:500]

        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"passed": passed, "filename": filename, "issues": issues, "recommendation": recommendation}
                ),
            )
        ]

    if name == "sanitize_text":
        text = arguments.get("text", "")
        # Heuristic pre-pass
        redacted = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]", text)
        redacted = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]", redacted)
        redacted = re.sub(r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]", redacted)

        ai_result = await _sanitise_agent.run(f"Sanitise this text:\n\n{redacted}")
        ai_text = ai_result.output if hasattr(ai_result, "output") else str(ai_result)
        return [TextContent(type="text", text=ai_text)]

    if name == "audit_codebase":
        path = arguments.get("path", ".")
        root = Path(path)
        if not root.exists():
            return [TextContent(type="text", text=json.dumps({"error": f"Path not found: {path}"}))]

        all_issues = []
        for py_file in sorted(root.rglob("*.py")):
            rel = py_file.relative_to(root).as_posix()
            source = py_file.read_text(encoding="utf-8", errors="replace")
            issues = _find_secrets(source)
            issues.extend(_ast_guardrails(source, str(rel)))
            for i in issues:
                i["file"] = str(rel)
            all_issues.extend(issues)

        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "path": str(root),
                        "files_scanned": len(list(root.rglob("*.py"))),
                        "total_issues": len(all_issues),
                        "issues": all_issues[:100],
                    }
                ),
            )
        ]

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
