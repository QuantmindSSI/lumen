# Lumen Integration Guide

This document describes how to integrate **Lumen** (twin-force memory and context framework) into agentic coding platforms:

- [OpenCode](#opencode)

Each platform communicates with Lumen through one of three interfaces:

| Interface | Best For | Platforms |
|-----------|----------|-----------|
| **MCP Server** (stdio) | Native tool calling | OpenCode, Claude Desktop |
| **HTTP API** | Remote / cloud deployments | Custom agents, CI/CD |
| **Python SDK** (`ConversationMemory`) | In-process, LangChain agents | Custom Python agents |

---

## Prerequisites

```bash
cd lumen
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Initialize the palace
lumen init --device generic
```

The FastAPI server must be running for HTTP-based integrations:

```bash
lumen serve --host 0.0.0.0 --port 8848
```

Or use the built-in MCP stdio server (no HTTP required):

```bash
python -m lumen.integrations.mcp_server
```

---

## OpenCode

OpenCode supports [MCP servers](https://modelcontextprotocol.io) natively. The cleanest integration is via the `mcp:` block in `opencode.json`.

### Option A: MCP Stdio Server (Recommended)

Create `.opencode/opencode.json` in your project root:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "lumen-memory": {
      "type": "local",
      "command": ["python", "-m", "lumen.integrations.mcp_server"],
      "enabled": true,
      "env": {
        "LUMEN_DEVICE": "generic",
        "LUMEN_API_KEY": ""
      }
    }
  }
}
```

**What the agent can now do:**

- `lumen_search` — semantic + lexical hybrid search across the memory palace
- `lumen_store` — persist a code insight, decision, or snippet
- `lumen_assemble` — retrieve and assemble context in one call
- `lumen_turn` — log a full conversation turn with implicit feedback
- `lumen_feedback` — log explicit feedback on a retrieved memory
- `lumen_status` — show palace health (rooms, active chunks, TFC state)

### Option B: HTTP API via Remote MCP

If Lumen runs on a remote host or in Docker:

```json
{
  "mcp": {
    "lumen-remote": {
      "type": "remote",
      "url": "http://localhost:8848/mcp",
      "enabled": true,
      "headers": {
        "X-API-Key": "{env:LUMEN_API_KEY}"
      }
    }
  }
}
```

### Option C: OpenCode Skill for Lumen Workflows

Skills teach the agent *when* and *how* to use Lumen. Create:

```
.opencode/skills/lumen-memory/SKILL.md
```

```markdown
---
name: lumen-memory
description: >
  Use when the user asks to remember, recall, persist, or search across
  project decisions, code snippets, architecture notes, or conversation
  history stored in the Lumen memory palace. Trigger keywords:
  "remember", "recall", "what did we decide", "search my notes",
  "persist", "store this", "Lumen".
---

# Lumen Memory Skill

## When to use
- The user wants to save a decision, pattern, or insight for later retrieval.
- The user asks "what did we decide about X?" or "find my note on Y".
- You need to assemble relevant prior context before making a code change.

## Workflow
1. If the user asks to retrieve: call `lumen_search` or `lumen_assemble`.
2. If the user shares an insight to persist: call `lumen_store` with an appropriate `room` (e.g., `architecture`, `decisions`, `snippets`).
3. After a successful turn (user request + your response): call `lumen_turn` to store both messages and log implicit feedback.
4. If the user says a retrieved memory was wrong or unhelpful: call `lumen_feedback` with `was_useful: false`.

## Room naming conventions
| Room | Purpose |
|------|---------|
| `conversations` | Default chat history |
| `decisions` | ADRs, design decisions, trade-offs |
| `architecture` | Component diagrams, boundaries, interfaces |
| `snippets` | Reusable code patterns, one-liners |
| `bugs` | Root-cause analyses, incident post-mortems |
| `onboarding` | Palace Construction wizard output |

## Example
User: "Remember that we decided to use SQLite for the cache layer."
→ Call `lumen_store` with `room="decisions"`, `content="Use SQLite for cache layer because..."`
```

### Option D: Custom OpenCode Agent

For dedicated memory management, define a subagent:

```json
{
  "agent": {
    "lumen-librarian": {
      "description": "Specialist for storing, retrieving, and curating Lumen memories.",
      "mode": "subagent",
      "model": "anthropic/claude-sonnet-4-6",
      "prompt": "You are the Lumen Librarian. You have access to lumen_search, lumen_store, lumen_assemble, lumen_turn, and lumen_feedback. Your job is to help the user manage their long-term memory palace. Always use the correct room names."
    }
  }
}
```

---

## Platform Compatibility

Lumen's MCP server (stdio) and HTTP API are supported by any platform that speaks MCP or JSON/REST, including
OpenCode, Claude Desktop, and any custom agent using the Python SDK. The OpenCode section above provides
configuration examples; the same `python -m lumen.integrations.mcp_server` command works for any MCP-compatible
platform.

---

## Security & Privacy Notes

1. **Sovereign mode**: By default Lumen stores everything locally in SQLite (`~/.lumen/store/lumen.db`). No cloud leakage.
2. **API Key**: Set `LUMEN_API_KEY` and provide it via headers / env vars for multi-user environments.
3. **P2P sharing**: `lumen p2p share` is opt-in; memories do not leave the device unless explicitly shared.
4. **Forgetting**: Lumen supports compliance-driven forgetting (`lumen compliance audit`). Platforms should respect GDPR/CCPA deletion requests by calling the forgetting API.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Database not initialised` | Run `lumen init --device generic` |
| `embedder_fallback` warning | Download embedding model: `lumen init --download-model` |
| MCP server not found | Ensure `lumen` package is in the Python environment used by the platform |
| 413 Request too large | Increase `LUMEN_REQUEST_MAX_SIZE_BYTES` (default 1 MB) |
| CORS errors in browser | Set `LUMEN_ALLOWED_ORIGINS` to your domain |

---

## Next Steps

1. Start the Lumen server: `lumen serve`
2. Configure your platform's MCP / tool registry with the endpoints above.
3. Seed initial memories: `lumen illuminate` (interactive onboarding wizard).
4. Iterate on room taxonomy for your team.

For programmatic Python usage, see `lumen/integrations/langchain.py` and `lumen/conversation.py`.
