# Lumen Integration Guide

This document describes how to integrate **Lumen** (twin-force memory and context framework) into agentic coding platforms:

- [OpenCode](#opencode)
- [GitHub Copilot](#github-copilot)
- [Devin](#devin)

Each platform communicates with Lumen through one of three interfaces:

| Interface | Best For | Platforms |
|-----------|----------|-----------|
| **MCP Server** (stdio) | Native tool calling | OpenCode, Copilot, Claude Desktop |
| **HTTP API** | Remote / cloud deployments | Devin, custom agents, CI/CD |
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

## GitHub Copilot

GitHub Copilot Chat (VS Code Insiders / VS Code 1.99+) supports MCP servers via the `@mcp` participant or direct tool registration.

### VS Code Settings (MCP)

Add to your VS Code `settings.json`:

```json
{
  "github.copilot.chat.mcp.servers": [
    {
      "name": "lumen-memory",
      "command": "python",
      "args": ["-m", "lumen.integrations.mcp_server"],
      "env": {
        "LUMEN_DEVICE": "generic"
      }
    }
  ]
}
```

Restart VS Code. In Copilot Chat you can now ask:

> "@lumen-memory search for our decision on database indexing"

> "@lumen-memory store this in room decisions: We will use composite indexes on (user_id, created_at)"

### Copilot Extension (Programmatic)

For deeper VS Code integration, register Lumen as a custom participant in an extension:

```typescript
// src/extension.ts
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  const participant = vscode.chat.createChatParticipant('copilot.lumen', async (request, context, response, token) => {
    const query = request.prompt;
    // Call Lumen HTTP API
    const res = await fetch('http://localhost:8848/assemble', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 5 })
    });
    const data = await res.json();
    response.markdown(data.assembled_context || 'No relevant memories found.');
  });
  participant.iconPath = new vscode.ThemeIcon('database');
}
```

---

## Devin

Devin supports custom tools via the **Devin Tools API** and arbitrary HTTP calls in its bash/web tools.

### Option A: Devin Custom Tool (Recommended)

Register Lumen as a custom tool in your Devin workspace settings or organization profile.

**Tool Definition:**

```yaml
name: lumen_memory
description: >
  Search, store, and manage long-term memory in the Lumen palace.
  Use this when you need to recall prior decisions, persist new insights,
  or assemble relevant context before writing code.
parameters:
  - name: action
    type: string
    enum: [search, store, assemble, turn, feedback, status]
    required: true
  - name: query
    type: string
    description: Search query or content to store (for search/store/assemble/turn)
  - name: room
    type: string
    description: Room name (e.g., decisions, architecture, snippets)
    default: conversations
  - name: content
    type: string
    description: Content to store (for store action)
  - name: user_msg
    type: string
    description: User message (for turn action)
  - name: assistant_msg
    type: string
    description: Assistant message (for turn action)
  - name: chunk_id
    type: integer
    description: Chunk ID for feedback
  - name: was_useful
    type: boolean
    description: For feedback action
endpoint:
  type: http
  base_url: http://host.docker.internal:8848
  # If Lumen runs inside Devin's environment, use localhost:8848
  headers:
    Content-Type: application/json
```

**Devin can then call:**

```
Use lumen_memory with action="search", query="SQLite cache decision", room="decisions"
Use lumen_memory with action="store", room="architecture", content="We adopted hexagonal ports-and-adapters..."
Use lumen_memory with action="turn", user_msg="Add retry logic", assistant_msg="Added exponential backoff..."
```

### Option B: Direct HTTP from Devin

If custom tools are unavailable, Devin can call the Lumen API via `curl` in bash:

```bash
# Search
curl -s -X POST http://localhost:8848/search \
  -H "Content-Type: application/json" \
  -d '{"query":"database indexing decision","top_k":5}'

# Store
curl -s -X POST http://localhost:8848/store \
  -H "Content-Type: application/json" \
  -d '{"content":"Use composite indexes on (user_id, created_at)","room":"decisions"}'

# Assemble context for current task
curl -s -X POST http://localhost:8848/assemble \
  -H "Content-Type: application/json" \
  -d '{"query":"implement retry logic","top_k":5}'
```

### Option C: Devin + Lumen in Same Container

If you run Devin with a custom Docker image, embed Lumen and start it on boot:

```dockerfile
FROM cognition/devin:latest

RUN pip install "lumen @ git+https://github.com/QuantumindSSI/lumen"
RUN lumen init --device generic

COPY lumen-entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

```bash
#!/bin/bash
# lumen-entrypoint.sh
lumen serve --host 0.0.0.0 --port 8848 &
exec /usr/local/bin/devin-start
```

---

## Platform Comparison

| Feature | OpenCode | GitHub Copilot | Devin |
|---------|----------|----------------|-------|
| **Native MCP** | Yes | Yes (1.99+) | Via custom tools |
| **In-process SDK** | Yes (Python) | No | No |
| **HTTP API** | Yes | Yes (custom ext) | Yes |
| **Skill/Agent scoping** | `.opencode/skills/` | Extension manifest | Workspace tools |
| **Multi-tenant** | `user_id` per request | Per-workspace | Per-session |
| **Feedback loop** | `lumen_feedback` + implicit | Explicit via chat | Tool call result |

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

For programmatic Python usage, see `lumen/integrations/langchain.py` and `lumen/lumen/conversation.py`.
