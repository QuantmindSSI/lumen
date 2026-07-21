# OpenAI Codex CLI + Lumen Integration

[OpenAI Codex CLI](https://github.com/openai/codex) is an open-source coding agent from OpenAI. It supports **MCP servers** natively.

## Configuration

Add to your `~/.codex/config.json` (or project-local `.codex/config.json`):

```json
{
  "mcp_servers": {
    "lumen-memory": {
      "command": "python",
      "args": ["-m", "lumen.integrations.mcp_server"],
      "env": {
        "LUMEN_DEVICE": "generic",
        "LUMEN_STORE_PATH": "${HOME}/.lumen/store",
        "LUMEN_MODEL_PATH": "${HOME}/.lumen/models"
      }
    }
  }
}
```

Or use the CLI to add it:

```bash
codex mcp add lumen-memory \
  --command "python -m lumen.integrations.mcp_server" \
  --env LUMEN_DEVICE=generic
```

## Usage

Once configured, Codex can call Lumen tools automatically:

```bash
codex "Search my memory palace for decisions about the database layer"
codex "Remember that we decided to use PostgreSQL for the main store"
codex "Assemble relevant context for implementing user authentication"
```

Codex will invoke:
- `lumen_search` for recall queries
- `lumen_store` for persistence requests
- `lumen_assemble` for context-rich tasks
- `lumen_turn` after completing conversation turns

## Full Agent Mode

In **agent mode**, Codex will automatically persist its reasoning and decisions to Lumen:

```bash
codex --mode agent "Refactor the API to use dependency injection"
```

Make sure your `~/.codex/instructions.md` includes:

```markdown
## Memory Management
- Use lumen_store to persist important decisions, patterns, and architectural notes.
- Use lumen_search to recall prior decisions before making changes.
- After completing a task, call lumen_turn to log the conversation.
```
