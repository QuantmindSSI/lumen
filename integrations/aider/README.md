# Aider + Lumen Integration

[Aider](https://aider.chat) is an AI pair programming tool that works with your local git repo. It does not support MCP natively, but you can integrate Lumen via **custom commands** and the **HTTP API**.

## Option 1: Aider Custom Commands

Add to your `~/.aider.conf.yml`:

```yaml
# Aider configuration with Lumen integration

# Custom slash commands
commands:
  # Search Lumen palace
  - name: lumen-search
    description: Search the Lumen memory palace
    run: |
      curl -s -X POST http://localhost:8848/v1/search \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"$args\",\"top_k\":5}"

  # Store a decision in Lumen
  - name: lumen-store
    description: Store a memory in the Lumen palace
    run: |
      curl -s -X POST http://localhost:8848/v1/store \
        -H "Content-Type: application/json" \
        -d "{\"content\":\"$args\",\"room\":\"decisions\"}"

  # Assemble context from Lumen
  - name: lumen-context
    description: Assemble relevant context from Lumen
    run: |
      curl -s -X POST http://localhost:8848/v1/assemble \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"$args\",\"top_k\":5}"
```

Usage in Aider:
```
/lumen-search database indexing decision
/lumen-store We will use composite indexes on (user_id, created_at)
/lumen-context implement retry logic
```

## Option 2: Python Hook (Advanced)

Create `~/.aider/lumen_hook.py`:

```python
"""Aider pre-commit hook that stores file changes in Lumen."""

import subprocess
import sys


def main():
    # Read the diff or commit message from stdin
    content = sys.stdin.read()
    if not content.strip():
        return

    # Store the commit/context in Lumen
    subprocess.run(
        [
            "curl", "-s", "-X", "POST", "http://localhost:8848/v1/store",
            "-H", "Content-Type: application/json",
            "-d", f'{{"content":{repr(content[:2000])},"room":"commits","source_type":"import"}}',
        ],
        capture_output=True,
    )


if __name__ == "__main__":
    main()
```

## Option 3: Shell Alias

Add to your shell profile:

```bash
# Aider with Lumen context injection
aider-lumen() {
  context=$(curl -s -X POST http://localhost:8848/v1/assemble \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"$*\",\"top_k\":3}" | jq -r '.assembled_context // empty')
  if [ -n "$context" ]; then
    aider --message "$context\n\n$*"
  else
    aider --message "$*"
  fi
}
```

Usage:
```bash
aider-lumen "Refactor the database layer to use connection pooling"
```
