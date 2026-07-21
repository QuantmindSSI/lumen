# Devin + Lumen Integration

## Option 1: Devin Custom Tool (Recommended)

Register this tool definition in your Devin workspace settings:

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
    description: Search query or content to store
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
  headers:
    Content-Type: application/json
```

## Option 2: Direct HTTP (Bash tool)

If custom tools are unavailable, use these bash commands in Devin:

```bash
# Start Lumen server in background (one-time)
nohup lumen serve --host 0.0.0.0 --port 8848 > /tmp/lumen.log 2>&1 &

# Search memories
curl -s -X POST http://localhost:8848/search \
  -H "Content-Type: application/json" \
  -d '{"query":"database indexing decision","top_k":5}' | jq .

# Store a memory
curl -s -X POST http://localhost:8848/store \
  -H "Content-Type: application/json" \
  -d '{"content":"Use composite indexes on (user_id, created_at)","room":"decisions"}' | jq .

# Assemble context for current task
curl -s -X POST http://localhost:8848/assemble \
  -H "Content-Type: application/json" \
  -d '{"query":"implement retry logic","top_k":5}' | jq .

# Log a full conversation turn
curl -s -X POST http://localhost:8848/turn \
  -H "Content-Type: application/json" \
  -d '{"user_msg":"Add retry logic","assistant_msg":"Added exponential backoff with jitter.","room":"conversations"}' | jq .

# Check palace status
curl -s http://localhost:8848/status | jq .
```

## Option 3: Devin Docker Image (Advanced)

Embed Lumen directly into your Devin sandbox Dockerfile:

```dockerfile
FROM cognition/devin:latest

RUN pip install "lumen @ git+https://github.com/QuantumindSSI/lumen"
RUN lumen init --device generic

COPY lumen-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

```bash
#!/bin/sh
# lumen-entrypoint.sh
lumen serve --host 0.0.0.0 --port 8848 &
exec /usr/local/bin/devin-start
```
