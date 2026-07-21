---
description: Quick health check of the Lumen memory palace — active chunks, rooms, embedder status.
---

Run a quick health check on the Lumen Memory Palace.

1. Call the `lumen_status` MCP tool.
2. Summarize the key health indicators in a compact format:
   - Rooms / Loci / Active Chunks
   - Retention rate
   - Embedder model and availability
   - TFC state (e, a, τ, r)
   - Forgetting pipeline status
3. If the server is down, suggest starting it with `lumen serve` or `python3 -m lumen.api.server`.
4. Keep the response concise — one short paragraph or a bullet list.
