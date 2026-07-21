# Lumen + OpenCode Integration

## Lumen Dashboard
When Lumen is running, an effectiveness dashboard is available at:
**http://localhost:8848/dashboard**

The dashboard shows:
- Real-time memory palace topology (rooms, loci, chunks)
- Retrieval effectiveness metrics (R@10, nDCG@10, latency)
- Twin-Force Controller state (mnemonic bias, attention temperature)
- Forgetting pipeline health (L1 decay, L2 interference, L3 budget)
- Cost & efficiency comparisons vs. cloud alternatives
- Enterprise readiness score (7-dimension radar)
- Live memory search demo

## Starting the Dashboard
```bash
# Start the Lumen API server (includes dashboard)
python3 -m lumen.api.server
# Or:
lumen serve
```
Then open http://localhost:8848/dashboard in a browser.

Alternatively, view `/metrics` for machine-readable monitoring data at http://localhost:8848/metrics.

## Using Lumen from OpenCode
When chatting with OpenCode, the Lumen MCP tools are available:

- `lumen_store` — persist a memory
- `lumen_search` — search your palace
- `lumen_assemble` — retrieve and assemble context
- `lumen_turn` — log a conversation turn
- `lumen_feedback` — rate memory usefulness
- `lumen_status` — check palace health
- `lumen_dashboard` — display real-time effectiveness dashboard

To check current effectiveness: `lumen_dashboard` (full dashboard) or `lumen_status` (summary). When users ask about performance, SOTA benchmarks, or memory health, call `lumen_dashboard` for a comprehensive view.