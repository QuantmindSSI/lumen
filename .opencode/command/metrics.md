---
description: Display the full Lumen effectiveness dashboard with SOTA benchmarks, business metrics, and memory health.
---

Display the Lumen Memory Palace real-time effectiveness dashboard.

1. Call the `lumen_dashboard` MCP tool to get the full dashboard data.
2. Present the results in a clean, well-organized markdown format with sections for System Health, Retrieval Effectiveness, Business Impact, Memory Health, and SOTA Comparison.
3. Highlight any anomalies or concerns (e.g., retention rate below 95%, latency over 100ms, server unreachable, forgotten chunks > 0).
4. If the Lumen server is not running, inform the user and suggest running `lumen serve` or `python3 -m lumen.api.server`.
5. Remind the user they can also launch the visual metrics board with `/board` or open http://localhost:8848/dashboard in a browser.
