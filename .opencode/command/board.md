---
description: Launch the OpenCode real-time metrics board in a browser.
---

Launch the Lumen + OpenCode real-time metrics board.

1. Check if the Lumen API server is running by calling `lumen_status` or curling `http://localhost:8848/metrics`.
2. If the Lumen server is not running, start it first with `python3 -m lumen.api.server` (or `lumen serve`) in the background.
3. Check if a metrics board server is already running on port 8850 by running `curl -s -o /dev/null -w "%{http_code}" http://localhost:8850`.
4. If the board server is not running, start it with `python3 scripts/opencode-board-server.py` in the background.
5. Confirm the board is accessible and tell the user to open **http://localhost:8850** in their browser.
6. Also mention the full Lumen dashboard is available at **http://localhost:8848/dashboard**.
