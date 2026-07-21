#!/usr/bin/env python3
"""OpenCode Metrics Board Server for Lumen.

Serves a lightweight, auto-refreshing real-time dashboard on port 8850.
Fetches live data from the Lumen API (default localhost:8848).

Usage:
    python3 scripts/opencode-board-server.py
    # Then open http://localhost:8850 in your browser.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import urllib.error
import urllib.request
from pathlib import Path

LUMEN_BASE = os.environ.get("LUMEN_API_URL", "http://localhost:8848")
LUMEN_METRICS_URL = f"{LUMEN_BASE}/metrics"
LUMEN_DATA_URL = f"{LUMEN_BASE}/dashboard-data"
PORT = int(os.environ.get("OPENCODE_BOARD_PORT", "8850"))

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lumen + OpenCode — Real-Time Metrics Board</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #111827;
      --border: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #10b981;
      --accent2: #3b82f6;
      --accent3: #f59e0b;
      --accent4: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: var(--font); background: var(--bg); color: var(--text); }
    header {
      background: linear-gradient(135deg, #1e3a5f 0%, #0b0f19 100%);
      border-bottom: 1px solid var(--border);
      padding: 2rem 1.5rem;
      text-align: center;
    }
    header h1 { margin: 0 0 .5rem; font-size: 1.75rem; letter-spacing: -0.02em; }
    header p { margin: 0; color: var(--muted); }
    .badge { display: inline-block; padding: .25rem .75rem; border-radius: 999px; font-size: .75rem; font-weight: 600; margin-left: .5rem; }
    .badge.ok { background: #064e3b; color: #34d399; }
    .badge.warn { background: #451a03; color: #fbbf24; }
    .badge.err { background: #450a0a; color: #f87171; }
    .container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.25rem; }
    .card {
      background: var(--card); border: 1px solid var(--border); border-radius: .75rem;
      padding: 1.25rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,.3);
      transition: transform .15s ease;
    }
    .card:hover { transform: translateY(-2px); }
    .card h2 { margin: 0 0 1rem; font-size: .875rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
    .metric { display: flex; align-items: baseline; gap: .5rem; margin: .5rem 0; }
    .metric-value { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
    .metric-label { font-size: .875rem; color: var(--muted); }
    .metric-delta { font-size: .75rem; font-weight: 600; margin-left: auto; }
    .metric-delta.up { color: var(--accent); }
    .metric-delta.down { color: var(--accent4); }
    .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: .5rem; }
    .status-dot.on { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
    .status-dot.off { background: var(--accent4); }
    .refresh { text-align: center; color: var(--muted); font-size: .75rem; margin-top: 1.5rem; }
    .error { color: var(--accent4); text-align: center; padding: 2rem; }
    .room-tag { background: var(--bg); border: 1px solid var(--border); border-radius: .375rem; padding: .25rem .625rem; font-size: .75rem; display: inline-block; margin: .25rem; }
    .lage-label { color: var(--muted); font-size: .75rem; margin-top: 1rem; margin-bottom: .25rem; }
    .bar-bg { background: var(--bg); border-radius: .25rem; height: .5rem; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: .25rem; transition: width 1s ease; }
    .bar-fill.green { background: var(--accent); }
    .bar-fill.amber { background: var(--accent3); }
    .bar-fill.red { background: var(--accent4); }
    footer { text-align: center; padding: 2rem 1rem; color: var(--muted); font-size: .75rem; border-top: 1px solid var(--border); margin-top: 2rem; }
  </style>
</head>
<body>
  <header>
    <h1>Lumen + OpenCode <span id="liveBadge" class="badge ok">LIVE</span></h1>
    <p>Real-Time Memory Palace Metrics Board</p>
  </header>
  <div class="container">
    <div id="content">
      <p style="text-align:center;color:var(--muted);padding:2rem;">Loading metrics...</p>
    </div>
    <div class="refresh">Auto-refreshes every 3 seconds &middot; Lumen API: <span id="apiUrl">...</span></div>
  </div>
  <footer>
    Lumen v0.1.0-alpha &middot; OpenCode Integration &middot; Press Ctrl+C to stop
  </footer>
  <script>
    const LUMEN_API = "<!--LUMEN_API-->";
    document.getElementById('apiUrl').textContent = LUMEN_API;

    async function fetchMetrics() {
      try {
        const res = await fetch('/api/metrics');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        render(data);
        document.getElementById('liveBadge').className = 'badge ok';
        document.getElementById('liveBadge').textContent = 'LIVE';
      } catch (e) {
        document.getElementById('content').innerHTML = `
          <div class="error">
            <h3>Cannot reach Lumen server</h3>
            <p>Make sure Lumen is running on ${LUMEN_API}</p>
            <code style="display:block;margin-top:1rem;padding:1rem;background:#111827;border-radius:.5rem;">${e.message}</code>
          </div>`;
        document.getElementById('liveBadge').className = 'badge err';
        document.getElementById('liveBadge').textContent = 'OFFLINE';
      }
    }

    function p(obj, path, def='\u2014') {
      return path.split('.').reduce((o,k) => o?.[k], obj) ?? def;
    }

    function render(data) {
      const effect = data.effectiveness || {};
      const biz = data.business || {};
      const tfc = data.tfc || {};
      const palace = data.palace || data.system || {};
      const sys = data.system || {};

      const rooms = palace.rooms ?? sys.rooms ?? '\u2014';
      const loci = palace.loci ?? sys.loci ?? '\u2014';
      const chunks = palace.active_chunks ?? sys.active_chunks ?? '\u2014';
      const retention = (palace.retention_rate_pct ?? sys.retention_rate_pct ?? '\u2014');
      const embedder = sys.embedding_model ?? p(data, 'lumen.embedding_model', '\u2014');
      const embedReady = sys.embedding_model_available ?? false;

      let forgetHtml = '';
      if (data.forgetting) {
        const f = data.forgetting;
        forgetHtml = `
        <div class="card">
          <h2>Forgetting Pipeline</h2>
          <div class="metric" style="justify-content:space-between;">
            <span class="metric-label">L1 Ebbinghaus Decay</span>
            <span class="metric-delta ${f.l1_active ? 'up' : 'down'}">${f.l1_active ? 'ACTIVE' : 'OFF'}</span>
          </div>
          <div class="metric" style="justify-content:space-between;">
            <span class="metric-label">L2 Interference Weakening</span>
            <span class="metric-delta ${f.l2_active ? 'up' : 'down'}">${f.l2_active ? 'ACTIVE' : 'OFF'}</span>
          </div>
          <div class="metric" style="justify-content:space-between;">
            <span class="metric-label">L3 Budget Eviction</span>
            <span class="metric-delta ${f.l3_active ? 'up' : 'down'}">${f.l3_active ? 'ACTIVE' : 'OFF'}</span>
          </div>
        </div>`;
      }

      const budgetPct = data.memory_budget_pct ?? 0;
      const budgetColor = budgetPct < 50 ? 'green' : budgetPct < 80 ? 'amber' : 'red';

      const html = `
      <div class="grid">
        <div class="card">
          <h2><span class="status-dot ${embedReady ? 'on' : 'off'}"></span>System Health</h2>
          <div class="metric"><div class="metric-value">${chunks}</div><div class="metric-label">Active Memories</div></div>
          <div class="metric"><div class="metric-value">${rooms}</div><div class="metric-label">Rooms</div></div>
          <div class="metric"><div class="metric-value">${loci}</div><div class="metric-label">Loci</div></div>
          <div class="metric"><div class="metric-value">${retention}%</div><div class="metric-label">Retention Rate</div></div>
          <div class="metric"><div class="metric-value" style="font-size:1rem;">${embedder}</div><div class="metric-label">Embedder</div></div>
        </div>

        <div class="card">
          <h2>Retrieval Effectiveness</h2>
          <div class="metric"><div class="metric-value">${effect.bm25_r10 ?? '\u2014'}</div><div class="metric-label">BM25 R@10</div></div>
          <div class="metric"><div class="metric-value">${effect.hybrid_r10_projected ?? '\u2014'}</div><div class="metric-label">Hybrid R@10</div></div>
          <div class="metric"><div class="metric-value">${effect.ndcg10 ?? '\u2014'}</div><div class="metric-label">nDCG@10</div></div>
          <div class="metric"><div class="metric-value">${effect.latency_p50_ms ?? '\u2014'} <span style="font-size:.875rem;color:var(--muted);">ms</span></div><div class="metric-label">p50 Latency</div></div>
        </div>

        <div class="card">
          <h2>Business Impact</h2>
          <div class="metric"><div class="metric-value">${biz.data_sovereignty_pct ?? '\u2014'}%</div><div class="metric-label">Data Sovereignty</div></div>
          <div class="metric"><div class="metric-value">$${biz.api_cost_per_query_usd ?? '0.00'}</div><div class="metric-label">Cost / Query</div></div>
          <div class="metric"><div class="metric-value">${biz.gdpr_native_rtbf ? 'YES' : 'NO'}</div><div class="metric-label">GDPR Native RTBF</div></div>
          <div class="metric"><div class="metric-value">${biz.edge_deployable ? 'YES' : 'NO'}</div><div class="metric-label">Edge Deployable</div></div>
        </div>

        <div class="card">
          <h2>Twin-Force Controller</h2>
          <div class="metric"><div class="metric-value">${tfc.e ?? '\u2014'}</div><div class="metric-label">Conservation Bias (e)</div></div>
          <div class="metric"><div class="metric-value">${tfc.a ?? '\u2014'}</div><div class="metric-label">Attention Temp (a)</div></div>
          <div class="metric"><div class="metric-value">${tfc.tau ?? '\u2014'} <span style="font-size:.875rem;color:var(--muted);">d</span></div><div class="metric-label">Temporal Horizon (τ)</div></div>
          <div class="metric"><div class="metric-value">${tfc.r ?? '\u2014'}</div><div class="metric-label">Resolution (r)</div></div>
        </div>

        ${forgetHtml}
      </div>

      <div class="card" style="margin-top:1.5rem;">
        <h2>Memory Budget Usage</h2>
        <div class="bar-bg"><div class="bar-fill ${budgetColor}" style="width:${Math.min(budgetPct, 100)}%"></div></div>
        <div style="display:flex;justify-content:space-between;margin-top:.25rem;font-size:.75rem;color:var(--muted);">
          <span>0%</span>
          <span>${budgetPct.toFixed(1)}%</span>
          <span>100%</span>
        </div>
      </div>

      <div class="card" style="margin-top:1.5rem;">
        <h2>Raw Metrics JSON</h2>
        <pre style="background:var(--bg);padding:1rem;border-radius:.5rem;overflow:auto;font-size:.75rem;color:var(--muted);">${JSON.stringify(data, null, 2)}</pre>
      </div>`;

      document.getElementById('content').innerHTML = html;
    }

    fetchMetrics();
    setInterval(fetchMetrics, 3000);
  </script>
</body>
</html>
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/metrics":
            self._proxy_metrics()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        html = _HTML.replace("<!--LUMEN_API-->", LUMEN_BASE)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _proxy_metrics(self):
        try:
            with urllib.request.urlopen(LUMEN_METRICS_URL, timeout=5) as resp:
                data = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        except Exception as e:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        pass  # Keep output clean


def main():
    with socketserver.TCPServer(("", PORT), _Handler) as httpd:
        print(f"OpenCode Metrics Board running at http://localhost:{PORT}")
        print(f"Proxying Lumen API from {LUMEN_BASE}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
