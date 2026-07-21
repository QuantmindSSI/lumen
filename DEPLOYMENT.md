# Lumen Production Deployment Guide

**Version:** 1.0.0  
**Target:** v0.1.0-alpha  
**Platforms:** Raspberry Pi 5, Jetson Orin Nano, Generic x86_64, ARM64 SBCs

---

## 1. Pre-Deployment Checklist

### 1.1 Hardware Requirements

| Platform | RAM | Storage | OS | Network |
|---|---|---|---|---|
| Raspberry Pi 5 | 4 GB | 32 GB SD / NVMe | Raspberry Pi OS 64-bit | Optional LAN |
| Jetson Orin Nano | 8 GB | 128 GB NVMe | JetPack 6.x | Optional LAN |
| Generic x86_64 | 8 GB | 50 GB SSD | Ubuntu 22.04+ / Debian 12+ | Optional LAN |
| Orange Pi 5 | 8 GB | 64 GB eMMC | Armbian / Ubuntu | Optional LAN |

### 1.2 Environment Prerequisites

```bash
# Python 3.10+
python3 --version

# pip and venv
python3 -m pip --version
python3 -m venv --help

# SQLite with WAL support (default on most systems)
python3 -c "import sqlite3; print(sqlite3.sqlite_version)"  # >= 3.35.0

# Basic build tools (for llama-cpp-python and numba)
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev

# (Optional) Disable swap on SD/eMMC to reduce flash wear
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile
```

---

## 2. Installation

### 2.1 Standard Installation (Recommended)

```bash
# Create a dedicated virtual environment
python3 -m venv ~/.venvs/lumen
source ~/.venvs/lumen/bin/activate

# Install Lumen from PyPI (when published)
pip install lumen

# Or install from source
git clone https://github.com/QuantumindSSI/lumen.git
cd lumen/lumen
pip install -e ".[dev]"
```

### 2.2 Edge-Optimized Installation

```bash
# For Raspberry Pi 5 (low-RAM, flash-wear conscious)
pip install lumen \
  --no-binary :all: \
  --prefer-binary onnxruntime

# Download spaCy model for NER
python -m spacy download en_core_web_sm

# Optional: local LLM support for sleep-phase consolidation
pip install llama-cpp-python
```

### 2.3 Docker Deployment

```bash
# Build
docker build -t lumen:latest ./lumen

# Run (production)
docker run -d \
  --name lumen \
  -p 8848:8848 \
  -v ~/.lumen:/root/.lumen \
  -e LUMEN_DEVICE=generic \
  -e LUMEN_LOG_LEVEL=info \
  --restart unless-stopped \
  lumen:latest

# Or use docker-compose
docker compose -f lumen/docker-compose.yml up -d
```

**docker-compose.yml** (provided in repo):
```yaml
version: "3.8"
services:
  lumen:
    build: ./lumen
    ports:
      - "8848:8848"
    volumes:
      - ~/.lumen:/root/.lumen
    environment:
      - LUMEN_DEVICE=generic
      - LUMEN_LOG_LEVEL=info
      - LUMEN_API_HOST=0.0.0.0
      - LUMEN_API_PORT=8848
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8848/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
```

---

## 3. Configuration

### 3.1 Device Profiles

Create `~/.lumen/config.toml`:

```toml
# Raspberry Pi 5
[device.rpi5]
context_budget = 2048
memory_limit_mb = 300
embedding_model = "bge-small-en-v1.5"
embedding_dims = 384
vector_index = "sqlite-vec"
enable_hnsw = false
enable_kuzu = false
consolidation_cpu_percent = 5
scheduler_granularity = 300

# Jetson Orin Nano
[device.jetson-orin]
context_budget = 4096
memory_limit_mb = 800
vector_index = "usearch"
enable_kuzu = true
enable_frqad = true
enable_local_llm = true

# Generic x86 / server
[device.generic]
context_budget = 4096
memory_limit_mb = 1024
vector_index = "usearch"
enable_kuzu = true
enable_frqad = true
```

### 3.2 Environment Variables

```bash
# Required
export LUMEN_DEVICE="generic"          # rpi5 | jetson-orin | orange-pi | generic
export LUMEN_CONTEXT_BUDGET="4096"     # tokens
export LUMEN_MEMORY_LIMIT="500mb"      # RAM cap

# Twin-Force Controller defaults
export LUMEN_TFC_E="0.5"               # conservation bias (0-1)
export LUMEN_TFC_A="0.5"               # attentional temperature (0-1)
export LUMEN_TFC_TAU="7d"              # temporal horizon
export LUMEN_TFC_R="3"                 # resolution level (0-5)

# Sovereign mode
export LUMEN_SOVEREIGN="true"          # block all external API calls
export LUMEN_LOG_LEVEL="info"          # debug | info | warning | error
```

### 3.3 SQLite Tuning for Production

```bash
sqlite3 ~/.lumen/store/lumen.db <<EOF
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA mmap_size=268435456;  -- 256 MB
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-32768;    -- 32 MB page cache
EOF
```

---

## 4. Health Checks & Monitoring

### 4.1 API Health Endpoint

```bash
# Default health check
curl http://localhost:8848/health

# Expected response:
{
  "status": "ok",
  "device": "generic",
  "palace_rooms": 12,
  "active_chunks": 2341,
  "context_usage": "3.2K / 4K tokens",
  "tfc": {"e": 0.62, "a": 0.44, "tau": "7d", "r": 3}
}
```

### 4.2 CLI Status

```bash
lumen status

# Output:
#   ⚡ Twin-Force Controller: ACTIVE
#     Memory Palace: 12 rooms, 147 loci, 2,341 chunks
#     Context Window: 3.2K tokens (budget: 4K)
#     Last consolidation: 3m ago
#     Forgetting queue: 18 items pending
```

### 4.3 Log Monitoring

```bash
# Structured JSON logs
tail -f ~/.lumen/logs/audit.jsonl | jq .

# Log format:
# {"timestamp":"2026-07-20T12:00:00Z","component":"LUMEN:FORCE:MNEMONIC",
#  "level":"info","event":"CONSOLIDATE","room":"preferences","new_chunks":12}
```

---

## 5. Security Hardening

### 5.1 File Permissions

```bash
# Secure the Lumen home directory
chmod 700 ~/.lumen
chmod 600 ~/.lumen/config.toml
chmod 600 ~/.lumen/store/lumen.db
```

### 5.2 Systemd Service (Linux)

Create `/etc/systemd/system/lumen.service`:

```ini
[Unit]
Description=Lumen Memory API Server
After=network.target

[Service]
Type=simple
User=lumen
Group=lumen
WorkingDirectory=/home/lumen
Environment="LUMEN_DEVICE=generic"
Environment="LUMEN_LOG_LEVEL=info"
Environment="LUMEN_SOVEREIGN=true"
Environment="PATH=/home/lumen/.venvs/lumen/bin"
ExecStart=/home/lumen/.venvs/lumen/bin/uvicorn lumen.api.server:app --host 0.0.0.0 --port 8848
Restart=always
RestartSec=5
MemoryMax=900M
CPUQuota=80%
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/lumen/.lumen

[Install]
WantedBy=multi-user.target
```

```bash
sudo useradd -r -s /bin/false lumen
sudo mkdir -p /home/lumen/.lumen
sudo chown -R lumen:lumen /home/lumen/.lumen
sudo systemctl daemon-reload
sudo systemctl enable --now lumen
```

### 5.3 Reverse Proxy (Nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name lumen.local;

    ssl_certificate /etc/ssl/certs/lumen.crt;
    ssl_certificate_key /etc/ssl/private/lumen.key;

    location / {
        proxy_pass http://127.0.0.1:8848;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

---

## 6. Backup & Disaster Recovery

### 6.1 What to Back Up

```bash
~/.lumen/config.toml          # Device profile and TFC settings
~/.lumen/palace.toml          # Room topology
~/.lumen/user.toml            # Per-user weights
~/.lumen/store/lumen.db       # SQLite database (WAL mode)
~/.lumen/store/lumen.db-wal   # WAL file (must be backed up together)
~/.lumen/store/vectors.usearch # USearch index (if used)
~/.lumen/logs/audit.jsonl     # Compliance audit trail
```

### 6.2 Automated Backup Script

```bash
#!/bin/bash
# /usr/local/bin/lumen-backup.sh

BACKUP_DIR="/backup/lumen/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Checkpoint WAL before backup
sqlite3 ~/.lumen/store/lumen.db "PRAGMA wal_checkpoint(TRUNCATE);"

# Atomic copy
cp ~/.lumen/config.toml "$BACKUP_DIR/"
cp ~/.lumen/palace.toml "$BACKUP_DIR/"
cp ~/.lumen/store/lumen.db "$BACKUP_DIR/"
cp -r ~/.lumen/logs "$BACKUP_DIR/"

# Compress
tar czf "${BACKUP_DIR}.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

# Retention: keep last 14 days
find /backup/lumen -name "*.tar.gz" -mtime +14 -delete
```

---

## 7. Scaling & Performance

### 7.1 Expected Footprints

| Platform | Package Size | Runtime RAM | Storage per 10k Memories |
|---|---|---|---|
| RPi5 | ~180 MB | ~90 MB | ~35 MB |
| Jetson Orin | ~250 MB | ~180 MB | ~35 MB |
| x86_64 | ~300 MB | ~200 MB | ~35 MB |

### 7.2 Performance Tuning

```bash
# Disable CPU frequency scaling for latency
sudo cpupower frequency-set -g performance

# Increase file descriptor limits
ulimit -n 65536

# Tune kernel for SQLite WAL
sudo sysctl -w vm.swappiness=1
sudo sysctl -w vm.dirty_ratio=5
sudo sysctl -w vm.dirty_background_ratio=2
```

### 7.3 Monitoring Metrics

Key metrics to alert on:
- `lumen_memory_usage_mb` > 80% of `LUMEN_MEMORY_LIMIT`
- `lumen_context_usage_ratio` > 0.95
- `lumen_forgetting_queue_depth` > 1000
- `lumen_retrieval_latency_p99` > 500ms
- `lumen_consolidation_failures` > 0 in 1h

---

## 8. Upgrade Procedures

### 8.1 In-Place Upgrade

```bash
# 1. Backup
lumen-backup.sh

# 2. Stop service
sudo systemctl stop lumen

# 3. Upgrade
source ~/.venvs/lumen/bin/activate
pip install --upgrade lumen

# 4. Run migrations
lumen migrate

# 5. Restart
sudo systemctl start lumen

# 6. Verify
lumen status
curl http://localhost:8848/health
```

### 8.2 Rollback

```bash
# Restore from backup
sudo systemctl stop lumen
sqlite3 ~/.lumen/store/lumen.db ".restore '${BACKUP_DIR}/lumen.db'"
# Reinstall previous version
pip install lumen==$PREVIOUS_VERSION
sudo systemctl start lumen
```

---

## 9. Troubleshooting

### 9.1 Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `sqlite3.OperationalError: database is locked` | WAL checkpoint stuck | `PRAGMA wal_checkpoint(TRUNCATE);` |
| High RAM usage | USearch index loaded into RAM | Switch to `vector_index = "sqlite-vec"` |
| Slow retrieval | No FTS5 index | Run `lumen palace rebuild` |
| `SovereignViolation` error | External API blocked | Set `LUMEN_SOVEREIGN=false` or use offline embedder |
| SD card wear | Too many random writes | Enable WAL + `synchronous=NORMAL` |

### 9.2 Diagnostic Commands

```bash
# Full system report
lumen status --verbose

# Check database integrity
sqlite3 ~/.lumen/store/lumen.db "PRAGMA integrity_check;"

# View recent errors
lumen compliance audit --level error --last 1h

# Test retrieval latency
lumen context assemble --benchmark --query "test query"
```

---

## 10. Support & Community

- **Documentation:** https://docs.lumen.ai
- **Issues:** https://github.com/QuantumindSSI/lumen/issues
- **Discussions:** https://github.com/QuantumindSSI/lumen/discussions
- **Security:** security@lumen.ai
- **Matrix:** #lumen:matrix.org

---

*Last updated: 2026-07-20*  
*Maintainer: Lumen Engineering Team*
