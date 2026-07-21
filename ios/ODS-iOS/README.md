# ODS Mobile — iOS / iPadOS / macOS

A native Apple-platform port of the ODS private-AI-server experience. Not a 1:1 Docker clone — a ground-up rearchitecture for the constraints and strengths of Apple Silicon.

> **Philosophy:** Your data. Your device. One app.

---

## What It Is (vs What It Isn't)

| ODS Desktop | ODS Mobile |
|---|---|
| Docker + multiple containers | Single native app |
| llama-server + Ollama | `MLXLLM` on-device inference |
| Open WebUI in browser | SwiftUI chat interface |
| Qdrant (vector DB) | GRDB + on-device embeddings |
| n8n workflows | Swift-native Shortcuts-like automations |
| Whisper + Kokoro | iOS `Speech` + `AVSpeechSynthesizer` |
| P2P Beam | `MultipeerConnectivity` / Bluetooth LE |

### Constraints We Embrace
- **No Docker** → Swift Package Manager
- **No shell installers** → App Store / TestFlight
- **RAM ceiling (6-24 GB)** → Quantized models (Q4_K_M, 2B-9B range)
- **Neural Engine** → CoreML + MLX acceleration
- **SQLite is native** → Lumen palace runs with zero porting friction

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SwiftUI (iOS 17+)                        │
│  ┌─────────┐  ┌─────────────┐  ┌─────────┐  ┌──────────┐  │
│  │  Chat   │  │Memory Palace│  │Dashboard│  │Settings  │  │
│  └────┬────┘  └──────┬──────┘  └────┬────┘  └────┬─────┘  │
└───────┼──────────────┼──────────────┼────────────┼────────┘
        │              │              │            │
        ▼              ▼              ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                      ODSApp Core                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Inference Engine│  │ Lumen Palace │  │ Voice Service  │ │
│  │ (MLXLLM)        │  │ (GRDB/SQLite)│  │ (Speech/AV)    │ │
│  └─────────────────┘  └──────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Developer)

### Prerequisites
- macOS 14+ with Xcode 16
- iOS 17+ device or simulator
- Apple Silicon Mac (M1+) for best MLX performance

### Build
```bash
cd ios/ODS-iOS
open Package.swift  # Opens in Xcode
# Or:
swift build
```

### Run
1. Select target: iPhone 15 Pro (simulator) or connected device
2. Build & Run (`Cmd+R`)
3. Download a model on first launch (2B-4B quantized)

---

## Model Zoo (On-Device)

| Model | Size | Quant | RAM | Speed | Quality |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | Q4_K_M | ~400 MB | Very Fast | Basic |
| Qwen3-1.7B | 1.7B | Q4_K_M | ~1.1 GB | Fast | Good |
| Qwen3-4B | 4B | Q4_K_M | ~2.5 GB | Moderate | Very Good |
| Phi-4 Mini | 3.8B | Q4_K_M | ~2.3 GB | Moderate | Very Good |
| Gemma-4-2B | 2B | Q4_K_M | ~1.3 GB | Fast | Good |

**Recommendation:** Start with Qwen3-1.7B for iPhone 15/16 (8 GB RAM). Use Qwen3-4B on iPhone 16 Pro (8 GB) or iPad Pro (16 GB).

---

## Lumen Palace on iOS

Lumen's core memory model ports directly to iOS because it uses **SQLite** (+ optional USearch which we replace with native vector ops):

- **Rooms, Loci, Chunks** → GRDB tables
- **BM25** → SQLite FTS5 (built-in since iOS 3.0)
- **Dense retrieval** → CoreML-optimized embedding model + cosine similarity in Swift
- **Forgetting (L1/L2/L3)** → Scheduled background tasks (`BGTaskScheduler`)

The palace file is just a `.db` in the app's documents directory. Back up with iCloud.

---

## Features

### ✅ Implemented / Planned
- [x] SwiftUI chat interface with streaming text
- [x] MLX on-device inference
- [x] Lumen Memory Palace (rooms, loci, chunks, search)
- [x] iOS-native speech-to-text
- [x] iOS-native text-to-speech
- [x] Model download / management
- [x] Token budget display
- [x] Settings (temperature, context window, device profile)
- [ ] Agent workflows (planned v0.2)
- [ ] P2P memory sync via MultipeerConnectivity (planned v0.3)
- [ ] Image generation (planned v0.3 via Stable Diffusion CoreML)
- [ ] macOS native app ( catalyst / AppKit )

---

## File Structure

```
ios/ODS-iOS/
├── Package.swift
├── README.md
├── Sources/
│   └── ODSApp/
│       ├── ODSApp.swift              # App entry
│       ├── Models/
│       │   ├── ChatMessage.swift
│       │   └── MemoryModels.swift    # Port of Lumen schema
│       ├── Services/
│       │   ├── InferenceService.swift    # MLXLLM wrapper
│       │   ├── LumenPalaceService.swift  # Memory CRUD + search
│       │   └── VoiceService.swift        # STT + TTS
│       ├── Views/
│       │   ├── ChatView.swift
│       │   ├── MemoryPalaceView.swift
│       │   ├── DashboardView.swift
│       │   └── SettingsView.swift
│       └── ViewModels/
│           └── ChatViewModel.swift
└── Tests/
    └── ODSAppTests/
```

---

## License

Apache-2.0 (same as ODS and Lumen).

---

*Built by the resistance — now pocket-sized.*
