# Proprietary R&D Brainstorm: Agentic Memory Architecture for Sovereign AI

## Status: Internal Strategy Document | Classification: Confidential

---

## Executive Summary

We are at an inflection point. In the last 18 months (2024-2026), agentic memory has gone from an afterthought in LLM research to one of the most active subfields. Multiple convergent trends — sovereign AI, edge deployment, personalised agents, and the economic pressure of token cost — have created a clear product opportunity. No existing solution ties together **(1) mnemonic memory-palace architectures, (2) biologically-grounded forgetting, (3) user-specific cognitive personalisation, and (4) deployment on small-board computers** into a single product. This document maps the research landscape and proposes a differentiated product architecture.

---

## 1. The SOTA Landscape: What the Research Tells Us

### 1.1 The Unified Memory Compaction View (Colaco & Lahjouji, arXiv 2607.08032, Jul 2026)

The most recent survey paper reframes all agent memory problems — KV-cache eviction, prompt pruning, recurrent state bounding, and agent long-term memory — as a single **rate-distortion problem**: what context-derived information to retain vs. discard, at what fidelity, under a resource budget, to preserve downstream utility. Two critical findings:

- **Every layer uses attention-magnitude or recency to decide what to keep**, and every layer fails the same way — discarding information the query will later need.
- **Repeated compaction (what agents actually do in production) is almost never measured.** Benchmarks test single-turn compression.

**Product implication:** Our system must be a *compaction stack* where forgetting is a first-class citizen, not a patch. The rate-distortion lens gives us a unified optimisation objective across all memory layers.

### 1.2 The Forgetting Revolution (Apr–Jun 2026)

Six major papers in Q2 2026 establish forgetting as a fundamental capability:

| Paper | Key Insight | Relevance to Sovereign AI |
|---|---|---|
| **FSFM** (Gu et al., 2604.20300) | Four forgetting mechanisms: passive decay, active deletion, safety-triggered, adaptive reinforcement | Resource-constrained scenarios; 100% security-risk elimination |
| **Oblivion** (Rana et al., 2604.00131) | Decouples read/write paths; forget = decay-driven *inaccessibility* not deletion | Adaptive memory access reduces latency vs flat storage |
| **ScrapMem** (Chang & Ren, 2605.03804) | **Optical Forgetting**: progressive resolution reduction of older multimodal memories | 93% storage reduction on edge; Episodic Memory Graph for causal-temporal structure |
| **Forget to Improve** (Wu et al., 2606.25115) | Net-value-per-byte scoring for KEEP/SHARE/TRUST decisions; **Jetson testbed** | 2.7× memory reduction, 2.4× uplink reduction; injection success 0.75→0 |
| **Learning What to Remember** (Chen & Cheng, 2606.12945) | 7-factor cognitive value model (emotional intensity, goal relevance, value alignment, self/user relevance, task utility, reliability, usage history) | Learns what a *specific user* considers important; interpretable weights; runs on CPU with no API calls |
| **GateMem** (Ren et al., 2606.18829) | Multi-principal shared-memory governance benchmark | User-aware access control and active forgetting on deletion request |

### 1.3 Memory Palace & Mnemonic Architectures for Agents

The most directly relevant paper is **EMoT — Enhanced Mycelium of Thought** (Stummer, arXiv 2603.24065, Mar 2026):

- First reasoning framework to combine **Memory Palace** with **5 mnemonic encoding styles**
- Four-level hierarchical cognition (Micro → Meso → Macro → Meta)
- **Strategic dormancy** (thought nodes go dormant and reactivate) — architecturally essential (quality collapses from 4.2→1.0 when disabled)
- Outperforms Chain-of-Thought on cross-domain synthesis (4.8 vs 4.4)
- Uses visual-spatial mnemonics, narrative encoding, musical-rhythmic encoding, body-kinesthetic encoding, and semantic-logical encoding

**Also relevant: SuperLocalMemory V3.3** (Bhardwaj, 2604.04514, Apr 2026):
- 7-channel cognitive retrieval: semantic, keyword, entity graph, temporal, spreading activation, consolidation, **Hopfield associative** channels
- **Fisher-Rao Quantization-Aware Distance** — 100% precision preferring high-fidelity embeddings over quantized ones (vs 85.6% for cosine)
- **Ebbinghaus Adaptive Forgetting** with lifecycle-aware quantization — first mathematical forgetting curve in local agent memory
- **100% CPU-only, open source, 5000+ monthly downloads**

### 1.4 Hierarchical Memory Architectures

**MARS — Memory-Augmented Agentic Recommender System** (Shen et al., 2605.14401, May 2026):
- Treats memory as a partially-observable problem → **hierarchical belief state**
- Three tiers: Event Memory (raw signals) → Preference Memory (fine-grained chunks with strength/evidence tracking) → Profile Memory (coherent NL narrative)
- **Six-operation lifecycle**: extraction, reinforcement, weakening, consolidation, forgetting, resynthesis
- Agentic LLM-based planner schedules operations (not fixed heuristics)
- +26.4% HR@1, +10.3% NDCG@10 vs baselines

**Human-Inspired Memory Architecture** (Kerestecioglu et al., 2605.08538, May 2026):
- Six mechanisms: sleep-phase consolidation, interference-based forgetting, engram maturation, reconsolidation on retrieval, entity KGs, hybrid multi-cue retrieval
- Dedup-based consolidation: 97.2% retention precision with 58% store reduction

**Engram** (Wang, 2606.09900, Jun 2026):
- Bi-temporal data model, fast write (no LLM), async fact extraction, temporal KG with supersession
- **Lean context beats full history**: 83.6% vs 73.2% at 8× fewer tokens

---

## 2. The Sovereign AI / Edge / Small-Board Opportunity

### 2.1 Why This Matters for Sovereignty

Current agent memory systems assume cloud or server-grade deployment. Sovereign AI — fully offline, user-owned, user-controlled — requires:

| Constraint | Implication |
|---|---|
| Limited RAM (1–8 GB on Raspberry Pi 5 / Jetson Nano / Orange Pi) | Memory store must be aggressively compacted, quantised, or compressed |
| No cloud API calls | All embedding, retrieval, forgetting decisions must run locally |
| Flash storage endurance | Write-amplification from frequent memory updates wears SD cards; minimise writes |
| CPU-only inference (no GPU) | Embedding models must be < 500 MB; retrieval must use efficient approximate search |
| Single-user personalisation | Memory *must* encode what *this specific user* cares about |
| Power footprint | Tiny battery or USB-powered → energy-aware forgetting |

### 2.2 Directly Relevant Edge Research

**Forget to Improve** (Wu et al., 2606.25115) — the only paper with a real Jetson testbed:
- Heterogeneous setup: 2 robot arms + hub
- Net-value-per-byte scoring governs KEEP, SHARE, TRUST
- **2.7× memory reduction, 2.4× uplink reduction**
- Proves forgetting-by-value improves agents

**ScrapMem** (Chang & Ren, 2605.03804):
- Optical forgetting: reduce resolution of older entries → **93% storage reduction**
- EM-Graph: causal-temporal structure for recall

**SuperLocalMemory V3.3** (Bhardwaj, 2604.04514):
- Runs entirely on CPU, zero LLM calls needed
- 7-channel retrieval, quantisation-aware metrics
- npm/PyPI packages with 5000+ monthly downloads

**Internalizing Tool Knowledge via QLoRA** (Shemla et al., 2605.17774):
- 82.6% input length reduction by fine-tuning tool knowledge into weights
- Gemma 4B / Qwen3-4B with 8-bit QLoRA
- 62% less memory, 2.5× faster on small models

---

## 3. Proposed Product Architecture: The Mnemonic Memory Engine (MME)

This is our differentiated product proposal — a sovereign AI agent memory system that uses **cognitively-grounded mnemonic encoding, personalised forgetting curves, and hierarchical memory palaces optimised for small-board computers**.

### 3.1 Architectural Pillars

#### Pillar 1: Memory Palace Construction via User Research

**Insight:** No current system constructs the agent's memory "palace" structure based on individual user research. We propose:

- **Onboarding questionnaire + interaction mining**: extract the user's mental models, domain vocabulary, priority dimensions, and relationship structures
- **Personalised palace topology**: the agent's memory hierarchy (rooms, loci, spatial relationships) mirrors the user's actual cognitive organisation:
  - A doctor → palace organised by patient types, then conditions, then treatments
  - A software engineer → palace organised by projects, then modules, then issues
  - A student → palace organised by subjects, then concepts, then flashcards
- **5 encoding styles (from EMoT)**: assign the user's dominant encoding modality (visual-spatial, narrative, semantic-logical, rhythmic, kinesthetic) to each memory tier

#### Pillar 2: Multi-Factor Value Model (from Chen & Cheng 2026)

- Seven factors with learned user-specific weights:
  1. Goal relevance (to user's current/past goals)
  2. Value alignment (to user's stated values)
  3. Self/user relevance (ego-involvement)
  4. Task utility (for future tasks the user does)
  5. Emotional intensity (user's emotional investment)
  6. Reliability (provenance trust score)
  7. Usage history (retrieval frequency + recency)
- Weights learned per-user via gradient-free optimisation on CPU (proven in paper)
- Single scalar `V(m)` controls: encoding depth, forget risk, retrieval rank

#### Pillar 3: Biologically-Inspired Forgetting Stack

Four-layer forgetting inspired by FSFM + Oblivion + Ebbinghaus:

| Layer | Mechanism | Edge Optimisation |
|---|---|---|
| L1: Passive Decay | Ebbinghaus curve with user-specific decay rate (fast/slow learner detection) | No active computation; timer-based eviction |
| L2: Interference-Based | When new memory uses same "locus" → older must weaken | Locality-aware; minimises conflict checks |
| L3: Budget-Curated | When RAM/eFlash budget exceeded → evict lowest `V(m)/byte` (from Forget to Improve) | Runs at idle/low-power checkpoints |
| L4: Safety-Triggered | User says "forget" / PII detected / malicious input → immediate deletion with provenance removal | Write-ahead log for secure deletion on SD |

#### Pillar 4: Optical Quantisation & Progressive Compression

From ScrapMem + SuperLocalMemory:

- **Embedding quantisation levels**: FP32 → FP16 → INT8 → binary hash → *discard*
- **Optical forgetting schedule**: after T days without retrieval, drop to next quantisation level
- **Fisher-Rao distance** for quantisation-aware retrieval (not cosine similarity)
- **Wear-aware writes**: batch compaction events to flash in large sequential writes to extend SD/emmc life

#### Pillar 5: Hierarchical Tiers (MARS-inspired)

```
Profile Memory (stable, compressed, NL narrative)
    ↑ ↓ consolidates into
Preference Memory (mutable chunks with strength/evidence)
    ↑ ↓ extracted from
Event Memory (raw, lossless, short-lived)
```

- Event memory: circular buffer in RAM (configurable, default 24h of interaction)
- Preference memory: quantised embeddings in SQLite + vector index on SD/eMMC
- Profile memory: natural language summary updated at sleep-phase consolidation
- **Sleep-phase consolidation** (from Human-Inspired Architecture): triggered on device idle / plugged in

### 3.2 Small-Board Computer Specs

Target platforms and expected performance budgets:

| Board | RAM | Storage | Expected Memory Store | Model Size Limit |
|---|---|---|---|---|
| Raspberry Pi 5 | 4-8 GB | 32-128 GB eMMC/SD | 500 MB - 1 GB | < 2B params (4-bit) |
| Jetson Orin Nano | 8 GB | 64 GB | 1-2 GB | < 3B params (4-bit) |
| Orange Pi 5 | 4-16 GB | 32-256 GB | 500 MB - 2 GB | < 2B params (4-bit) |
| Radxa Rock 5 | 4-16 GB | 32-256 GB | 500 MB - 2 GB | < 2B params (4-bit) |

Implementation approach:
- **Embedding model**: BGE-micro or all-MiniLM-L2-v2 (22-80 MB, 384 dims) quantised to INT8
- **Vector store**: SQLite + hnswlib (no server process) or USearch
- **Forgetting scheduler**: `timeloop`-style periodic task at 5-minute granularity
- **Sleep consolidation**: cron-like schedule at 03:00 or when `iostat` shows disk idle

### 3.3 User Research → Palace Construction Pipeline

```
User research data
    ↓
Knowledge mining pipeline (local, no LLM)
    ├── Extracts domain taxonomy (NLP: noun phrase extraction, frequency analysis)
    ├── Identifies user's priority dimensions (pairwise comparison during onboarding)
    └── Maps relationship types (hierarchical, associative, causal)
    ↓
Memory Palace blueprint
    ├── Rooms = high-level domains (max 7 ± 2 — Miller's Law)
    ├── Loci within rooms = sub-domains / key concepts
    └── Spatial relationships = semantic distance (vector similarity)
    ↓
Agent runtime
    └── Places new memories at optimal loci
        └── Forgetting = locus becoming dim / inaccessible
```

---

## 4. Competitive / Landscape Positioning

### What exists today:
| Product | Strengths | Gaps for Sovereign AI | Gaps for Palace/Mnemonic |
|---|---|---|---|
| Mem0 | RAG-based, well-documented | Cloud-first; no forgetting model | Flat memory, no hierarchy |
| Letta (MemGPT) | Virtual context management | Heavy LLM dependency | No personalisation |
| Zep | Good production features | Server-based | No palace concept |
| SuperLocalMemory | CPU-only, open source | General-purpose; no user research pipeline | No sovereign AI specific (yet) |
| Cognee | Graph-based memory | Requires significant RAM | No forgetting physics |

### Our differentiation:
1. **Palace topology from user research** — no competitor does this
2. **Learned per-user multi-factor value model** — replaces global recency/similarity
3. **Forgetting physics optimised for eMMC/SD endurance**
4. **CPU-only, zero-API, sovereign-by-design**
5. **Hierarchical belief state with mnemonic encoding at each tier**

---

## 5. Research Roadmap & Open Challenges

### Phase 1 (0-6 months): Foundation
- [x] Replicate Chen & Cheng (2606.12945) multi-factor value model on CPU
- [x] Implement FSFM taxonomy as software library (4 forgetting mechanisms)
- [x] Port ScrapMem's optical forgetting resolution scaling (link to quantisation)
- [x] Benchmark embedding models on RPi5 / Jetson Orin Nano for latency and memory

### Phase 2 (6-12 months): Palace Construction
- [ ] Develop onboarding questionnaire → palace blueprint pipeline (no LLM)
- [ ] Implement 5 mnemonic encoding styles from EMoT as pluggable serialisers
- [ ] Build episodic memory graph (from ScrapMem) with temporal causal edges
- [ ] Integrate Engram's bi-temporal model for provenance and supersession chains
- [ ] User-specific Ebbinghaus decay rate estimation (interaction pattern analysis)

### Phase 3 (12-18 months): Edge Optimisation
- [ ] Fisher-Rao Quantization-Aware Distance integration (from SLM V3.3)
- [ ] Sleep-phase consolidation scheduler for idle-time memory compaction
- [ ] Wear-aware flash write batching
- [ ] Peer-to-peer memory sharing between devices (from Forget to Improve SHARE)
- [ ] GateMem compliance: multi-user access control in shared household device

### Phase 4 (18-24 months): Autonomous Evolution
- [ ] Agentic scheduler (from MARS): LLM on RPi5 decides when to consolidate/forget/resynthesise
- [ ] Self-evolving palace: user's priorities shift → palace reorganisation triggered
- [ ] Memory = soft prompt internalisation (from SuperLocalMemory parameterization + PEAM)

---

## 6. Key Technical Decisions to Validate

| Decision | Options | Recommendation | Rationale |
|---|---|---|---|
| Embedding model | BGE-small / all-MiniLM / Nomic embed-text | BGE-small-en-v1.5 quantised to INT8 | Best size/quality trade-off; 33 MB, 384-dim |
| Vector index | hnswlib / USearch / FAISS / SQLite-vec | hnswlib for ≤500k vectors; SQLite-vec for <50k | hnswlib fastest on CPU for medium scale |
| Forgetting trigger | Timer / budget / event | Hybrid: budget (primary) + timer (background) | Budget catches OOM; timer catches stale |
| Palace serialisation | JSON / SQLite / custom binary | SQLite + memory-mapped binary vectors | SQLite: proven reliability on SBCs; indexed queries |
| User research format | Guided interview / free text / Q&A | Structured pairwise comparison (5 min) + free-text mining | Low user burden; rich signal |

---

## 7. Summary of Key Papers to Digest

*Prioritised reading list for engineering team:*

1. **Chen & Cheng (2606.12945)** — *Learning What to Remember* ← VALUE MODEL
2. **Colaco & Lahjouji (2607.08032)** — *What to Keep, What to Forget* ← UNIFYING THEORY
3. **Gu et al. (2604.20300)** — *FSFM* ← FORGETTING TAXONOMY
4. **Chang & Ren (2605.03804)** — *ScrapMem* ← EDGE FORGETTING
5. **Wu et al. (2606.25115)** — *Forget to Improve* ← JETSON DEPLOYMENT
6. **Stummer (2603.24065)** — *EMoT* ← MNEMONIC ENCODING
7. **Bhardwaj (2604.04514)** — *SuperLocalMemory V3.3* ← CPU-ONLY RETRIEVAL
8. **Shen et al. (2605.14401)** — *MARS* ← HIERARCHICAL MEMORY
9. **Kerestecioglu et al. (2605.08538)** — *Human-Inspired Memory* ← SLEEP CONSOLIDATION
10. **Wang (2606.09900)** — *Engram* ← BI-TEMPORAL RETRIEVAL

---

## 8. Context Engineering Framework & The Twin Unified Force

### 8.1 The Duality: Two Forces, One Cognition

Every living memory system is governed by a twin unified force — two opposing but complementary drives that together produce coherent cognition:

```
FORCE A: MNEMONIC (Conservation / Depth / Structure)
  ─ preserves, organises, spatialises, consolidates
  ─ tends toward durability, hierarchy, compression
  ─ the Memory Palace — rooms, loci, relationships, atemporal maps

FORCE B: CONTEXTUAL (Flow / Presence / Navigation)
  ─ surfaces, assembles, temporalises, attends
  ─ tends toward relevance, recency, linear unfolding
  ─ the Context Window — tokens, position, attention, live narrative
```

| Dimension | Mnemonic Force | Contextual Force |
|---|---|---|
| Time | Past→Present (retrieval) | Present→Future (anticipation) |
| Space | Rooms & loci (stable topology) | Path & gaze (movement through) |
| Fidelity | Compressed, quantised, lossy | Ephemeral, high-res, lossless |
| Operation | Consolidation (write) | Attention (read) |
| Failure mode | Fossilisation (cannot update) | Amnesia (cannot retain) |
| Sovereign optimisation | Store minimisation | Context budget allocation |

**The unification:** An agent's behaviour at any moment is the product of these two forces in dynamic equilibrium. The mnemonic force pulls experience into durable structure; the contextual force pulls structure into live attention. Neither is useful alone.

### 8.2 Context Engineering as a Practice

Context engineering is the discipline of designing the interface between these two forces — how the Contextual Force queries the Mnemonic Force, and how the Mnemonic Force shapes the Contextual Force.

#### 8.2.1 The Context Assembly Pipeline

```
Current State (sensors, timestep, user input)
    │
    ▼
Force B: Contextual query
    ├── Attentional filter: what is salient NOW?
    ├── Temporal scope: what time horizon matters?
    └── Goal binding: what task frame is active?
    │
    ▼
Force A: Mnemonic response
    ├── Spatial retrieval: which palace rooms are relevant?
    ├── Factual retrieval: which knowledge chunks?
    ├── Episodic retrieval: which past experiences?
    └── Value-weighted ranking: V(m) scores each candidate
    │
    ▼
Assembled Context (the working window)
    ├── System prompt (stable identity instructions)
    ├── Retrieved memories (from Force A)
    ├── Current interaction state (from sensors/user)
    ├── Tool/action affordances
    └── Attention budget (how many tokens remain)
    │
    ▼
Agent reasoning step → action → observation
    │
    ▼
Force A: Consolidation
    └── Encode new experience into palace
        ├── Assign to optimal locus (spatial indexing)
        ├── Compute V(m) value (7-factor model)
        ├── Trigger interference checks (same-locus entries)
        └── Schedule optical quantisation (future decay)
```

#### 8.2.2 Context Engineering Principles

**Principle 1: Context is a budget, not a cache.**
Treat context window as a scarce resource to be *allocated* deliberately, not filled reactively. Every token should justify its presence via expected value per byte (from Forget to Improve).

**Principle 2: Mnemonic retrieval should anticipate, not react.**
The palace should pre-emptively surface candidates before the agent asks. Spreading activation (from SuperLocalMemory) warms up related rooms when a topic is entered.

**Principle 3: Forgetting is a context-engineering decision.**
What leaves the context window is not lost — it returns to the palace at reduced fidelity. The forgetting stack (L1-L4) is the feedback loop that trains the context window to be more selective.

**Principle 4: The context window has a spatial analogue.**
In a memory palace, you can only see one room at a time, but you know the layout of the whole building. Similarly, the context window should contain both the current "room" (immediate details) and a minimap (high-level palace topology).

### 8.3 The Twin-Force Controller (TFC)

A proposed architectural component that manages the dynamic equilibrium:

```
TFC state variables:
  e = mnemonic/conservation bias       (0 = pure flow, 1 = pure structure)
  a = attentional temperature           (0 = narrow focus, 1 = broad scan)
  τ = temporal horizon                  (how far back to retrieve)
  r = resolution level                  (current quantisation tier)

TFC update rules:
  After each interaction:
    if novelty > threshold → decrease e (favour encoding new rooms)
    if repetition > threshold → increase e (favour consolidation)
    if context pressure > budget → trigger optical forgetting
    if goal shift detected → reset τ, flush event memory
    if user satisfaction ↓ → increase a (wider retrieval net)
    if user satisfaction ↑ → decrease a (narrow, efficient retrieval)

TFC as a learned policy:
  The user-specific weights from Pillar 2 (7-factor value model)
  naturally produce a personalised TFC — some users are "builders"
  (high e, structured palace), others are "explorers"
  (low e, fluid context, rapid forgetting).
```

### 8.4 Engineering Context for Small-Board Constraints

On a Raspberry Pi 5 or Jetson, the twin forces must be managed with extreme discipline:

| Resource | Mnemonic Force Strategy | Contextual Force Strategy |
|---|---|---|
| RAM (≤ 8 GB) | Keep only active palace rooms in memory; swap cold rooms to compressed SQLite | Context window < 4K tokens; everything else must be retrieved, not resident |
| CPU (≤ 4 cores) | Async consolidation at <= 5% CPU; batch writes at idle | Retrieval must use quantised distances (Fisher-Rao) — cosine similarity on FP32 is too expensive |
| Flash writes | Wear-aware batching; write-amplification < 1.1x | No writes on the contextual path — write path is fully asynchronous |
| Power (≤ 15W) | Consolidation only on external power or battery > 50% | Context assembly must not exceed 1W per inference |

### 8.5 The Memory-Context Cycle as a Unified Field

```
        FORCE A (Mnemonic / Palace)
        │ stores, structures, forgets
        │
        ▼
  ┌───────────┐         ┌──────────────┐
  │  LONG-TERM │ ──────► │   CONTEXT    │
  │  STORAGE   │ retrieve │   ENGINE     │
  │  (rooms,   │ ◄────── │  (assembler, │
  │   loci,    │ update  │   scheduler, │
  │   facts)   │         │   attention) │
  └───────────┘         └──────────────┘
        ▲                       │
        │ consolidate           │ produce
        │                       ▼
        │               ┌──────────────┐
        └───────────────│  AGENT       │
         encode         │  REASONING   │
                        │  (action,    │
                        │  thought)    │
                        └──────────────┘
                               │
                               │ generate
                               ▼
                        ┌──────────────┐
                        │  USER        │
                        │  INTERACTION │
                        └──────────────┘
                               │
                               │ feedback
                               ▼
                        FORCE B (Contextual / Flow)
                        │ attends, decides, forgets-in-context
```

**The twin unified force thesis:** Memory and context are not separate systems. They are the same system viewed at different timescales. Memory is context that has been consolidated. Context is memory that has been retrieved. The palace is the slow-time architecture of the agent's mind; the context window is its fast-time surface. Sovereign AI demands that both be optimised together, on the same small board, for the same user.

### 8.6 Research Frontier: Self-Evolving Twin Dynamics

Open questions for Phase 4+:
- Can the twin-force controller (e, a, τ, r) be learned end-to-end from user interaction signals?
- Does the user's dominant mnemonic encoding style (from EMoT's 5 styles) predict their optimal TFC parameters?
- Can the palace topology self-reorganise as the user's priorities drift, without requiring rebuild?
- What is the minimal context budget for useful sovereign agent behaviour? 1K tokens? 512?
- At what point does the forgetting overhead exceed the retrieval savings on an SBC?

---

## 9. Search Algorithms: Retrieving Context from Agentic Understanding

The retrieval engine is the critical bridge between Force A (stored palace) and Force B (live context). On an SBC with tight budgets, the search algorithm *is* the architecture — everything else is storage and scheduling. Below is a taxonomy of every relevant algorithm, how they compose, and how they map to agentic understanding.

### 9.1 The Search in the Palace Metaphor

```
SEARCH TYPE          PALACE ANALOGUE             AGENTIC USE CASE
──────────────       ──────────────────          ─────────────────────
Exact match          Opening a labelled door     User: "remember my API key"
Lexical search       Calling into a room         User: "what did I say about X?"
Semantic search      Walking to a similar room   User: "find things like this idea"
Graph traversal      Moving through corridors    "What entities are connected to this?"
Spreading activation Echo-location               "What's related to what I'm thinking about?"
Hierarchical zoom    Moving floor ↔ room ↔ desk  "Drill into this project's sub-tasks"
Temporal search      Rewinding a recording       "What happened last Tuesday?"
Associative recall   Smell triggering a memory   "This reminds me of..."
Anticipatory fetch   Pre-opening likely doors    Agent predicts what user will need next
Multi-hop search     Exploring connected rooms   "I need to find A, and A connects to B"
```

### 9.2 Algorithm Taxonomy

#### 9.2.1 Dense Vector Search

| Algorithm | Metric | Storage Per Vector | Recall@10 | SBC Viable? |
|---|---|---|---|---|
| Cosine similarity | `cos(a,b) = a·b / (||a||·||b||)` | 4 bytes × dim (FP32) | High | Yes, with INT8 quantisation |
| Dot product | `a·b` | 4 bytes × dim (FP32) | High | Yes, with INT8 |
| Euclidean (L2) | `sqrt(Σ(a-b)²)` | 4 bytes × dim (FP32) | High | Avoid — no sqrt on edge |
| Fisher-Rao FRQAD | geodesic on statistical manifold | Same as source | Highest | **Recommended** — 100% precision preferring high-fidelity, zero prior art in production, works on CPU |

**FRQAD (from SLM V3.3)**: Replaces cosine similarity entirely. Operates on the Gaussian statistical manifold — treats each embedding as a distribution, measures geodesic distance. Achieves 100% precision at preferring high-fidelity over quantised embeddings (cosine: 85.6%). **This should be the default distance metric in our system.**

#### 9.2.2 Approximate Nearest Neighbour (ANN) Indexes

| Index | Build Time | Query Time | Memory | SBC Notes |
|---|---|---|---|---|
| **HNSW** (hierarchical navigable small world) | O(n log n) | O(log n) | High (1.1–2× data) | Best for ≤500k vectors; multi-layer graph fits in RAM |
| **IVF** (inverted file index) | O(n) | O(sqrt(n)) | Low-Medium | Coarse quantiser requires training; good for >1M |
| **SQLite-vec** | O(n) | O(n) (brute force) | Low (in-DB) | Best for ≤50k vectors; zero extra processes |
| **USearch** | O(n log n) | O(log n) | Medium | Smaller memory than HNSW; pure C, SIMD; good RPi5 candidate |
| **DiskANN-style** | O(n log n) | O(log n) | Low (on disk) | Too much I/O for SD cards; not recommended |

**Recommendation for SBCs:**
- < 10k memories → SQLite-vec (no dependencies, no server, crash-safe)
- 10k–500k memories → USearch or HNSW in memory-mapped mode
- Palace rooms as HNSW graphs, loci as flat arrays within rooms

#### 9.2.3 Sparse / Lexical Search

| Algorithm | Strengths | SBC Viability |
|---|---|---|
| BM25 (Okapi) | Exact keyword match; OOV robustness | **Yes** — lightweight, in SQLite FTS5 |
| SPLADE | Learned sparse vectors; end-to-end | Yes — distill to < 50 MB |
| Elasticsearch-style inverted index | Production-grade | Overkill for SBC — SQLite FTS5 is enough |

**BM25 + SQLite FTS5 should ship as the baseline lexical channel** — zero ML dependency, instant on any SBC, handles named entities and code identifiers that embeddings smear.

#### 9.2.4 Hybrid Fusion (Dense + Sparse)

```
Retrieved from dense channel:   [d1, d2, d3, ...]  scores: s_dense
Retrieved from sparse channel:  [s1, s2, s3, ...]  scores: s_sparse

Fusion strategies (all viable on CPU):
  Reciprocal Rank Fusion (RRF):    score = Σ 1/(k + rank)    k=60 standard
  Linear interpolation:            score = α·s_dense + (1-α)·s_sparse   α learned per user
  Learning-to-rank (LTR):          score = w·[features]       features = {dense, sparse, V(m), recency, ...}
```

**RRF is the default fusion strategy for Phase 1-2** — no training, robust, single parameter. Phase 3+ can learn user-specific α via the same gradient-free optimiser from the 7-factor value model.

#### 9.2.5 Graph-Based Search

Three graph modalities, each requiring a different traversal algorithm:

| Graph Type | Search Algorithm | Use Case | SBC Feasibility |
|---|---|---|---|
| Entity KG (from Human-Inspired) | BFS/DFS from seed entities | "Find everything related to Project X" | Yes — adjacency lists in SQLite |
| Temporal KG (from Engram) | Point-in-time walk + supersession chain | "What did I believe about Y last week?" | Yes — but merge-on-read to avoid materialising full history |
| Palace topology (rooms → loci → chunks) | Hierarchical drill-down + lateral roam | "Go to the medicine room, find the allergy locus" | Yes — tree structure, O(log n) |

**Key insight for SBCs:** Run graph traversals only when dense/lexical retrieval has already established a seed set. Never run unseeded graph search — it's too expensive.

#### 9.2.6 Spreading Activation (from SLM V3.3)

```
Algorithm:
  1. Seed nodes = top-k from dense/lexical/hybrid retrieval
  2. Activate neighbours in entity KG with decay factor γ (0.3–0.5)
  3. Activate 2-hop neighbours with γ²
  4. Collect all nodes with activation > threshold T
  5. Score = activation × V(m)

Computational cost: O(seed × avg_degree × hops)
SBC viability: Yes, for γ ≥ 0.3 (fast decay limits spread to ~2 hops)
```

**Use case:** When the user changes topics or the agent is mid-reasoning and needs to "warm up" related knowledge before the next query. The TFC's attentional temperature `a` directly controls the spreading activation threshold — higher `a` = wider spread.

#### 9.2.7 Associative / Hopfield Retrieval (from SLM V3.3)

Modern Hopfield networks (Ramsauer et al., 2021) can store and retrieve patterns with exponential storage capacity:

```
Pattern storage:    W = Σ x_i · x_iᵀ    (outer product of memory vectors)
Pattern retrieval:  x_new = W · x_query   (one-step associative recall)
```

**SBC viability:** Limited. Hopfield retrieval is O(dim²) per query. Only practical for very small associative stores (< 100 vectors) or when using chunked/quantised variants. We should flag this as Phase 4 research.

#### 9.2.8 Speculative Retrieval / Anticipatory Prefetching

```
Trigger conditions (evaluated at every step):
  1. User is mid-sentence → prefetch top-3 from current topic room
  2. Agent just took an action → prefetch likely next action's context
  3. TFC's "a" (attentional temperature) is high → prefetch adjacent rooms
  4. Time since last retrieval > threshold → prefetch recent event summaries

Cost: prefetch adds 1 query to the background queue per trigger
Benefit: reduces perceived latency by 40-60% (cache hit rate in shadow mode)
SBC: all embedding queries are async; prefetched results sit in a small LRU cache (256 KB)
```

### 9.3 The Composed Search Pipeline

The actual retrieval path at runtime fuses multiple algorithms into a single ranked list:

```
User input / Agent internal query
    │
    ▼
┌───────────── Stage 1: Intent Classification ─────────────┐
│  Is this query:                                           │
│  • Factual lookup?           → route to exact/lexical     │
│  • Exploratory/open-ended?   → route to dense semantic    │
│  • Relational/connective?    → route to graph traversal   │
│  • Temporal ("last time")?   → route to temporal search   │
│  • Meta-cognitive ("why")?   → route to associative       │
│  (Classifier: < 5 MB distilled BERT, runs in < 10ms)     │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────── Stage 2: Parallel Retrieval ─────────────────┐
│  Channel 1: BM25 (SQLite FTS5)             ← lexical    │
│  Channel 2: FRQAD cosine (USearch/HNSW)     ← dense      │
│  Channel 3: Entity graph traversal          ← relational │
│  Channel 4: Temporal filter (τ horizon)     ← temporal   │
│  Channel 5: Prefetched cache (if hit)       ← anticipatory│
│  (All channels run in parallel via thread pool,           │
│   max 4 threads, each < 20ms)                            │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────── Stage 3: Fusion & Reranking ─────────────────┐
│  Step 3a: Reciprocal Rank Fusion (RRF) on all candidates │
│  Step 3b: Score = RRF_score × V(m) × recency_boost       │
│  Step 3c: Apply context budget (eliminate lowest V/B)    │
│  Step 3d: Deduplicate superseded facts (Engram bi-temporal)│
│  (All steps O(n) on candidates, n < 200 typical)         │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────── Stage 4: Context Assembly ───────────────────┐
│  • Take top-k candidates (k = budget / avg_token_cost)    │
│  • Format as provenance-tagged context blocks              │
│  • Insert into prompt with room labels (palace minimap)   │
│  • If insufficient budget → trigger TFC: lower τ or a     │
└──────────────────────────────────────────────────────────┘
    │
    ▼
Agent reasoning step
    │
    ▼
┌───────────── Stage 5: Feedback & Repair ──────────────────┐
│  If agent could not answer from retrieved context:        │
│    1. Lower threshold, re-run Stage 2 (wider net)         │
│    2. If still failing → mark retrieval as failed         │
│       → TFC increases attentional temperature a           │
│    3. Log query embedding + failure → training data       │
│       for future α learning                               │
└──────────────────────────────────────────────────────────┘
```

### 9.4 Agentic Understanding: Beyond Keyword & Similarity

Search in our system must be guided by **what the agent understands about the user's state**, not just surface form:

#### 9.4.1 Theory-of-Mind Guided Retrieval

```
Cues that shape the search intent:
  • User's emotional state (from sentiment analysis on input)
    → boost emotional-intensity factor in V(m)
  • User's current goal (from goal-tracking stack)
    → boost goal-relevance factor in V(m)
  • User's recent satisfaction (from previous turn sentiment)
    → lower attentional temperature if frustrated (narrower, more precise search)
  • User's cognitive load (from input brevity, hesitation markers)
    → prefer concise, previously-successful responses
```

#### 9.4.2 Epistemic State Search

The agent tracks **what it already knows vs what it needs to know**:

```
Epistemic vector:
  Known facts:       {f₁, f₂, ...}  (from retrieved context this session)
  Assumed gaps:      {g₁, g₂, ...}  (from plan steps with missing context)
  Established truths: {t₁, t₂, ...}  (from confirmed facts with reliability > 0.9)

Search bias:
  If query ∈ g₁...gₙ → prioritise retrieval channels that have low V(m) uncertainty
  If query ⊆ known  → skip retrieval (save budget)
  If query contradicts t₁ → boost reliability factor, flag contradiction
```

#### 9.4.3 Curiosity-Driven Exploration

When the agent has spare context budget and no pressing user need:

```
Curiosity signal:
  • Rooms with oldest last-access timestamps
  • Loci with highest variance in V(m) (unresolved or conflicting memories)
  • Entity graph nodes with fewest connections (potential missing links)

Exploration triggers:
  • User is idle > 30s → schedule 1 exploratory query
  • Context window has padding > 50% → fill with 1 curiosity candidate
  • Sleep-phase consolidation can run exploratory traversals at zero latency cost

Exploration is a "free" memory maintenance operation — it surfaces forgotten
items for potential reconsolidation or safe deletion.
```

#### 9.4.4 Goal-Tree Guided Retrieval

```
User interaction → inferred goal tree:
   Level 0: "Plan a trip to Japan"
      Level 1: "Book flights"          ← active
      Level 1: "Find accommodation"    ← pending
      Level 1: "Plan itinerary"        ← pending
         Level 2: "Visit Kyoto"
         Level 2: "Visit Tokyo"

At each turn, the search engine:
  1. Retrieves context for the active goal (Level 1+)
  2. Prefetches context for adjacent goals (same Level 1 parent)
  3. For the goal "Visit Kyoto" → enters the "Kyoto" locus in the "Travel: Japan" room
```

### 9.5 Search Algorithm Selection by Query Type

| Query Type | Primary Algorithm | Secondary | Edge Cost | Latency Target |
|---|---|---|---|---|
| "What is my API key?" | Exact string match (SQLite) | BM25 | ~0 | < 5ms |
| "What did I say about X?" | BM25 + FRQAD hybrid | Entity graph | Low | < 30ms |
| "Find things like this" | FRQAD cosine (dense) | Spreading activation | Medium | < 50ms |
| "What's connected to Y?" | Entity KG BFS (2-hop) | FRQAD from seed | Medium | < 80ms |
| "What happened last week?" | Temporal filter + BM25 | FRQAD on events | Low | < 40ms |
| "Why did we decide Z?" | Temporal KG (supersession chain) | Episodic event log | Medium | < 100ms |
| "Summarise my Project X" | Hierarchical zoom: room→loci→chunks | Entity KG aggregation | High | < 200ms |
| "This reminds me of..." | Hopfield associative (if active) | FRQAD of last 3 turns | Low | < 30ms |
| "What should I do next?" | Goal-tree guided + curiosity | Spreading activation from active goals | Medium | < 80ms |
| Open-ended exploration | Curiosity (lowest V(m), oldest access) | Graph traversal | Low (sleep phase) | N/A (async) |

### 9.6 SBC-Optimised Search Implementation

```
Search engine resource budget (RPi 5, 4 GB):
  ┌──────────────────────────────────────────────┐
  │  Component               RAM     CPU         │
  ├──────────────────────────────────────────────┤
  │  Embedding model (INT8)  22 MB   10ms/query   │
  │  HNSW index              50 MB   <5ms/query   │
  │  SQLite FTS5              8 MB   <2ms/query   │
  │  Entity KG (adjacency)   12 MB   <1ms/traversal│
  │  V(m) weights store      64 KB   <1ms          │
  │  Prefetch cache         256 KB   N/A           │
  │  Thread pool (4)         512 KB   N/A           │
  │  Total footprint        <100 MB                │
  └──────────────────────────────────────────────┘

Time budget per query:
  Stage 1 (intent):     < 10ms
  Stage 2 (parallel):   < 50ms  (worst: BM25 + FRQAD + KG simultaneously)
  Stage 3 (fusion):     < 5ms
  Stage 4 (assemble):   < 5ms
  Stage 5 (feedback):   < 2ms
  ─────────────────────────────────
  Total (p95):          < 75ms  — fast enough for real-time interaction

Power cost per query: ~0.5J (0.015 Wh) — 66,000 queries per kWh
```

### 9.7 Open Research Questions for Search

1. **Can FRQAD be implemented as a SIMD kernel on ARM NEON?** Would give 4–8× speedup on RPi5 vs naive scalar.
2. **Does spreading activation from the palace topology outperform entity-KG-based activation for the same compute budget?** If yes, we can drop the entity KG entirely.
3. **Can the intent classifier (Stage 1) be replaced by the TFC's own state variables?** If the TFC knows e, a, τ, r, maybe the query routing can be deterministic.
4. **At what memory scale does HNSW rebuild cost outweigh retrieval benefit on eMMC?** Initial estimate: ~500k vectors. Below that, brute-force with FRQAD may be faster due to zero index overhead.
5. **Can curiosity-driven exploration be gated by battery level?** "If battery < 20%, suppress all non-essential retrieval."

---

## 10. Immediate Next Steps

1. Fork/replicate **Learning What to Remember** and calibrate 7 factors for edge-latency (< 50ms inference on RPi5)
2. Build minimal memory palace prototype: 3-tier MARS-style memory with EMoT's visual-spatial mnemonic as the primary encoding style
3. Implement the Twin-Force Controller as a configurable governor (e, a, τ, r state machine)
4. **Implement FRQAD (Fisher-Rao distance) as a drop-in replacement for cosine similarity** — this is the single highest-impact search optimisation
5. **Build the composed search pipeline** (Stages 1-5) with intent routing and parallel multi-channel retrieval
6. **Port BM25 + SQLite FTS5 as the lexical fallback channel** — zero ML dependency, handles names/code
7. **Benchmark HNSW vs USearch vs brute-force FRQAD** on RPi5 across 10k/50k/200k memory sizes
8. Benchmark optical forgetting resolution schedule on eMMC wear (write amplification factor under different schedules)
9. Design 5-minute structured user-research interaction to extract domain taxonomy and priority dimensions
10. Prototype context assembly pipeline with budget-aware retrieval (Forget to Improve net-value-per-byte scoring)
11. Implement epistemic state tracker for retrieval-side budget pruning
12. File provisional patent on "Twin-Force Memory-Context Architecture for Sovereign AI Agents"
