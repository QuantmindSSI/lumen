# Lumen Brand Bible

## Sovereign Memory for Local AI Agents

**Version:** 1.0.0  
**Status:** Public Release  
**Classification:** Open Brand Guidelines

---

## 0. The Core Truth

> **Memory without context is a closed palace.**  
> **Context without memory is an empty room.**  
> **Lumen is the light that connects them.**

Lumen is the memory system built specifically for **local AI** — the agents that run on your hardware, process your data, and answer to no cloud.

This brand bible is our constitution. It governs how we speak, how we look, how we build, and how we treat the humans who trust us with their agent's mind.

---

## 1. Brand Name & Etymology

### **Lumen**

- **Latin** *lūmen* = light, clarity, the opening through which light passes
- **Biology** = the inner space of a vessel where flow happens
- **Optics** = the unit of luminous flux
- **Culture** = root of *illuminate*, *luminary*, *luminous*

**Why it matters for local AI:**

Local AI agents live in darkness — disconnected from the cloud's vast training data, isolated on edge devices with limited memory and context. Lumen is the light they generate for themselves. Not borrowed from a data center. Not rented from an API. **Their own light.**

| Criterion | Score | Rationale |
|---|---|---|
| Memorability | 5/5 | Two syllables, exists in every major language, easy to spell |
| Meaning-fit | 5/5 | Light generated locally; cavity/flow channel = memory and context |
| Sovereign identity | 5/5 | Carries its own light — needs no external fire |
| Technical identity | 4/5 | `lumen`, `lumenctl`, `liblumen` — all natural |
| International | 5/5 | Same spelling in French, Spanish, Italian, Portuguese, Romanian |
| Extension | 5/5 | Luminary (contributor), Illuminate (onboarding), Prism (debug), Beam (sync) |

---

## 2. Brand Strategy: The Sovereign Position

### 2.1 The Problem We Solve

Modern AI memory systems are built for the cloud:
- They assume gigabytes of RAM
- They require always-on internet
- They rent your data back to you
- They forget nothing — or forget everything at once

**Local AI can't use these.** A Raspberry Pi 5 has 4 GB RAM. A Jetson Orin Nano has 8 GB. Your laptop might have 16 GB. These devices need memory that is:
- **Bounded** — knows when to forget
- **Personal** — learns what's important to *you*
- **Sovereign** — never leaves your hardware
- **Biologically grounded** — forgets like a mind, not like a database

### 2.2 Brand Essence

**What Lumen is:**
A sovereign-first, edge-native memory and context framework that gives local AI agents a durable, personal, bounded mind — using mnemonic palace architecture, biologically-grounded forgetting, and a twin-force controller that balances memory depth against context breadth.

**What Lumen is not:**
- Not a vector database (we include one, but it's not the product)
- Not a cloud service (zero API calls, zero network dependency)
- Not a prompt framework (we supply context, not templates)
- Not "AI infrastructure" (we are the agent's mind, not its plumbing)

### 2.3 Brand Archetype: The Sage-Architect

Lumen sits at the intersection of two archetypes:

| Archetype | Expression |
|---|---|
| **Sage** | Wisdom, memory, truth-seeking, depth, learning from the past |
| **Architect** | Structure, palace-building, systems thinking, intentional design |

**Voice:**
- Precise but not academic
- Calm but not cold
- Deep but not obscure
- Technical but welcoming
- Confident but never arrogant

**Brand adjectives:** *luminous, precise, enduring, personal, sovereign, twin-natured, forgetful-by-design*

### 2.4 Competitive Positioning

```
                         CLOUD-HEAVY
                              │
                Mem0, Zep, Letta, LangMem
                              │
                         LOW  │  HIGH
                     STRUCTURE │ STRUCTURE
                                │
                    (flat RAG)  │  (hierarchy)
                                │
                              │ LUMEN
                              │ Cognee
                              │
                         SOVEREIGN / EDGE
```

Lumen occupies the **high-structure, sovereign/edge** quadrant. We are the only entrant building a *palace* for agents that live on your desk, not in a data center.

### 2.5 Target Personas

| Persona | Need | Lumen Entry Point |
|---|---|---|
| **Sovereign AI builder** | Offline agent with personal memory on home server | `pip install lumen` on RPi5 |
| **Privacy-first developer** | Memory with first-class forgetting, audit, and deletion | Compliance module + safety triggers |
| **Edge robotics engineer** | Agent remembers task context across sessions on Jetson | `lumen init --device jetson` |
| **Personal AI researcher** | Platform to study personalised memory palaces | Python SDK + user-research pipeline |
| **Enterprise compliance officer** | GDPR-ready memory with provenance chains and audit logs | Structured compliance exports |
| **Open-source contributor** | Meaningful project with real physics and no corporate gatekeeper | GitHub issues + Luminary program |

---

## 3. Visual Identity

### 3.1 The Vesica Mark (Primary)

Two overlapping circles of equal radius, offset so each centre lies on the other's circumference. The intersection forms the *vesica piscis* — the ancient symbol of unity between two forces.

```
        ┌───┐
     ┌──┤   ├──┐
     │  │ ✦ │  │     ← the eye of light at the intersection
     └──┤   ├──┘
        └───┘
    Memory  Context
    (cold)  (warm)
```

**Meaning:**
- Left circle = **Memory** (Mnemonic Force) — deeper, cooler blue
- Right circle = **Context** (Contextual Force) — warmer, active amber
- Intersection = **Lumen** — the light at the union, the agent's cognition
- The eye is also an aperture — light entering, context being assembled from memory
- Three elements mirror the three-tier memory hierarchy: event → preference → profile

**Why the vesica for local AI:**
The vesica is the oldest symbol of self-contained unity. It needs no external frame. It generates its own meaning from the intersection. Just like a local AI agent generates its own intelligence from the intersection of what it remembers and what it attends to.

### 3.2 Color Palette

```
PRIMARY PALETTE

Memory Blue        #1B2A4A   Deep navy — stable, archival, deep
  └── Light        #3D5A80   Pull quotes, secondary backgrounds
  └── Dark         #0F1A2E   Backgrounds, code blocks, terminal

Context Amber      #E8A838   Warm, active, present
  └── Light        #F4C56E   Highlights, hover states
  └── Dark         #B87D1E   Active buttons, links

Lumen White        #F5F0E8   Warm off-white — paper, light
  └── Pure         #FFFFFF   Cards, code blocks
  └── Dim          #E8E0D4   Borders, dividers

ACCENT PALETTE

Twin Pulse         #7C5CBF   The intersection colour — purple/violet
  └── used for: gradients between Memory & Context
  └── used for: the intersection eye in the logo

Signal Green       #2D8A5E   Success, consolidated, retrieved
Compliance Red     #B83A3A   Deletion, safety-trigger, critical
Exploration Gold   #D4A840   Curiosity, prefetch, background

SEMANTIC USAGE

  Memory storage UI:      Memory Blue backgrounds
  Context window UI:      Context Amber accents
  Retrieval/query:        Lumen White with Twin Pulse highlights
  Forgetting:             Compliance Red (deletion) or muted grey (decay)
  Consolidation:          Signal Green pulsing
  Exploration:            Exploration Gold glow
```

### 3.3 Typography

| Use | Font | Fallback | Rationale |
|---|---|---|---|
| Headings (display) | **EB Garamond** | Georgia, serif | 400 years of history; luminous, architectural, timeless. The serif evokes the palace — carved, permanent |
| Headings (UI) | **Inter** | system sans-serif | Clean, highly legible at small sizes, excellent spacing |
| Body text | **Inter** | system sans-serif | Works across documentation, UI, marketing |
| Code / technical | **JetBrains Mono** | monospace | Ligatures for `->`, `=>`, `!=` — developer-friendly |
| Brand wordmark | **EB Garamond** italic | Georgia italic | Classical weight for "lumen" wordmark |

### 3.4 Visual Design Principles

1. **Light as medium**: Backgrounds feel like lit paper, not dark glass. Lumen White (#F5F0E8) as default bg.
2. **Duality as order**: Every UI acknowledges the twin-force — two columns, paired cards, split timelines. Structural, not decorative.
3. **Depth through shadow**: Shallow shadows (palace rooms receding). Memory is layered.
4. **Motion as retrieval**: Loading states animate outward from a point — retrieving from depth, not spinning.
5. **Typography hierarchy**: Garamond (timeless memory) + Inter (functional context). The twin forces in type.

### 3.5 Iconography

- **Stroke-based** (2px), not filled
- **Rounded square base** for memory icons (rooms, loci, palace)
- **Circular base** for context icons (window, attention, query)
- The combination creates a distinctive visual language

---

## 4. Technical Identity

### 4.1 Architecture Naming

The twin-force concept is encoded *in the architecture itself*.

```
lumen/
  ├── force/
  │   ├── mnemonic/           # Force A: Memory Palace
  │   │   ├── palace/         #   Topology (rooms, loci, corridors)
  │   │   ├── consolidation/  #   Write path + scheduling
  │   │   └── forgetting/     #   L1-L4 forgetting stack
  │   └── contextual/         # Force B: Context Engine
  │       ├── assembly/       #   Context window builder
  │       ├── retrieval/      #   Search algorithms
  │       └── attention/      #   Budget allocation
  ├── lumen/                  # The Unification
  │   ├── controller/         #   TFC: e, a, τ, r state machine
  │   ├── lifecycle/          #   6-operation lifecycle
  │   └── user/               #   Per-user weights, 7-factor V(m)
  └── sovereign/              # Edge-specific optimisations
      ├── frqad.py            # Fisher-Rao distance (SIMD NEON)
      ├── optical.py          # Progressive quantisation schedule
      └── wear.py             # Flash-aware write batching
```

**Naming conventions:**
- Memory is *substantive, concrete* (room, locus, chunk, fact, palace)
- Context is *active, verbal* (assembly, attention, retrieval, intent, query)
- The union is *light* (lumen, illumination, beam, prism)
- Forgetting is *natural* (decay, fade, dim, release — never "delete")

### 4.2 API Design Principles

```python
import lumen

# Force A: Memory operations are nouns (places in a palace)
lumen.memory.store(fact="user prefers dark mode", room="preferences")
lumen.memory.locus("preferences.dark_mode")
lumen.memory.consolidate()
lumen.memory.forget("preferences.dark_mode")  # graceful

# Force B: Context operations are verbs (actions in the window)
lumen.context.assemble(query="what theme does the user want?")
lumen.context.budget()
lumen.context.retrieve(similar_to=..., n=5)

# Unification: Lumen operations (the light)
lumen.status()  # TFC state: e, a, τ, r
lumen.controller.adjust(e=0.7, a=0.3)
```

### 4.3 Error Codes

```
LME-1001  RoomNotFound             (mnemonic error)
LME-1002  LocusConflict            (interference)
LME-1003  ConsolidationFailed      (write path)

LCX-2001  BudgetExceeded           (context too large)
LCX-2002  RetrievalEmpty           (no matching context)
LCX-2003  AssemblyTimeout

LLM-3001  TFCStuck                 (equilibrium unattainable)
LLM-3002  ValueModelUncalibrated   (V(m) not yet learned)
LLM-3003  SovereignViolation       (blocked external API call)
```

---

## 5. Brand Voice & Language

### 5.1 The Lumen Lexicon

| Say this | Not this | Why |
|---|---|---|
| Lumen (noun, proper) | "the platform", "the system" | Lumen is a mind, not infrastructure |
| Memory Palace | vector database, knowledge base | We store memories, not vectors |
| Context Window | prompt, system prompt | We manage attention, not text |
| Twin Force | dual system, two components | It's one unified phenomenon |
| Room | category, bucket, index | Physical, navigable space |
| Locus (pl. loci) | slot, position | Architectural precision |
| Illumination | retrieval, fetch | Light emerging from memory |
| Dimming / Decay | deletion, removal, eviction | Natural, biological process |
| Consolidation | backup, save, sync | Sleep-phase memory strengthening |
| The agent's mind | the model, the LLM | We give agents minds |
| Sovereign | local, offline, private | Positive claim of self-ownership |
| Light (noun) | information, data | The medium of cognition |

### 5.2 Voice Examples

| Context | Lumen Voice | Anti-Voice |
|---|---|---|
| Tagline | *Where memory meets context* | "The best memory platform for AI agents" |
| Docs intro | *Lumen gives your agent a palace of memory and a window of attention — two forces, one mind.* | "Lumen is a high-performance memory management system..." |
| Error | *The room you're looking for has faded. Let me search nearby loci.* | "Error 404: Room not found in vector database." |
| Forgetting | *This memory has dimmed with time. It will be released in 7 days unless reinforced.* | "This entry will be deleted from cache in 7 days." |
| CLI output | *Palace: 12 rooms lit. 3 rooms dimming. Context: warm and focused.* | "12 rooms active, 3 rooms pending eviction." |
| Website hero | *Your agent deserves a mind of its own. Lumen is the light between memory and context — running on your hardware, for your data, your way.* | "Enterprise-grade agentic memory infrastructure..." |
| GitHub tagline | *Twin-force memory for sovereign AI agents — on your hardware, with your data.* | "Open-source memory framework for LLMs" |

### 5.3 The Local AI Manifesto

> **Local AI is not a compromise.** It is a choice.
>
> A choice to own your data.  
> A choice to run on hardware you control.  
> A choice to build an agent that remembers *you*, not a demographic.
>
> Lumen exists because local AI needs a mind worthy of its sovereignty. Not a rented database. Not a cloud cache. A **palace** — structured, personal, bounded, alive.
>
> We believe:
> - Forgetting is as important as remembering
> - Memory should decay naturally, not expire arbitrarily
> - Context should assemble from personal history, not generic retrieval
> - An agent's mind should fit on a Raspberry Pi
> - Your data should never leave your device unless *you* choose
>
> We build for one user, one device, one mind at a time.
>
> **Others sell you memory storage. We give your agent a mind.**

---

## 6. Brand Applications

### 6.1 Website / Landing Page

- **Hero**: Animated vesica on Lumen White background. Two circles pulse gently; intersection glows.
- **Typography**: EB Garamond headlines, Inter body.
- **Colour**: Lumen White background, Memory Blue for code, Context Amber for CTAs.
- **Structure**: Top fold = twin-force concept. Scroll = palace metaphor. Bottom = device compatibility grid.
- **Demo**: Interactive TFC visualisation (e slider, a slider, watch palace/context respond).

### 6.2 Documentation

- Split-pane layout: left nav (rooms of the palace), right content (light of the current page).
- Colour-coded sections: Memory (blue tint), Context (amber tint), TFC (purple).
- Every page ends with a "From memory" footnote — quote from earlier docs or related concept.

### 6.3 Terminal / CLI

- Prompt: `lumen@host:~$` with option to use `◆`
- Status bars: `■■■■■■■□□□` (filled = active, empty = dimmed)
- Colours: Memory Blue for storage, Context Amber for queries, Twin Pulse for TFC state

### 6.4 Physical / Swag

- **Stickers**: Vesica logo holographic overlay on intersection
- **T-shirts**: Small chest logo, back text: "Remember to forget."
- **Enamel pins**: Vesica in Memory Blue + Context Amber
- **Patch**: Embroidered vesica for field-device enclosures

---

## 7. Ecosystem Naming

| Component | Name | Rationale |
|---|---|---|
| Core engine | `lumen-core` | The light |
| Twin-force controller | `lumen-tfc` | Structural |
| User research pipeline | `lumen-illuminate` | "Bring to light" user's cognitive structure |
| Visualisation / debug | `lumen-prism` | Split light into components |
| P2P sharing protocol | `lumen-beam` | Direct light between peers |
| Compliance module | `lumen-archive` | Responsible record-keeping |
| Forgetting physics | `lumen-decay` | Natural decay, not deletion |
| Documentation | `lumen-docs` | docs.lumen.ai |
| Contribution program | `lumen-luminary` | Contributors as light-bearers |
| Newsletter / blog | `The Lumen` | Singular, definitive |

---

## 8. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| "Lumen" used by Lumen Technologies (telecom) | Different market (AI memory vs networking). Low confusion. Trademark in AI class. |
| Name too abstract | Grounded by concrete product terms: Palace, Room, Locus, Forget |
| Religious connotation (lumen = light in liturgy) | Brand tone stays secular-technical |
| "Forgetting" concerns enterprise | Frame as *intentional knowledge governance*. Pair with compliance module and audit logs. |
| Sovereign-first limits TAM | Intentional wedge. Enterprise can self-host. The constraint differentiates. |

---

## 9. Summary: The Lumen Identity

| Layer | Expression |
|---|---|
| **Name** | Lumen — light at the intersection of memory and context |
| **Mark** | Vesica piscis — two equal circles, Memory + Context, intersection = Lumen |
| **Colour** | Memory Blue (#1B2A4A) + Context Amber (#E8A838) + Lumen White (#F5F0E8) |
| **Type** | EB Garamond (headings) + Inter (body) + JetBrains Mono (code) |
| **Architecture** | `force::mnemonic` | `force::contextual` | `lumen::controller` |
| **Voice** | Precise, calm, luminous — "Your agent deserves a mind of its own." |
| **Position** | Only high-structure sovereign/edge agentic memory framework |
| **Promise** | Twin forces, unified mind. On your hardware. For your data. Your way. |
