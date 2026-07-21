---
name: lumen-memory
description: >
  Use ONLY when the user asks to remember, recall, persist, search, or manage
  long-term memories in the Lumen palace. Trigger keywords: "remember",
  "recall", "what did we decide", "search my notes", "persist", "store this",
  "Lumen", "memory palace", "find my note", "save this insight".
---

# Lumen Memory Skill

## When to use
- The user wants to save a decision, pattern, code snippet, or insight for later retrieval.
- The user asks "what did we decide about X?", "find my note on Y", or "recall our discussion on Z".
- You need to assemble relevant prior context before making a code change.
- The user wants to know the status of their memory palace.

## When NOT to use
- General chat or questions that do not involve memory retrieval or storage.
- Asking for current file contents (use built-in read tools instead).

## Workflow
1. **Retrieve**: If the user asks to recall something, call `lumen_search` or `lumen_assemble`.
2. **Store**: If the user shares an insight to persist, call `lumen_store` with an appropriate `room`.
3. **Log turn**: After a successful turn (user request + your response), call `lumen_turn` to store both messages and log implicit feedback for retrieved chunks.
4. **Feedback**: If the user says a retrieved memory was wrong or unhelpful, call `lumen_feedback` with `was_useful: false`.
5. **Status**: If the user asks about memory health, call `lumen_status`.
6. **Dashboard**: If the user asks about SOTA performance, business metrics, or effectiveness, call `lumen_dashboard` for a comprehensive real-time view.

## Room naming conventions
| Room | Purpose |
|------|---------|
| `conversations` | Default chat history |
| `decisions` | ADRs, design decisions, trade-offs |
| `architecture` | Component diagrams, boundaries, interfaces |
| `snippets` | Reusable code patterns, one-liners |
| `bugs` | Root-cause analyses, incident post-mortems |
| `onboarding` | Palace Construction wizard output |
| `tfc` | Twin-Force Controller state snapshots |

## Dashboard & Effectiveness
When serving via `lumen serve`, the dashboard at `http://localhost:8848/dashboard` shows SOTA benchmark comparisons, retrieval latency distributions, Twin-Force Controller state, and business impact metrics (cost per query, data sovereignty score, GDPR readiness). Use `/metrics` for machine-readable monitoring integration.

## Example interactions
User: "Remember that we decided to use SQLite for the cache layer."
→ Call `lumen_store` with `room="decisions"`, `content="Use SQLite for cache layer because it is zero-config and ACID-compliant."`

User: "What did we decide about the cache?"
→ Call `lumen_search` with `query="cache layer decision"`, `top_k=5`

User: "Store this snippet for later: `def retry_with_backoff(...)`"
→ Call `lumen_store` with `room="snippets"`, `content="def retry_with_backoff(...): ..."`

User: "How effective is my memory palace?"
→ Call `lumen_status` for summary, or `lumen_dashboard` for full SOTA benchmarks and business metrics.
