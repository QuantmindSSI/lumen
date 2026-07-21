---
name: lumen-memory
description: >
  Use when the user asks to remember, recall, persist, or search across
  project decisions, code snippets, architecture notes, or conversation
  history stored in the Lumen memory palace. Trigger keywords:
  "remember", "recall", "what did we decide", "search my notes",
  "persist", "store this", "Lumen".
---

# Lumen Memory Skill

## When to use
- The user wants to save a decision, pattern, or insight for later retrieval.
- The user asks "what did we decide about X?" or "find my note on Y".
- You need to assemble relevant prior context before making a code change.

## Workflow
1. If the user asks to retrieve: call `lumen_search` or `lumen_assemble`.
2. If the user shares an insight to persist: call `lumen_store` with an appropriate `room` (e.g., `architecture`, `decisions`, `snippets`).
3. After a successful turn (user request + your response): call `lumen_turn` to store both messages and log implicit feedback.
4. If the user says a retrieved memory was wrong or unhelpful: call `lumen_feedback` with `was_useful: false`.

## Room naming conventions
| Room | Purpose |
|------|---------|
| `conversations` | Default chat history |
| `decisions` | ADRs, design decisions, trade-offs |
| `architecture` | Component diagrams, boundaries, interfaces |
| `snippets` | Reusable code patterns, one-liners |
| `bugs` | Root-cause analyses, incident post-mortems |
| `onboarding` | Palace Construction wizard output |

## Example
User: "Remember that we decided to use SQLite for the cache layer."
→ Call `lumen_store` with `room="decisions"`, `content="Use SQLite for cache layer because..."`
