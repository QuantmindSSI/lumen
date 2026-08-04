# Lumen App Icon — Design Specification

**Version:** 1.0
**Date:** 2026-08-04
**Status:** Draft for community contribution

---

## 1. Overview

This document specifies the primary app icon and favicon for the Lumen local-first agentic memory framework. The icon must communicate four ideas simultaneously:

1. **Memory** — structured, retrievable, persistent
2. **Light / Illumination** — the Latin root of *Lumen*; context brought forth from darkness
3. **Sovereignty** — local, protected, under the user's control
4. **Twin-Force Balance** — the dynamic equilibrium between mnemonic conservation and contextual attention

---

## 2. Core Symbol: The Lumen Core

### Primary Geometry

The icon centers on a **crystalline memory core** — a softened hexagonal prism or diamond silhouette that evokes both a lantern chamber and a palace vault. This outer shell is architectural, not organic.

```
         /\\
        /  \\
       / ✦  \\
      /  /\\  \\
      \\ /  \\ /
       \\    /
        \\  /
         \/
```

*ASCII seed: a four-pointed star (✦) suspended inside a hexagonal lantern pane.*

### The Inner Light

At the exact center of the core sits a **four-pointed star** or **compass rose** rendered as a bright ember. It is not a cold LED or a generic bulb. It is a sustained, warm ignition point — the moment a forgotten memory is retrieved and re-illuminated.

- **Center color**: near-white (#FFFFFF) with a micro-glow bloom
- **Bloom falloff**: transitions through electric blue (#3B82F6) to emerald (#10B981)

### The Twin Rings

Two thin, concentric orbital rings wrap the inner star at slightly different angles, representing the Twin-Force Controller (TFC):

| Ring | Angle Offset | Color | Meaning |
|---|---|---|---|
| Inner | 0° (horizontal) | Emerald (#10B981) | Mnemonic conservation force |
| Outer | 15° tilted | Amber (#F59E0B) | Contextual attention force |

The rings do not close fully; each has a **deliberate gap** at the lower-right quadrant, suggesting open-endedness and dynamic adjustment.

---

## 3. Color System

### Primary Palette (Default)

| Role | Hex | Usage |
|---|---|---|
| Deep Space | `#0B0F19` | Icon background, dark mode default |
| Card / Panel | `#111827` | Secondary fill, shadow depth |
| Border / Line | `#1F2937` | Ring outlines, geometric borders |
| Text / Muted | `#9CA3AF` | Secondary labels |
| Primary Accent | `#10B981` | Inner ring, success states |
| Secondary Accent | `#3B82F6` | Star bloom, info states |
| Tertiary Accent | `#F59E0B` | Outer ring, attention states |
| Alert | `#EF4444` | Rare, for error/eviction states only |

### Monochrome Variant (For print, etched metal, embossed packaging)

- **Etched Copper**: `#B45309` lines on `#111827` background
- **Silver / Steel**: `#D1D5DB` lines on `#0B0F19` background
- Pure black-and-white line art for fax, legal headers, or low-res printing

### Adaptive Background

The icon must work on:
- Dark background (`#0B0F19`) — default
- Light background (`#FFFFFF`) — invert outer shell to `#1F2937`, keep inner star colorful
- Transparent background — shell carries its own subtle backdrop shape

---

## 4. Composition & Proportions

### Grid

- **Canvas**: 1024×1024 px (master). Scales cleanly to 512, 256, 128, 64, 32, 16.
- **Safe margin**: 10% padding on all sides so the icon does not bleed at small sizes.
- **Center weight**: the star sits at optical center (very slightly above mathematical center, ~52% from bottom).

### Hierarchy of Detail by Size

| Size | Detail Level |
|---|---|
| 1024×1024 | Full detail: textured core face, micro-halftone bloom, both rings with gaps, subtle inner shadow |
| 512×512 | Standard app icon: clean rings, visible star bloom, no texture |
| 128×128 | macOS/iOS icon: rings thicken slightly, star remains 4-pointed, gaps on rings still visible |
| 64×64 | Favicon / toolbar: core silhouette + star only; rings may reduce to a single implied ellipse |
| 32×32 | Minimal: hexagonal core + 4-pointed star. Rings omitted. |
| 16×16 | Terminal badge: diamond + dot. Readable at 1:1 pixel scale. |

---

## 5. Typography Pairing (For Wordmark Lockup)

When the wordmark appears alongside the icon, use:

- **Font**: Geometric sans-serif with wide proportions (e.g., *Inter*, *Space Grotesk*, or *Sora*)
- **Tracking**: +5% letter-spacing
- **Case**: ALL CAPS
- **Weight**: Semibold (600)
- **Color**: `#E5E7EB` on dark; `#111827` on light

**Lockup spacing**: The wordmark sits to the right of the icon, baseline-aligned with the star's optical center, separated by 1× the star's width.

---

## 6. Symbolic Details for Close Inspection

The following are **easter eggs** visible only at high resolution or when explained:

1. **The Keystone Gap** — The lower-right gap in the outer ring is shaped like a tiny shield, referencing data sovereignty.
2. **TFC State** — In animated or dynamic versions, the angle between the two rings can shift to reflect the live `e` (conservation) and `a` (attention) values.
3. **Resolution Degradation** — In states where Lumen has downgraded memory precision (INT8, BINARY), the star can dim or the rings can lose saturation — a literal visualization of the optical degradation pipeline.

---

## 7. Technical Requirements

| Output | Format | Notes |
|---|---|---|
| Master | SVG | All geometry as paths; no raster effects |
| App Icon Set | PNG @ 1x, 2x, 3x | iOS, Android, macOS, Windows |
| Favicon | ICO (16, 32, 48) + SVG | Browser tab |
| Social / OpenGraph | PNG 1200×630 | Wordmark lockup centered |
| GitHub Repo Social | PNG 1280×640 | Dark background, single icon centered |
| Print / Sticker | CMYK PDF | Monochrome copper variant |
| Animated | Lottie JSON / GIF 256×256 | Subtle 4-second breathing loop; rings rotate 3° counter-phase |

---

## 8. What to Avoid

| Avoid | Because |
|---|---|
| Brain silhouettes | Overused in AI/memory branding; implies biological rather than structured memory |
| Cloud icons | Contradicts local-first/sovereign positioning |
| Chat bubbles | Reduces the product to a chatbot attachment |
| Gradient meshes or photorealistic glass | Do not scale; fail at 16×16 |
| Human faces or figures | Irrelevant to the system; introduces bias |
| Cold blue-only palettes | Feels corporate and sterile; Lumen is warm, ember-like |

---

## 9. Prompt for LLM Image Generators

Paste this directly into Midjourney, DALL-E 3, or Stable Diffusion:

> A minimalist vector app icon for "Lumen," an AI memory system. Dark charcoal rounded-square background. Centered is a geometric lantern or crystal core: a hexagon with a bright four-pointed star inside. Two thin concentric rings — one emerald green, one amber — orbit the star at slightly different angles, suggesting balanced forces. Single-pixel crisp linework. Flat color blocks, no gradients. Clean enough to read at 64×64 pixels. Transparent safe margin. Style of art-deco precision, sober and architectural.

For **Midjourney**, append: `--v 6 --style raw --ar 1:1`

For **DALL-E 3**, add: "Vector illustration, flat design, centered composition, transparent background implied."

---

## 10. Contribution Path

If you are a designer or want to iterate on this spec:

1. Fork the repo
2. Place drafts in `docs/design/assets/`
3. Open a discussion or PR referencing this spec
4. The maintainers will vote; we aim for convergence, not design-by-committee

---

*This specification is versioned. If the TFC state model changes significantly, the ring symbolism should be reviewed for consistency.*
