# VelocityLLM — Website Design Specification

**Purpose:** Complete design + content brief for the VelocityLLM product website, including graphics direction and team information. Hand this to the design/code AI model building the site.

---

## 1. Product Identity

**Name:** VelocityLLM
**Tagline:** A dynamic batching engine for LLM inference serving — proven to raise throughput and guarantee latency SLAs, across multiple models, with real benchmark data.

**Brand theme:** Fire/molten orange — technical, high-energy, "hardware pushed to its limit and intelligently controlled."

**Color palette:**
- Primary: `#FF7A00`
- Accents: `#FF9500`, `#FFC300`
- Background (dark mode default): `#0A0400` / `#1a0800`
- Text: `#F5F0EA` (primary), `#9C9490` (secondary/muted)

**Typography:** Bold geometric sans-serif for headings (Inter, Space Grotesk, or similar); clean sans-serif for body text.

---

## 2. Hero Section — Replace the Phoenix Graphic

**Do not use the phoenix bird illustration from the previous version of this site.**

**New graphic direction:** A high-quality, realistic **3D GPU/chip architecture visualization** — not a simple SVG or CSS animation. Requirements:

- Do not settle on a single production route upfront. Have Antigravity **explore and compare multiple approaches** for generating/rendering this asset, for example:
  - Pre-made professional `.glb`/`.gltf` models sourced from marketplaces such as Sketchfab or Meshy.ai (search terms: "microchip," "GPU die," "circuit board," "processor")
  - Google's **Stitch** (or similar AI design-to-code/asset tools) if it can produce a suitable 3D or high-fidelity animated visual for this concept
  - Any other 3D-generation or 3D-design tool capable of producing a professional, non-cartoonish result (procedural shader-based visuals in Three.js/WebGL are acceptable too, as long as the final quality bar is met — the constraint is visual quality and conceptual fit, not the specific tool)
- Evaluate candidates against this bar: does it look like professional product/hardware marketing render, not a placeholder or an amateur procedural mockup? Reject anything that reads as generic or low-effort, regardless of which tool produced it.
- Whatever asset is chosen, render/integrate it in-browser using **Three.js** (via `@react-three/fiber` + `@react-three/drei` for React integration), or `@google/model-viewer` as a simpler fallback if the asset is a static GLB.
- **Visual concept:** The chip should visually communicate the static-vs-dynamic batching story directly — e.g., one half of the die's cores rendered dim/idle, the other half glowing/active, OR an animated state that transitions from mostly-idle to mostly-active on scroll/load. This should make the core value proposition legible without reading any text.
- Idle-state animation: slow auto-rotation or a subtle emissive pulse, so the hero feels alive even before the user scrolls or interacts.
- Ensure the final asset is optimized/compressed for web (e.g., Draco compression on a GLB) so page load stays fast, regardless of which generation route was used.

**Hero copy:**
- H1: "VelocityLLM"
- Subheadline: one or two sentences on SLA-aware dynamic batching and proven multi-model benchmark results.
- Two CTAs: "View on GitHub" · "See the Benchmarks"

---

## 3. Content Sections

### A. The Problem (short, visual, not a wall of text)
Static batching under-utilizes GPUs and silently breaks latency promises under load — one or two sentences plus a simple before/after visual motif (can reuse the chip's idle vs. active state).

### B. How It Works (visual flow, 3-4 steps, icons + short labels only)
1. Request arrives
2. SLA-aware admission control decides accept / queue / reject
3. Adaptive AIMD controller tunes concurrency to live GPU state
4. Request is served — or honestly declined, never silently late

### C. Benchmark Results (the most important section — animate the numbers)
- Four model cards (TinyLlama-1.1B, Llama-3.2-1B, StableLM-2-1.6B, Qwen2.5-1.5B), each with a large headline stat and a static-vs-dynamic mini comparison bar.
- One unified animated chart: **SLA compliance, static vs. dynamic, across all four models** — this is the strongest, most consistent finding (dynamic = 100% in every model; static as low as 66.7%) and should be the visual centerpiece of the page.
- Numbers should count up/animate into view on scroll.
- **Reference data to use (real, not placeholder):**

| Model | Throughput Change | p99 Latency Change | SLA (Static → Dynamic) |
|---|---|---|---|
| TinyLlama-1.1B | -1.2% | -14.9% | 100% → 100% |
| Llama-3.2-1B | +14.7% | -24.6% | 80% → 100% |
| StableLM-2-1.6B | -1.3% | -32.9% | 80% → 100% |
| Qwen2.5-1.5B | +517.6% | -91.2% | 66.7% → 100% |

### D. Model-Agnostic Claim
Simple horizontal strip of the four tested model names — no table, no verbose text.

### E. Tech Stack (badge/pill row)
Python · vLLM · FastAPI · CUDA · PagedAttention · AIMD Control

### F. Team
See Section 5 below for full details and layout guidance.

### G. Footer
GitHub link · contact · "Built as an industry-guided academic project under Dr. Viomesh K. Singh."

---

## 4. Explicitly Excluded Content

Never include on this site:
- Literal test-case counts (e.g., "29 tests passing")
- Phase-by-phase roadmap/checklist content
- Verbose README-style implementation bullet lists
- Raw ASCII architecture diagrams
- Anything that reads like a GitHub repo page rather than a polished product site

---

## 5. Team Section — Content

Display as a clean 4-card grid (one per member), each card with: photo, name, role, GitHub icon-link, LinkedIn icon-link. College/department line is optional per-card (fill in if available; omit cleanly if not, rather than leaving a placeholder).

**Mentor:** Dr. Viomesh K. Singh

| Name | Role | GitHub | LinkedIn | Photo |
|---|---|---|---|---|
| Kartik R. Pagariya | Scheduler architecture, infrastructure, benchmarking | [github.com/kartikpagariya25](https://github.com/kartikpagariya25) | [linkedin.com/in/kartikpagariya1911](https://www.linkedin.com/in/kartikpagariya1911/) | https://media.licdn.com/dms/image/v2/D4E03AQHfIoFibBhVIA/profile-displayphoto-crop_800_800/B4EZqwcW3eKUAI-/0/1763896816275?e=1790208000&v=beta&t=qhM4SWwE-0WsnL9jFS8wPskQwHB_UivH7tPznjbzEDU |
| Vikrant K. Kadam | Core scheduler engine implementation (Phase 2 & 3) | [github.com/VikrantKadam028](https://github.com/VikrantKadam028) | [linkedin.com/in/vikrantkadam028](https://linkedin.com/in/vikrantkadam028/) | https://media.licdn.com/dms/image/v2/D4D03AQESIR9c5L1XLA/profile-displayphoto-scale_400_400/B4DZ8E_5OrGwAk-/0/1782495287789?e=1790208000&v=beta&t=gWOA3iGPYT_C_8lEnUVOtZKKJ4c1XWFSx081JJhpzMk |
| Pranali D. Yelavikar | Testing & robustness validation | [github.com/pranaliyelavikar14](https://github.com/pranaliyelavikar14) | [linkedin.com/in/pranali-yelavikar](https://www.linkedin.com/in/pranali-yelavikar-2b3178383/) | https://media.licdn.com/dms/image/v2/D4D03AQGPWm01WRkKtw/profile-displayphoto-crop_800_800/B4DZ9fRdCqJwAM-/0/1784009840442?e=1790208000&v=beta&t=YwO7HJSxg9IFq6xJLJxRXCbbBmGdkC82PCmapKP_23I |
| Aditya D. Dengale | Documentation & benchmarking support | [github.com/DevXDividends](https://github.com/DevXDividends) | [linkedin.com/in/adityadengale](https://www.linkedin.com/in/adityadengale/) | https://media.licdn.com/dms/image/v2/D4D03AQEn0yj5zoQwHg/profile-displayphoto-crop_800_800/B4DZjHJmYJH0AM-/0/1755687840857?e=1790208000&v=beta&t=PuSgkFXGYMNt2eb6GCukzn8AR1XSTAHT_V_wdap0-YQ |

> Note: Roles for Pranali and Aditya are placeholder suggestions — confirm the exact roles with them before this goes live. LinkedIn profile photo URLs above are time-limited signed links (they expire); before shipping, download and host each photo locally in the site's own assets rather than hotlinking to LinkedIn's CDN.

---

## 6. Motion & Technical Requirements

- Scroll-triggered fade/slide-in per section (Framer Motion or GSAP ScrollTrigger).
- Transitions: subtle, 200-400ms — avoid slow/heavy motion.
- Framework: React / Next.js.
- Repository structure: place this site's code in a `frontend/` folder inside the main VelocityLLM monorepo, with a `backend/` folder reserved alongside it for any future API/proxy layer.
- Fully responsive, dark mode as default.
- Lazy-load the 3D model and any heavy assets below the fold.
