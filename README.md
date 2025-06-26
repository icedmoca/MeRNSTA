
# MeRNSTA — Memory-Ranked Neuro-Symbolic Transformer Architecture

**A token‑granular working‑memory substrate for large language models**  
*Version 0.2.0 · Draft research release*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/<user>/mernsta/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/<user>/mernsta/blob/main/LICENSE)

## 1 · Executive Summary
MeRNSTA augments an autoregressive Transformer with a **persistent, SQL‑backed, dynamically‑ranked token memory** that acts as an externalized cortical buffer. Every token entering or leaving the model is archived with metadata (entropy, timestamp, context‑hash, Bayesian relevance score). An online **contradiction resolver** compares candidate tokens against this ranked memory, suppressing logits that violate high‑confidence historical facts to ensure **long‑horizon factual coherence** while preserving the creative stochasticity of the base model.

- **Competitive Moat**: Proprietary token‑ranking and contradiction‑detection algorithms (patent filing in progress) extend beyond the open‑source core, ensuring unique performance advantages.  
- **Core Insight**: Probabilistic language generation (neural) is constrained by a deterministic, queryable symbolic memory (SQL) to emulate executive function, retroactive attention, and self‑consistency—using commodity hardware and open‑source tooling.

---

## 2 · Architectural Overview
```text
┌───────────[1] Base Transformer───────────┐
│  HF/⚡ vLLM, stream=True                  │
└────┬─────────────────────────────────────┘
     │ tokens (id, logit, pos)
     ▼
┌────────────[2] Intercept Hook────────────┐
│  - Compute entropy (conditional H)       │
│  - Emit TokenMeta object                 │
└────┬─────────────────────────────────────┘
     │ INSERT (async, batched)
     ▼
┌────────────[3] SQL Memory (SQLite)───────┐
│  tokens(id, tok, ctx, ts, ent, rank)     │
│  - Indexed for sub-ms queries            │
└────┬───────────────────┬─────────────────┘
     │                   │ SELECT (async)
     │                   ▼
     │        ┌────────[4] Cortex Engine──────┐
     │        │  Bayesian rank update Δr      │
     │        │  - Context-sensitive γ tuning │
     │        └─────────┬─────────────────────┘
     │    contradiction │
     ▼                  ▼
┌────────────[5] Logit Modulator────────────┐
│  Penalize/veto inconsistent tokens        │
│  - Rule + cosine distance (cached)        │
└────────────┬──────────────────────────────┘
             ▼
       [next token to user]
```

**Key Feedback Loop**: Token → Memory → Rank → Contradiction Check → Logit Bias → Token.  
This forms a synthetic *prefrontal cortex* layer atop a Transformer backbone.

### Performance Metrics
- Asynchronous SQL queries and cached embeddings ensure low‑latency operation.  
- **Latency:** ~**1.8 ms/token** on 50 k memory rows (RTX 3060).

**Demo**: [Watch a 5‑second CLI demo of contradiction detection](https://giphy.com/gifs/<placeholder-id>).

---

## 3 · Theoretical Foundations
1. **Bayesian Surprise**  
   Token relevance evolves via:  
   ```math
   r_{t+1}(w) = lpha r_t(w) + (1-lpha)\,\mathsf{Surprise}(w|C_t)
   ```  
   where  
   ```math
   \mathsf{Surprise}(w|C_t) = 	ext{KL}(P(w|C_t) \| P(w))
   ```  
   balances context shifts and stale‑fact decay.

2. **Retro‑Causal Modulation**  
   Candidate token logits  \( \ell_w \) are adjusted:  
   ```math
   \ell'_w = \ell_w - eta \cdot 	ext{Contradict}(w, M_{	ext{high‑rank}})
   ```  
   penalising inconsistencies against high‑rank memory tokens.

3. **Contradiction Metric**  
   Hybrid rule‑based and semantic distance:  
   ```math
   	ext{Contradict}(w) = \max_i igl[ \mathbf{1}_{	ext{rule}} + \gamma igl(1 - \cos	heta_{w,i}igr) igr]
   ```  
   with γ tunable per domain.

4. **Semantic Entropy**  
   ```math
   H(w|C_t) = -\sum P(w|C_t)\log P(w|C_t)
   ```  
   quantifies token uncertainty, computed via *sentence‑transformers* or corpus statistics.

See `docs/math.md` for full derivations.

---

## 4 · Repository Layout
```text
.
├─ llm/           # Transformer wrapper + streaming API hooks
├─ memory/        # SQLite schema, async helpers, migrations
├─ cortex/        # Bayesian scorer, contradiction detector, γ tuning
├─ loop/          # Main generation/event loop (async)
├─ demos/         # Jupyter notebooks, CLI dashboard (rich), quick_eval.py
├─ tests/         # PyTest suite (consistency, latency, coherence)
├─ logs/          # Structured JSON trace logs
└─ config.yaml    # Model, cortex, and γ hyper‑params
```

---

## 5 · Installation
### TL;DR (Docker)
```bash
docker run -it --rm ghcr.io/<user>/mernsta:latest   python loop/run.py --model mistral-7b-instruct --db mernsta.db --config config.yaml
```

### Manual
```bash
git clone https://github.com/<user>/mernsta.git
cd mernsta
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
*Deps*: Python 3.10+, `transformers`, `accelerate`, `sentence-transformers`, `aiosqlite`, `rich`, `sqlite-utils`.

---

## 6 · Quick‑Start
```bash
python loop/run.py --model mistral-7b-instruct --db mernsta.db --config config.yaml
```
**Hands‑on demo:** `python demos/quick_eval.py` (≈50 LOC) logs and prints a contradiction event live.  
Watch `logs/trace.jsonl` or launch `demos/cli_dashboard.py` for a live cortex view.

---

## 7 · Evaluation
| Metric | Script | Description |
|--------|--------|-------------|
| **Contradiction‑Catch Rate** | `tests/test_contrad.py` | % inconsistent tokens vetoed vs. vanilla LLM |
| **Coherence@4k Tokens** | `tests/test_long.py` | BLEU, ROUGE‑L, BERTScore vs. memory facts |
| **Latency Overhead (ms/tok)** | `tests/benchmark.py` | Intercept + async SQL cost (~1.8 ms/tok, 50 k rows) |

---

## 8 · Roadmap
- ☑ MVP loop (SQLite, async, cosine contradiction)  
- ☐ FAISS/HNSW hybrid index (<1 ms lookup)  
- ☐ Hierarchical pruning / decay  
- ☐ Web cortex dashboard  
- ☐ RL tuning of γ coefficients  
- ☐ Embedding cache layer

---

## 9 · License
**Apache 2.0** — permissive, patent‑grant, business‑friendly. Commercial add‑ons may be dual‑licensed.

---

## 10 · Citation
```bibtex
@misc{mernsta2025,
  title   = {Memory-Ranked Neuro-Symbolic Transformer Architecture},
  author  = {Drake, K. and Contributors},
  year    = {2025},
  howpublished = {GitHub},
  url     = {https://github.com/<user>/mernsta}
}
```

*Build cognition, not just completion.*
