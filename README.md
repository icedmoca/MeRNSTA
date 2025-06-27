
# MeRNSTA — Memory‑Ranked Neuro‑Symbolic Transformer Architecture

> ⚠️ **WARNING: OUTDATED VERSION**
>
> This README describes **v0.3.1** of MeRNSTA.
> The latest version is **v0.6.4** and is **closed source**.
> This repo does **not** reflect the current system architecture.

**Elastic working‑memory cortex for large language models**  
*Version&nbsp;0.3.1 · Research Preview*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/<user>/mernsta/actions)

![Status](https://img.shields.io/badge/status-🚧%20v0.6.4%20Closed--Source-critical?style=for-the-badge&color=red)


---

## 1 · Executive Summary
MeRNSTA fuses an autoregressive Transformer with an **elastic, SQL‑backed, dynamically‑ranked token memory** that scales from laptop SQLite to cloud‑scale FAISS/HNSW with only a config switch.  
Each token is logged with entropy, timestamp, context‑hash and a Bayesian relevance score.  
A **real‑time contradiction resolver** suppresses logits that conflict with high‑confidence memory, yielding *long‑horizon factual coherence* without sacrificing creativity.



---

## 2 · Architecture Overview
```text
┌───────────[1] Base  Transformer──────────┐
│   HF / vLLM  (stream=True)               │
└────┬─────────────────────────────────────┘
     │ tokens(id, logit, pos)
     ▼
┌──────────[2] Intercept Hook──────────────┐
│  • compute entropy H                     │
│  • emit TokenMeta                        │
└────┬─────────────────────────────────────┘
     │  INSERT  (async, batched)
     ▼
┌──────────[3] Cortex Store (elastic)──────┐
│  SQLite ▸ Postgres ▸ pgvector ▸ FAISS    │
│  rows indexed + embedding cache          │
└────┬─────────────┬───────────────────────┘
     │  SELECT     │
     │             ▼
     │   ┌────────[4] Cortex Engine────────┐
     │   │  Bayesian rank Δr               │
     │   │  PPO‑tuned γ                    │
     │   └──────────┬──────────────────────┘
     │ contradiction│
     ▼              ▼
┌──────────[5] Logit Guard─────────────────┐
│  penalise / veto conflicting logits      │
└──────────────────────────────────────────┘
```

**Feedback loop**  
*Token → Memory → Rank/γ → Contradiction → Logit‑bias → Token*

**Performance snapshot**  
* Latency ≈ **1.8 ms / token** @ 50 k rows (RTX 3060)  
* Storage toggle: `config.storage = {sqlite|postgres|faiss}`  

**Demo** → [15 s GIF: hallucination caught](https://giphy.com/gifs/<placeholder-id>)

---

## 3 · Mathematical Core

<table>
  <tr>
    <th style="text-align:center">Component</th>
    <th style="text-align:center">Equation</th>
    <th style="text-align:center">Purpose</th>
  </tr>
  <tr>
    <td><strong>Bayesian Surprise</strong></td>
    <td>$$r_{t+1}(w)=\alpha r_t(w)+(1-\alpha)\,\text{KL}\!\bigl(P(w\mid C_t)\,\parallel\,P(w)\bigr)$$</td>
    <td>Update token relevance on context change</td>
  </tr>
  <tr>
    <td><strong>Logit Penalty</strong></td>
    <td>$$\ell^{\prime}\_{w}=\ell\_{w}-\beta\,\text{Contradict}\bigl(w,M_{\text{hi}}\bigr)$$</td>
    <td>Suppress conflicting token probabilities</td>
  </tr>
  <tr>
    <td><strong>Contradiction Metric</strong></td>
    <td>$$\text{Contradict}(w)=\max_i\!\bigl[I_{\text{rule}}+\gamma\,(1-\cos\theta_{w,i})\bigr]$$</td>
    <td>Hybrid rule + semantic distance (γ auto-tuned)</td>
  </tr>
  <tr>
    <td><strong>Conditional Entropy</strong></td>
    <td>$$H(W\mid C_t)=-\sum\_{v\in V}P(v\mid C_t)\,\log P(v\mid C_t)$$</td>
    <td>Quantify lexical uncertainty</td>
  </tr>
</table>



Derivations → `docs/math.md`.

---

## 4 · Repository Layout
```text
.
├─ llm/         # Transformer wrapper & hooks
├─ storage/     # SQLite / Postgres / FAISS adapters
├─ cortex/      # Rank engine, PPO γ‑tuner, contradiction logic
├─ loop/        # Async generation loop
├─ demos/       # notebooks, quick_eval.py, CLI dashboard
├─ tests/       # latency, coherence, PPO evaluation
└─ config.yaml  # model + elastic storage switch
```

---

## 5 · Installation

### TL;DR
```bash
docker run -it ghcr.io/<user>/mernsta:latest    python loop/run.py --model mistral-7b-instruct --db mernsta.db
```

### Developer setup
```bash
git clone https://github.com/<user>/mernsta.git
cd mernsta
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
*Python 3.10 • transformers • accelerate • sentence‑transformers • aiosqlite • rich • sqlite‑utils • faiss‑cpu*

---

## 6 · Quick Demo
```bash
python demos/quick_eval.py          # 50‑line script—see contradiction event
python demos/cli_dashboard.py       # live cortex view (rich)
tail -f logs/trace.jsonl            # audit trail
```

---

## 7 · Evaluation

| Metric | Script | Δ vs. vanilla |
|--------|--------|---------------|
| Contradiction‑Catch F1 | `tests/test_contrad.py` | **+0.74** |
| Coherence@4k | `tests/test_long.py` | **+38 % BLEU** |
| Latency (ms/token) | `tests/benchmark.py` | **+1.8 ms** |

---

## 8 · Roadmap
- Postgres + pgvector GA  
- FAISS/HNSW (< 1 ms lookup @ 10 M tokens)  
- Web cortex dashboard & compliance PDF exporter  
- Memory‑aware RL fine‑tune module  
- Edge quantisation (<2 GB VRAM)  

---

## 9 · License
Apache 2.0 — open core; hosted “Cortex‑as‑a‑Service” under commercial SLA.

---

## 10 · Citation
```bibtex
@misc{mernsta2025,
  title={Memory-Ranked Neuro-Symbolic Transformer Architecture},
  author={Drake, K. and Contributors},
  year={2025},
  url={https://github.com/<user>/mernsta}
}
```

*Elastic memory · Audited cognition.*
