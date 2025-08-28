# MeRNSTA — Memory‑Ranked Neuro‑Symbolic Transformer Architecture

**World's First Fully Autonomous Cognitive AGI System**  
*Version 1.0.0 · Production-Ready · Self-Adaptive · Enterprise-Grade*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/icedmoca/mernsta/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![Agents](https://img.shields.io/badge/agents-23%20specialized-orange)](.)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](.)

---

## 🚀 **Quick Start (Unified Full AGI Mode)**

### Install
```bash
git clone https://github.com/icedmoca/mernsta.git
cd mernsta
pip install -r requirements.txt
```

### Optional: start Ollama (custom build) first
```bash
./scripts/start_ollama.sh start
```

### Start MeRNSTA (web + api + agents + background)
```bash
python main.py run
```

**Access Points**
- **💬 Web Chat:** `http://localhost:8000/chat`
- **🔌 REST API:** `http://localhost:8001/docs`
- **📊 Health:** `http://localhost:8000/health`

---

## 📚 **Documentation**

- [**Paper**](docs/paper.md): The full technical paper describing the MeRNSTA architecture.
- [**Predictive Causal Modeling**](docs/phases/PREDICTIVE_CAUSAL_MODELING_README.md): A detailed description of the predictive causal modeling and hypothesis generation system.
- [**Usage Guide**](docs/USAGE.md): Detailed usage examples.
 - [**Repo Map**](info.txt): High-level module interactions across the codebase.

---

## 🧭 Run Modes

- **Unified Full AGI** (recommended): `python main.py run`
- **Web UI only**: `python main.py web --port 8000`
- **API only**: `python main.py api --port 8001`
- **OS Integration (daemon/headless/interactive)**: `python system/integration_runner.py --mode daemon`
- **Enterprise suite** (Celery/Redis/metrics): `python main.py enterprise` or `python start_enterprise.py`

Docker/Compose options are available via `Dockerfile` and `docker-compose.yml`.

---

## ⚙️ Configuration (No Hardcoding)

- All parameters (models, thresholds, ports, routes) live in `config.yaml` and `.env` (see `config/environment.py`).
- Hot-reload support via `config/reloader.py`.
- Examples:
  - Network: `network.api_port`, `network.dashboard_port`, `network.ollama_host`
  - Memory: `memory.hybrid_mode`, `memory.hybrid_backends`, `similarity_threshold`
  - Multi-agent: `multi_agent.agents`, `multi_agent.debate_mode`
  - Visualizer: `visualizer.enable_visualizer`, `visualizer.port`

---

## 🧠 Architecture Overview

- Entry: `main.py` → `system/unified_runner.py` (starts Web UI + System Bridge API + agents + background tasks)
- API: `api/system_bridge.py` exposes `/ask`, `/memory`, `/goal`, `/reflect`, `/personality`, `/status`, and visualizer data endpoints
- Web: `web/main.py` (chat UI, visualizer pages)
- Memory/Cognition: `storage/phase2_cognitive_system.py`, `storage/memory_log.py`, `storage/spacy_extractor.py`, `vector_memory/hybrid_memory.py`
- Agents: `agents/registry.py` + 20+ specialized agents (planner, critic, debater, reflector, etc.)
- Observability: `monitoring/logger.py` (structured logs), `monitoring/metrics.py` (Prometheus)
- Tasks: `tasks/task_queue.py` (Celery: reconciliation, compression, health)

See `docs/paper.md` (Section 3.1.1) for a detailed module interaction appendix.

---

## 🔌 API Quick Examples

Ask:
```bash
curl -s -X POST "http://localhost:8001/ask" \
  -H 'Content-Type: application/json' \
  -d '{"query":"what do I like?"}'
```

Search memory:
```bash
curl -s -X POST "http://localhost:8001/memory" \
  -H 'Content-Type: application/json' \
  -d '{"query_type":"search","query":"my name"}'
```

Visualizer (enable in `config.yaml`):
- Dashboard: `http://localhost:8000/visualizer/`

---

## 🧪 Troubleshooting

- Verify Ollama/tokenizer endpoints:
```bash
python utils/ollama_checker.py --validate
python utils/ollama_checker.py --instructions
```
- API health: `curl http://localhost:8001/health`
- Web health: `curl http://localhost:8000/health`

---

## 🤝 **Contributing**

MeRNSTA is built for extensibility and community contribution:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature-enhancement`
3. **Implement with tests**: All changes must include comprehensive tests
4. **Ensure compatibility**: `pytest` must pass 100%
5. **Submit pull request**: With detailed explanation

---

## 📄 **License & Citation**

Licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file.
