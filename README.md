# 🪐 A.M.A.R.F. (Autonomous Modular Agent Refinement Framework)

A.M.A.R.F. is a production-grade, locally deployed MLOps Workflow Framework designed for designing, fine-tuning, validating, and operating specialized AI Agents in strict, offline (**Air-Gapped**) enterprise topologies. 

The architecture is built from the ground up to operate entirely within **User-Space**. It requires no root/sudo privileges, no Docker containers, and zero external cloud API dependencies, making it fully immune to typical corporate infrastructure restrictions.

---

## 💎 Core Manifesto & Design Philosophies

* **Strict Modularity (Lego Pattern):** Every component is completely isolated. The neural model layers, serverless storage engines, and execution sandboxes communicate via unified Python contracts. You can swap the LLM or Database backend without modifying a single line of business logic.
* **Continuous Refinement Loop:** Instead of basic semantic search, AMARF integrates a complete **Run-Test MLOps Workflow Pipeline**. The system automatically updates database vector spaces, executes pre-training evaluations, tracks matrix deviations, runs gradient optimizations, and deploys execution targets in an endless loop of self-improvement.
* **Dual-Mode Operational Versatility:**
  1. *Passive Mode (Advisory / Bot UI):* Acts as a high-speed semantic RAG assistant, feeding structured local context into chat interfaces or service desks.
  2. *Active Mode (Finite State Machine / Agentic AI):* Acts as an autonomous executive engine. It parses intents, filters commands via hardware guardrails, and safely fires native system utilities inside isolated user-space sandboxes.

---

## 🏗️ Repository Directory Layout & Component Map

```text
AMARF/
├── base_model/               # Offline transformer core storage
│   ├── config.json           # Model network token settings
│   └── vocab.txt             # Autonomous local dictionary matrix
├── core/                     # Enterprise Framework Package Layer
│   ├── __init__.py           # [Empty] Declares core namespace package
│   ├── knowledge/
│   │   └── vector_store.py   # Serverless SQLite + Binary BLOB Vector Store
│   ├── model/
│   │   └── embedding_engine.py # PyTorch DistilBert Mean-Pooling inference core
│   ├── train/
│   │   ├── loss_functions.py # PyTorch Multiple Negatives Ranking Loss (MNRL)
│   │   └── trainer.py        # Gradient tracker & model weight refiner
│   └── control/
│       └── __init__.py       # Sandbox execution engine & Telemetry auditor
├── data/                     # External Dynamic Knowledge Repositories
│   └── sql_theme.json        # Dynamic topic configuration, guardrails & tests
├── storage/                  # Audited execution logs (Generated at runtime)
│   └── telemetry.jsonl       # Structured JSON Lines telemetry dataset
└── workflow.py               # Main Interactive MLOps Platform Orchestrator
```

---

## 📐 Technical Specifications & Hardware Adaptations

* **Neural Core Network:** Localized `DistilBertModel` (~66,000,000 FP32 weights).
* **Vector Topology:** 768-dimensional geometrical space tensor mapping.
* **Auto-Recovery Pipeline:** The framework automatically validates model json structures on startup. If config sheets are missing or empty, AMARF-MODEL immediately regenerates them with reference token distributions.
* **In-Memory RAM Search:** Encodes vectors into compact binary `float32` strings (`BLOB` formatting) inside SQLite. During runtime queries, strings are instantly expanded into matrices via `numpy` for immediate Dot Product Cosine Similarity scoring.
* **Telemetry Auditing:** Real-time microsecond-accurate event logging. Automatically records `timestamp`, `module_name`, `execution_time_ms`, and `loss/score` metrics directly into structured JSONL format for future data analytics.

---

## 🚀 Execution & Deployment Guide (Refinement Lifecycle)

### Step 1: Environment Initialization
Ensure your environment is running Python 3.12 within an isolated workspace. Install the core matrix dependencies:
```bash
pip install torch transformers numpy
```

### Step 2: Zero-Hardcode Topic Loading
AMARF isolates data from code. To inject a new documentation topic (e.g., SQL Optimization, Network Engineering, or Server Logging), do not touch the Python files. Simply place a standardized topic `.json` configuration inside the `data/` directory.

### Step 3: Launch the Interactive Testbench Workflow
Trigger the full continuous improvement loop by running the core orchestrator:
```bash
python workflow.py
```

### Step 4: Interacting with the MLOps Pipeline
Once initialized, the platform executes the following automated workflow:
1. **Load & Clean**: Ingests your dynamic data blocks, clears artifacts, and builds the initial SQLite vector map.
2. **Pre-Test**: Runs your configured `test_suite` query against raw model weights to capture the base selection error.
3. **Train & Refine**: Fires 45 epochs of contrastive PyTorch optimization. The engine pulls matching patterns closer together and pushes alien concepts apart.
4. **Hot-Swap**: Suspends active matrices, performs a safe hot-swap to the freshly trained weights, and fully updates the relational vector space.
5. **Post-Test Execution**: Re-runs the user query. The newly refined model matches the correct block, activates **AMARF-CONTROL**, verifies safety through strict white-lists, and launches the native process via a safe `shell=False` sandbox.
6. **User Refinement Phase**: The script opens a clean CLI control panel, giving you the power to add training epochs, modify the verification queries on the fly, or freeze the current build state.

---

## 🎯 Production Test Suite Passports (Dynamic JSON Format)

To construct or swap a workspace target, adapt your external theme json following this rigid schema configuration:

```json
{
  "agent_name": "SQL_Admin_Agent",
  "agent_idea": "Autonomous SQL Server Optimization",
  "guardrails": {
    "allowed_commands": ["lsblk", "nmcli", "firewall-cmd", "python3", "echo", "ping"],
    "blocked_patterns": ["rm\\s+-rf", "format", "mkfs", "> /dev", "shutdown"]
  },
  "knowledge_chunks": [
    {
      "content": "Database Core Setup: To verify target engine system architectures, use execute: python3 -c \"import sys; print(sys.platform)\"",
      "syntax_type": "documentation_text"
    }
  ],
  "test_suite": {
    "user_query": "identify active sql database engine configuration"
  }
}
```
