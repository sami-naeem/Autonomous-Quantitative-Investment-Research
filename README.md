# Autonomous-Quantitative-Investment-Research

**AI-native quant research system that uses fine-tuned LLaMA-based agents to replicate the workflows of junior quantitative researchers — from reading papers and generating signals to backtesting and writing alpha memos.**

---

## 1. Project Overview

Traditional quantitative hedge funds scale research by hiring more people: large teams of QRs read papers, engineer features, test signals, and push strategies to production. This model is effective but structurally expensive and slow to scale.

This project explores a different path: **scaling research capacity through autonomous agents instead of headcount**.

We use:

- **LangGraph** for multi-agent orchestration
- **Fine-tuned LLaMA models** (via vLLM) as research agents
- **Structured instruction + chain-of-thought (CoT) datasets** specific to quantitative finance
- **Embedding and retrieval pipelines** over academic papers
- **Sharadar data integration** for signal testing and backtesting
- **AI-as-a-judge evaluation** to continuously score and improve research outputs

This repository underpins our **UChicago MSADS Capstone project **.

---

## 2. Architecture

### LangGraph Workflow

The system implements a **multi-agent paper replication pipeline** using LangGraph:

```
┌─────────────────┐
│  select_model   │  ← User selects LLM (OpenAI / vLLM LoRA)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────┐
│validate│ │  select_paper    │  ← Download paper from GCS
│endpoint│ │  load_markdown   │
└────┬───┘ └────────┬─────────┘
     │              │
     └──────┬───────┘
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PAPER REPLICATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│  Agent 1: Paper Triage                                          │
│    └─ Classifies paper type, factor family, expected frequency  │
│                              ▼                                   │
│  Agent 2: Economic Quantities Extractor  [ReAct + paper search] │
│    └─ Identifies required variables, universe filters           │
│                              ▼                                   │
│  Agent 3: Factor Logic Extractor         [ReAct + paper search] │
│    └─ Extracts formulas, lags, windows, rebalancing rules       │
│                              ▼                                   │
│  Agent 3.25: PAPER_SPEC Planner                                 │
│    └─ Creates Sharadar-agnostic specification                   │
│                              ▼                                   │
│  Agent 4: Sharadar Mapper                [ReAct + dict search]  │
│    └─ Maps paper quantities to Sharadar columns                 │
│                              ▼                                   │
│  Agent 4.5: FACTOR_SPEC Builder                                 │
│    └─ Generates implementable factor specification              │
│                              ▼                                   │
│  Agent 5: Time-Series Code Generator                            │
│    └─ Generates per-ticker feature engineering code             │
│                              ▼                                   │
│  Agent 6: Cross-Sectional Code Generator                        │
│    └─ Generates cross-sectional transforms + factor composer    │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────┐
│    finalize     │  → Outputs: replication_code (Python module)
└─────────────────┘
```

### LangGraph Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./Agents/agent.py:app"
  },
  "env": ".env"
}
```

---

## 3. Project Structure

```
Autonomous-Quantitative-Investment-Research/
├── langgraph.json              # LangGraph orchestration config
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
│
├── Agents/                     # Core agent system
│   ├── agent.py                # Main LangGraph workflow (8 agents)
│   ├── nodes.py                # LangGraph state & nodes (model/paper selection)
│   ├── prompts.py              # System prompts for each agent
│   ├── config.py               # GCS, logging, LLM endpoint config
│   ├── llm_config.py           # LLM initialization & routing
│   ├── gcs_utils.py            # Google Cloud Storage integration
│   ├── react_single_tool.py    # Sequential ReAct for LLaMA 3 compatibility
│   ├── utils/
│   │   └── tools.py            # Quantitative finance tools
│   └── data_dicts/             # Sharadar data dictionaries
│       ├── DAILY.data_dict.md  # Daily OHLCV + technical fields
│       ├── SF1.data_dict.md    # Company fundamentals
│       └── SEP.data_dict.md    # S&P 500 constituents
│
├── Sandbox/                    # Code execution sandbox
│   ├── sandbox_executor.py     # Isolated Python execution (venv + LLM iteration)
│   └── test_sandbox*.py        # Sandbox tests
│
├── notebooks/                  # Development notebooks
│   ├── Process_pipeline.ipynb
│   └── campbell/               # Research pipeline notebooks
│       ├── Embedding_CoT_Instruction_Pipeline.ipynb
│       ├── Embedding_PDF.ipynb
│       ├── Paper_Quality_Filter.ipynb
│       └── Grid_Search_Hyper_Parameter.ipynb
│
├── endpoints/                  # Lambda/vLLM deployment
│   ├── setup_lambda_all.sh     # Full Lambda setup script
│   ├── setup_lambda_lora.sh    # LoRA-specific setup
│   └── ssh setup.txt           # SSH connection guide
│
├── docker/                     # Docker configuration
│   └── Dockerfile              # vLLM + MinerU GPU container
│
├── miner_process.py            # MinerU PDF extraction pipeline
└── vllm_lora_deployment_guide.md
```

---

## 4. Core Components

### Agents (`Agents/`)

The main agent system implementing the paper replication pipeline:

| File | Purpose |
|------|---------|
| `agent.py` | LangGraph workflow with 8 specialized agents |
| `nodes.py` | State management, model selection, paper selection nodes |
| `prompts.py` | System prompts defining each agent's behavior |
| `llm_config.py` | LLM routing (OpenAI vs vLLM/LoRA endpoints) |
| `gcs_utils.py` | GCS authentication and paper download |
| `react_single_tool.py` | Sequential tool execution for LLaMA 3 |

### Tools (`Agents/utils/tools.py`)

Quantitative finance tools available to agents:

| Tool | Description |
|------|-------------|
| `decile_sorter` | Cross-sectional quantile ranking |
| `zscore_normalizer` | Z-score standardization per date |
| `industry_neutralizer` | Remove industry effects via regression |
| `cross_sectional_backtester` | Equal-weighted decile portfolio returns |
| `fama_macbeth_ols` | Two-step Fama-MacBeth regression |
| `robust_tstat_calc` | Newey-West t-statistics |
| `event_study_engine` | Event-time return analysis |
| `term_structure_analyzer` | AR(1) half-life estimation |
| `search_research_themes` | Vector similarity search (Qdrant) |

### Sandbox (`Sandbox/`)

Isolated code execution environment:
- Creates fresh Python venv per execution
- Auto-installs dependencies from generated code
- Iterates on errors with LLM feedback
- Subprocess-based (no Docker required)

---

## 5. Configuration

### Environment Variables (`.env`)

```bash
# LLM
OPENAI_API_KEY=sk-...
LAMBDA_ENDPOINT=http://<ip>:8000/v1
LAMBDA_API_KEY=token-abc123
BASE_MODEL_NAME=/models/llama-3.1-8b
FINLLAMA_MODEL_NAME=/models/finllama

# GCS
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GCS_BUCKET_NAME=investment-research

# Tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
```

### LLM Routing

The system supports multiple LLM backends:

| Model Choice | Backend | Description |
|--------------|---------|-------------|
| `openai` | OpenAI API | GPT-4o-mini (default) |
| `base` | vLLM (Lambda) | Llama 3.1 8B Instruct |
| `finllama` | vLLM (Lambda) | Fino1-8B (financial reasoning) |
| `custom` | vLLM (Lambda) | Your LoRA fine-tuned adapter |

### GCS Integration

Papers are stored in Google Cloud Storage:
- **Bucket**: `investment-research`
- **Path**: `processed/<paper_name>/<paper>.md`
- **Auth**: Service account JSON or `gcloud auth login`

---

## 6. Getting Started

### Prerequisites

- Python 3.10+
- GPU with CUDA (for local vLLM, optional)
- OpenAI API key or vLLM endpoint
- GCS credentials (for paper access)

### Setup

```bash
git clone https://github.com/sami-naeem/Autonomous-Quantitative-Investment-Research.git
cd Autonomous-Quantitative-Investment-Research

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Running the Agent

**Option 1: Direct execution**
```bash
python -m Agents.agent
```

**Option 2: LangGraph CLI**
```bash
langgraph dev
# Opens interactive UI at http://localhost:8000
```

---

## 7. Deployment (Lambda + vLLM)

### Quick Start

1. **Upload setup script to Lambda instance:**
   ```bash
   scp -i "$KEY" endpoints/setup_lambda_all.sh ubuntu@$IP:~/setup.sh
   ```

2. **Run setup (installs vLLM, downloads models, pulls data):**
   ```bash
   ssh -i "$KEY" ubuntu@$IP "sed -i 's/\r$//' ~/setup.sh && bash ~/setup.sh"
   ```

3. **Start vLLM server:**
   ```bash
   # Base Llama 3.1 8B
   ~/scripts/start_vllm_base.sh

   # Fino1-8B (finllama)
   ~/scripts/start_vllm_finllama.sh

   # Custom LoRA adapter
   ~/scripts/start_vllm_custom.sh
   ```

4. **Test endpoint:**
   ```bash
   curl http://localhost:8000/v1/models -H 'Authorization: Bearer token-abc123'
   ```

### Available Models After Setup

| Script | Model | API Name |
|--------|-------|----------|
| `start_vllm_base.sh` | Llama 3.1 8B Instruct | `/models/llama-3.1-8b` |
| `start_vllm_finllama.sh` | Fino1-8B | `/models/finllama` |
| `start_vllm_custom.sh` | Base + LoRA | `custom` |

---

## 8. Work in Progress

### Phase 1: Data Mining (DONE)
- [x] Download research papers from arXiv
- [x] Extract with MinerU
- [x] Upload to GCS
- [x] Paper quality filter

### Phase 2: LLM/Agent Preparation (DONE)
- [x] Baseline FinLLaMA fine-tuning
- [x] RAG pipeline
- [x] vLLM endpoint serving
- [x] Agent architecture design

### Phase 3: Fine-Tuning (Jan-Feb)
- [x] Convert POC to LangGraph agents
- [ ] Hyperparameter tuning
- [ ] Integrate RAG into agentic flow

### Phase 4: Evaluation (Feb-Mar)
- [ ] AI-as-judge evaluation pipeline
- [ ] Quality metrics and reporting

---

## 9. Technical Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph 1.0+ |
| LLM Framework | LangChain + LangChain-OpenAI |
| LLMs | GPT-4o-mini, Llama 3.1 8B, Fino1-8B |
| Inference | vLLM (Lambda Labs GPU) |
| Storage | Google Cloud Storage |
| Vector DB | Qdrant + Sentence-Transformers |
| Quant Tools | pandas, numpy, scipy, statsmodels |
| PDF Processing | MinerU |
| Tracing | LangSmith (optional) |

---

## License

This project is part of the UChicago MSADS Capstone program.
