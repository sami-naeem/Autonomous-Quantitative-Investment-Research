# Autonomous-Quantitative-Investment-Research

**AI-native quant research system that uses fine-tuned LLaMA-based agents to replicate the workflows of junior quantitative researchers — from reading papers and generating signals to backtesting and writing alpha memos.**


## Project Structure

agentic-quant-research/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── src/
│   └── quant_agent/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── base.yaml
│       │   ├── dev.yaml
│       │   └── prod.yaml
│       │
│       ├── data/
│       │   ├── schemas.py
│       │   └── loaders.py
│       │
│       ├── pipelines/
│       │   ├── paper_inventory_pipeline.py
│       │   ├── paper_quality_filter_pipeline.py
│       │   ├── cot_instruction_pipeline.py
│       │   ├── pdf_embedding_pipeline.py
│       │   └── finetuning_pipeline.py
│       │
│       ├── embeddings/
│       │   ├── models.py
│       │   ├── vector_store.py
│       │   └── index_utils.py
│       │
│       ├── agents/
│       │   ├── base_agent.py
│       │   ├── research_agent.py
│       │   ├── replication_agent.py
│       │   ├── signal_agent.py
│       │   └── orchestration.py
│       │
│       ├── tools/
│       │   ├── paper_search_tool.py
│       │   ├── paper_metadata_tool.py
│       │   ├── paper_quality_filter_tool.py
│       │   ├── pdf_embedding_tool.py
│       │   ├── cot_dataset_tool.py
│       │   ├── research_memory_tool.py
│       │   ├── signal_backtest_tool.py
│       │   ├── factor_library_tool.py
│       │   ├── performance_report_tool.py
│       │   ├── risk_report_tool.py
│       │   ├── agent_finetuning_tool.py
│       │   └── agent_evaluation_tool.py
│       │
│       ├── evaluation/
│       │   ├── ai_judge.py
│       │   ├── research_scores.py
│       │   └── reports.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── paths.py
│           └── seed_utils.py
│
├── notebooks/
│   ├── 01_paper_inventory/
│   │   ├── Paper_Quality_Filter.ipynb
│   ├── 02_instruction_cot/
│   │   ├── Embedding_CoT_Instruction_Pipeline.ipynb
│   │   ├── Copy of QuantAgent-FineTuning Test.ipynb
│   ├── 03_pdf_embedding/
│   │   ├── Embedding_PDF.ipynb
│   └── 04_architecture_and_planning/
│       ├── Model_Training_Schematic.ipynb
│       ├── scratchpads.ipynb
│
├── docs/
│   ├── problem-statement/
│   │   ├── Problem_Statement_and_Situational_Analysis.md
│   │   ├── Problem Statement & Situational Analysis v1 (1).docx
│   ├── pitch/
│   │   ├── Pitch Deck (2).pptx
│   │   ├── ADSP 34002 IP01 Capstone I.pptx
│   ├── architecture/
│   │   ├── Model Training Schematic (1).pptx
│   │   ├── system_design.md
│   └── research/
│       ├── literature_review.md
│
├── data/
│   ├── raw/
│   │   ├── arxiv_papers/
│   │   ├── quantconnect_exports/
│   │   ├── metadata/
│   ├── interim/
│   │   ├── cleaned_papers.parquet
│   │   ├── quality_scores.parquet
│   ├── processed/
│   │   ├── filtered_papers.parquet
│   │   ├── instruct_cot_dataset.parquet
│   └── models/
│       ├── llama_archetype_01/
│       ├── llama_archetype_02/
│       ├── tokenizer/
│
├── scripts/
│   ├── run_paper_inventory.py
│   ├── run_paper_quality_filter.py
│   ├── run_cot_instruction_pipeline.py
│   ├── run_pdf_embedding.py
│   ├── train_llama_agents.py
│   ├── evaluate_agents.py
│
├── tests/
│   ├── test_pipelines.py
│   ├── test_embeddings.py
│   ├── test_agents.py
│   ├── test_tools.py
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.yaml



## 1. Project Overview

Traditional quantitative hedge funds scale research by hiring more people: large teams of QRs read papers, engineer features, test signals, and push strategies to production. This model is effective but structurally expensive and slow to scale.

This project explores a different path: **scaling research capacity through autonomous agents instead of headcount**.

We use:

- **Fine-tuned LLaMA models** as research agents
- **Structured instruction + chain-of-thought (CoT) datasets** specific to quantitative finance
- **Embedding and retrieval pipelines** over thousands of academic papers
- **Tooling integration** (e.g., QuantConnect) for signal testing and backtesting
- **AI-as-a-judge evaluation** to continuously score and improve research outputs   

The end goal is an **AI-native quant research lab** where a small human team orchestrates dozens of specialized agents that autonomously propose, validate, and document trading strategies.

This repository underpins the **UChicago MSADS Capstone I project _“The hedge fund industry can be disrupted with Agentic AI”_**, and the associated pitch deck for an **agentic AI-based quantitative hedge fund**.   

---

## 2. System Concept

At full scale, the target architecture is:

- **10 LLaMA-based agent archetypes** (scout, replication researcher, signal evaluator, strategist, alpha librarian, etc.)
- **50 active agents** instantiated from those archetypes
- **~240 signals/year** grouped into **~25 orthogonal themes**
- Portfolio-level **Sharpe ≈ 3.5** at ~$10B GMV capacity, with a core team of ~7 humans.  

This repo focuses on building the **building blocks**:

1. **Paper Inventory + Quality Filter**
   - Ingest and clean a large corpus of quantitative finance papers (e.g., 15k+ arXiv papers).
   - Filter down to a high-quality subset suited for replication and signal discovery. 

2. **Instruction + CoT Dataset Pipeline**
   - Generate 5k–10k+ structured instruction / CoT traces in the prototype phase.
   - Long-run goal: ~50k high-quality instruct/CoT pairs per year for 10 agent archetypes.   

3. **PDF Embedding & Retrieval**
   - Embed full PDFs and metadata into a vector store (Faiss / Pinecone).
   - Support paper lookup, section-level retrieval, and agent memory.

4. **Fine-Tuning & Evaluation**
   - Fine-tune LLaMA with LoRA on the curated instruct/CoT dataset.
   - Evaluate agents with AI-as-a-judge scoring along axes like rigor, completeness, economic reasoning, and documentation quality.   

## 3. Core Pipelines

### 3.1 Paper Inventory and Quality Filter

Goal: build a curated universe of quantitative finance papers suitable for replication and signal extraction.

Main steps:

- Ingest paper metadata and PDFs
- Normalize and clean text and metadata
- Apply rules and learned models to score and filter papers
- Persist a filtered list for downstream pipelines

Implemented in:

- `notebooks/01_paper_inventory/Paper_Quality_Filter.ipynb`
- `src/quant_agent/pipelines/paper_inventory_pipeline.py`
- `src/quant_agent/pipelines/paper_quality_filter_pipeline.py`

Outputs:

- `data/processed/filtered_papers.parquet`
- Optional CSV or JSON exports

### 3.2 Instruction and CoT Generation

Goal: create high-quality instruction and CoT pairs that encode junior quant researcher workflows.

Main steps:

- Start from human-designed tasks and prompts
- Generate CoT traces using LLMs
- Filter and score traces
- Package them into training-ready datasets

Implemented in:

- `notebooks/02_instruction_cot/Embedding_CoT_Instruction_Pipeline.ipynb`
- `notebooks/02_instruction_cot/Copy of QuantAgent-FineTuning Test.ipynb`
- `src/quant_agent/pipelines/cot_instruction_pipeline.py`
- `src/quant_agent/pipelines/finetuning_pipeline.py`

Outputs:

- `data/processed/instruct_cot_dataset.parquet`

### 3.3 PDF Embedding and Retrieval

Goal: turn paper PDFs into searchable vector representations for agent memory and retrieval.

Main steps:

- Load and chunk PDFs into sections or paragraphs
- Compute text embeddings
- Store vectors in a vector database
- Provide retrieval APIs for agents

Implemented in:

- `notebooks/03_pdf_embedding/Embedding_PDF.ipynb`
- `src/quant_agent/pipelines/pdf_embedding_pipeline.py`
- `src/quant_agent/embeddings/`

Outputs:

- Vector indices and mapping files under `data/models/` and `data/processed/`

### 3.4 LLaMA Fine-Tuning and Agent Archetypes

Goal: fine-tune LLaMA using LoRA on the curated datasets to create specialized research agents.

Main steps:

- Load base LLaMA and tokenizer
- Apply LoRA adapters for each archetype
- Train on role-specific slices of the dataset
- Save adapters and evaluation artifacts

Implemented in:

- `notebooks/02_instruction_cot/Copy of QuantAgent-FineTuning Test.ipynb`
- `src/quant_agent/pipelines/finetuning_pipeline.py`
- `src/quant_agent/agents/`

Outputs:

- LoRA adapters under `data/models/`
- Metrics and logs under `evaluation/`

---

## 4. Tools

Agents access functionality through a set of tools implemented as Python modules under `src/quant_agent/tools/`.

The twelve tools are:

1. `paper_search_tool`
2. `paper_metadata_tool`
3. `paper_quality_filter_tool`
4. `pdf_embedding_tool`
5. `cot_dataset_tool`
6. `research_memory_tool`
7. `signal_backtest_tool`
8. `factor_library_tool`
9. `performance_report_tool`
10. `risk_report_tool`
11. `agent_finetuning_tool`
12. `agent_evaluation_tool`

Each tool exposes a narrow, well-defined interface and can be wired into your agent framework of choice.

---

## 5. Evaluation and Metrics

The project tracks both research quality and operational efficiency.

Research quality:

- Strength of empirical design and out-of-sample testing
- Statistical properties of signals
- Documentation and reproducibility
- Breadth and novelty of explored ideas

Operational efficiency:

- Idea velocity and validated hypotheses per period
- Effective researcher-equivalent throughput
- Cost of tuning and running agents

An AI-as-a-judge layer scores agent outputs and filters which traces are added back into the training corpus.

---

## 6. Getting Started

### 6.1 Prerequisites

- Python 3.10 or higher
- GPU with CUDA support recommended
- Access to LLaMA weights
- Access to any external APIs you plan to use (for example, backtesting or evaluation APIs)

### 6.2 Setup

```bash
git clone git@github.com:<your-org>/agentic-quant-research.git
cd agentic-quant-research
python -m venv .venv
source .venv/bin/activate
pip install -e .



-------------------------------------------------------------------------------------

## Phase 1: Data Mining 

    # Download research papers - Seb
        1 - tested local - DONE
        2 - upload to GCS - Done

    # Extract with MinerU - Seb
        1- Test Mineru Locally using vLLM and two other services. - DONE 
        2- uploaded via Gradio UI - DONE
        3- setup pipeline to use mineru and upload to GCS - done

    # Data PRocess in GCS - Sami
        pipeline to convert processed papers into a cleaner folder structure 

    # paper quality control- filter bad or poor quality papers -  Campbell

    # Tokenize extracted text to embeddings for RAG. - Campbell

# Phase 2: LLM/Agent Preparation

    #  baseline finllamn - alejandro
        prepare top 5 papers -> citations index
        prepare handcraft for fine tuning
    # conferences lookup- ale

# Phase 3: FineTune LLM

    #

# Phase 4: Evaluation Pipeline
    # eval paper - alejandro