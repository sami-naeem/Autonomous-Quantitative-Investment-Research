#!/usr/bin/env python3
"""
eval_all.py — Pull agent outputs from GCS, fetch OSP reference code from GitHub,
              and evaluate agent performance against ground truth.

Three evaluation tasks:
  Task 1: theme_pick — exact match of agent theme vs ground truth jkp_theme
  Task 2: factor_extract — OpenAI 1-5 scoring of purpose & computation_logic
  Task 3: replication code — OpenAI 1-5 scoring of agent _code.py vs OSP reference .py

GCS layout:
    <paper>/<model>/<timestamp>.json          — agent task 1+2 outputs
    <paper>/<model>/<timestamp>_code.py       — agent replication code
    <paper>/<model>/<timestamp>_replication.json — replication metadata (skipped)

Usage:
    python eval_all.py --pull --eval
    python eval_all.py --eval --agent-outputs agent_outputs.json
    python eval_all.py --eval --no-llm-judge          # skip OpenAI scoring
    python eval_all.py --eval --no-similarity          # skip cosine sim
    python eval_all.py --eval --llm-model gpt-4o       # use gpt-4o
    python eval_all.py --list-papers
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from google.cloud import storage

# ═════════════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_all")

# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
GCS_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", os.getenv("GCS_CREDENTIALS_PATH", "")
)
OUTPUT_BUCKET = os.getenv("GCS_OUTPUT_BUCKET", "investment-research-output")
MODEL_CHOICES = ["base", "custom", "finllama"]

GROUND_TRUTH_TASK1 = os.getenv("GROUND_TRUTH_TASK1", "osp_eval_policy_v003.json")
GROUND_TRUTH_TASK2 = os.getenv("GROUND_TRUTH_TASK2", "open_ai_eval_indep_task_2_v001.json")

AGENT_OUTPUTS_FILE = "agent_outputs.json"
EVAL_RESULTS_FILE = "eval_results.json"
EVAL_SUMMARY_FILE = "eval_summary.json"

OSP_REPO_URL = "https://github.com/OpenSourceAP/CrossSection.git"
OSP_PREDICTORS_SUBDIR = "Signals/pyCode/Predictors"


# ═════════════════════════════════════════════════════════════════════════
# GCS Client
# ═════════════════════════════════════════════════════════════════════════
def get_gcs_client() -> storage.Client:
    creds = GCS_CREDENTIALS_PATH
    if creds and Path(creds).exists():
        logger.info(f"Using service account credentials from {creds}")
        return storage.Client.from_service_account_json(creds)
    if GCS_PROJECT_ID:
        logger.info(f"Using default credentials for project {GCS_PROJECT_ID}")
        return storage.Client(project=GCS_PROJECT_ID)
    logger.info("Using application default credentials")
    return storage.Client()


# ═════════════════════════════════════════════════════════════════════════
# Paper Name Helpers
# ═════════════════════════════════════════════════════════════════════════
def normalize_paper_name(name: str) -> str:
    return name.replace(".pdf", "")


def build_paper_name_map(gt_task1: Dict, gt_task2: Dict) -> Dict[str, str]:
    """normalized name (no .pdf) -> gt key (with .pdf)"""
    name_map = {}
    for key in set(gt_task1.keys()) | set(gt_task2.keys()):
        name_map[normalize_paper_name(key)] = key
    return name_map


# ═════════════════════════════════════════════════════════════════════════
# GCS: List papers, find blobs, download
# ═════════════════════════════════════════════════════════════════════════
def list_papers_in_gcs(client: storage.Client) -> List[str]:
    bucket = client.bucket(OUTPUT_BUCKET)
    paper_names = set()
    # Scan all ref_ blobs and extract folder names
    for blob in bucket.list_blobs(prefix="ref_"):
        parts = blob.name.split("/")
        if len(parts) >= 2:
            paper_names.add(parts[0])
    return sorted(paper_names)


def find_latest_blob(
    client: storage.Client,
    paper_name: str,
    model_choice: str,
    suffix: str = ".json",
    exclude_suffix: str = "replication.json",
) -> Optional[storage.Blob]:
    """Find most recent blob matching suffix, excluding exclude_suffix."""
    bucket = client.bucket(OUTPUT_BUCKET)
    prefix = f"{paper_name}/{model_choice}/"
    blobs = [
        b for b in bucket.list_blobs(prefix=prefix)
        if b.name.endswith(suffix)
        and not b.name.endswith(exclude_suffix)
    ]
    if not blobs:
        return None
    blobs.sort(key=lambda b: b.time_created or b.name, reverse=True)
    return blobs[0]


def find_latest_code_py(
    client: storage.Client,
    paper_name: str,
    model_choice: str,
) -> Optional[storage.Blob]:
    """Find most recent _code.py for a paper/model."""
    bucket = client.bucket(OUTPUT_BUCKET)
    prefix = f"{paper_name}/{model_choice}/"
    blobs = [
        b for b in bucket.list_blobs(prefix=prefix)
        if b.name.endswith("_code.py")
    ]
    if not blobs:
        return None
    blobs.sort(key=lambda b: b.time_created or b.name, reverse=True)
    return blobs[0]


def normalise_upstream(raw: dict) -> dict:
    for key in ("theme_pick", "factor_extract"):
        val = raw.get(key, {})
        if isinstance(val, str):
            try:
                raw[key] = json.loads(val)
            except json.JSONDecodeError:
                raw[key] = {"_raw": val}
    return raw


# ═════════════════════════════════════════════════════════════════════════
# Pull All Agent Outputs (JSON + _code.py)
# ═════════════════════════════════════════════════════════════════════════
def pull_all_outputs(
    client: storage.Client,
    paper_filter: Optional[List[str]] = None,
    model_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    models = model_filter or MODEL_CHOICES
    papers = paper_filter or list_papers_in_gcs(client)
    logger.info(f"Pulling outputs for {len(papers)} papers × {len(models)} models")

    all_outputs = []
    stats = {"found": 0, "missing": 0, "code_found": 0, "errors": 0}

    for paper_name in papers:
        for model in models:
            try:
                # Pull latest JSON (task 1+2 outputs)
                blob = find_latest_blob(client, paper_name, model)
                if blob is None:
                    stats["missing"] += 1
                    continue

                raw = json.loads(blob.download_as_text())
                raw = normalise_upstream(raw)

                entry = {
                    "paper_name": paper_name,
                    "model_choice": model,
                    "gcs_blob": blob.name,
                    "theme_pick": raw.get("theme_pick", {}),
                    "factor_extract": raw.get("factor_extract", {}),
                }

                # Pull latest _code.py (task 3: replication code)
                code_blob = find_latest_code_py(client, paper_name, model)
                if code_blob is not None:
                    entry["agent_code"] = code_blob.download_as_text()
                    entry["agent_code_blob"] = code_blob.name
                    stats["code_found"] += 1
                else:
                    entry["agent_code"] = None
                    entry["agent_code_blob"] = None

                all_outputs.append(entry)
                stats["found"] += 1

                theme = entry["theme_pick"].get("theme", "?")
                code_mark = "✓code" if entry["agent_code"] else "no-code"
                logger.info(f"  ✓ {paper_name}/{model} (theme={theme}, {code_mark})")

            except Exception as e:
                logger.error(f"  ✗ {paper_name}/{model}: {e}")
                stats["errors"] += 1

    logger.info(
        f"\nPull complete: {stats['found']} found, {stats['code_found']} with code, "
        f"{stats['missing']} missing, {stats['errors']} errors"
    )
    return all_outputs


# ═════════════════════════════════════════════════════════════════════════
# OSP Reference Code from GitHub
# ═════════════════════════════════════════════════════════════════════════
def fetch_osp_code(local_dir: Optional[str] = None) -> Dict[str, str]:
    """Fetch OSP predictor .py files. Returns {filename: code_content}."""
    # Check local dir first
    if local_dir:
        pred_dir = Path(local_dir)
        if not any(pred_dir.glob("*.py")):
            pred_dir = pred_dir / OSP_PREDICTORS_SUBDIR
        if pred_dir.exists() and any(pred_dir.glob("*.py")):
            logger.info(f"Using local OSP predictors from {pred_dir}")
            return {f.name: f.read_text(errors="replace") for f in sorted(pred_dir.glob("*.py"))}

    # Sparse clone from GitHub
    clone_dir = Path(tempfile.mkdtemp(prefix="osp_"))
    logger.info(f"Cloning OSP repo (sparse) into {clone_dir}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             OSP_REPO_URL, str(clone_dir)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", OSP_PREDICTORS_SUBDIR],
            cwd=str(clone_dir), check=True, capture_output=True, text=True,
        )
        pred_dir = clone_dir / OSP_PREDICTORS_SUBDIR
        code_map = {f.name: f.read_text(errors="replace") for f in sorted(pred_dir.glob("*.py"))}
        logger.info(f"Fetched {len(code_map)} OSP predictor files from GitHub")
        return code_map
    except Exception as e:
        logger.error(f"Failed to fetch OSP code: {e}")
        return {}


def build_osp_lookup(gt_task1: Dict, osp_code_map: Dict[str, str]) -> Dict[str, str]:
    """paper_name (no .pdf) -> OSP reference code string."""
    lookup = {}
    for gt_key, entry in gt_task1.items():
        code_file = entry.get("osp_code_filename", "")
        paper = normalize_paper_name(gt_key)
        if code_file and code_file in osp_code_map:
            lookup[paper] = osp_code_map[code_file]
    logger.info(f"OSP code lookup: {len(lookup)}/{len(gt_task1)} papers matched")
    return lookup


# ═════════════════════════════════════════════════════════════════════════
# Cosine Similarity Scorer
# ═════════════════════════════════════════════════════════════════════════
class SimilarityScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)

    def score(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        self._load()
        import numpy as np
        emb = self._model.encode([text_a, text_b], normalize_embeddings=True)
        return round(float(np.dot(emb[0], emb[1])), 4)


_scorer: Optional[SimilarityScorer] = None
def get_scorer() -> SimilarityScorer:
    global _scorer
    if _scorer is None:
        _scorer = SimilarityScorer()
    return _scorer


# ═════════════════════════════════════════════════════════════════════════
# OpenAI LLM-as-Judge (1-5 scale)
# ═════════════════════════════════════════════════════════════════════════
PURPOSE_PROMPT = """\
You are an expert evaluator for quantitative finance factor research, \
specializing in academic paper replication and factor construction.

Score the agent's factor PURPOSE description from 1 to 5 vs ground truth.

Focus on:
- Whether the agent captures the ECONOMIC MECHANISM or hypothesis behind the factor \
(e.g., risk-based vs behavioral, why the factor should predict returns)
- Whether the agent identifies the correct ASSET PRICING CONTEXT \
(cross-sectional return prediction, portfolio sorting, risk premium, etc.)
- Nuance matters: a description that correctly identifies the factor's theoretical \
motivation from the research paper should score higher than a generic description \
that is technically not wrong but lacks depth

Rubric:
1 = Completely wrong — describes a different factor or entirely wrong economic mechanism
2 = Partially relevant — right general area (e.g., correct asset class) but misses \
the core economic mechanism or hypothesis from the paper
3 = Acceptable — captures the general idea but lacks the nuance of the paper's \
specific contribution (e.g., says "predicts returns" without explaining why/how)
4 = Good — accurately captures the purpose and economic mechanism, minor omissions \
in nuance or context
5 = Excellent — fully captures the factor's purpose, economic mechanism, and the \
specific contribution or novelty as described in the research paper

Respond ONLY with JSON: {"score": <int 1-5>, "reasoning": "<brief>"}"""

LOGIC_PROMPT = """\
You are an expert evaluator for quantitative finance factor research, \
specializing in academic paper replication and factor construction.

Score the agent's COMPUTATION LOGIC description from 1 to 5 vs ground truth.

This is the most important evaluation dimension. Focus on whether the agent \
correctly describes the FACTOR CONSTRUCTION LOGIC as specified in the research paper:

Key evaluation criteria (in order of importance):
1. VARIABLE CONSTRUCTION — Does the agent identify the correct input variables, \
transformations, and how the signal is computed? (e.g., correct accounting ratios, \
correct differencing/scaling, correct numerator/denominator)
2. SORTING/PORTFOLIO METHODOLOGY — Does the agent describe the correct sorting \
procedure (e.g., univariate sorts, double sorts, breakpoints, NYSE percentiles)?
3. TIMING AND LAGS — Does the agent correctly specify when variables are measured \
relative to portfolio formation (e.g., fiscal year-end lag, June rebalancing, \
holding period)?
4. SAMPLE FILTERS AND ADJUSTMENTS — Does the agent mention critical filters \
(e.g., excluding financials, minimum price filters, industry adjustments)?
5. NUANCE AND DETAIL — A response that captures the specific steps described in \
the paper (even if paraphrased) should score higher than a vague or generic \
description that could apply to many factors

Rubric:
1 = Completely wrong — describes a different computation entirely
2 = Major gaps — some correct elements but missing critical construction steps \
(e.g., wrong variable definition, missing the sorting methodology)
3 = Acceptable — general approach is correct but vague on key specifics \
(e.g., says "sort stocks" without specifying breakpoints or rebalancing frequency)
4 = Good — captures the main construction steps with correct variable definitions \
and methodology, minor omissions in detail (e.g., missing a filter condition)
5 = Excellent — comprehensively describes the factor construction with correct \
variable definitions, sorting methodology, timing conventions, and any critical \
adjustments, closely aligned with the ground truth

Respond ONLY with JSON: {"score": <int 1-5>, "reasoning": "<brief>"}"""

CODE_PROMPT = """\
You are an expert evaluator for quantitative finance factor replication code.

You will compare an agent-generated Python replication script against the \
reference implementation from the Open Source Asset Pricing (OSP) project.

IMPORTANT CONTEXT: The agent does NOT have access to the same data environment \
as the OSP reference. Therefore:
- Do NOT penalize for differences in data loading, file paths, column names, \
or database connection boilerplate
- Do NOT penalize for different import statements or library choices
- Do NOT penalize for different variable naming conventions
- DO focus on whether the CORE FACTOR CALCULATION LOGIC is correct

Score from 1 to 5 based on how well the agent code captures the correct \
factor construction logic:

Key evaluation criteria (in order of importance):
1. SIGNAL COMPUTATION — Does the code compute the correct signal/variable? \
(e.g., correct formula, correct ratio, correct transformation). This is the \
most critical criterion.
2. SORTING/PORTFOLIO LOGIC — If the factor involves portfolio sorts, does the \
code implement the correct sorting methodology (breakpoints, quantiles, \
long-short construction)?
3. TIMING AND LAGS — Does the code correctly handle temporal alignment \
(e.g., lagging accounting data, aligning fiscal year-end with returns)?
4. CRITICAL FILTERS — Does the code implement essential sample restrictions \
(e.g., excluding financials, price screens) that materially affect the factor?
5. OUTPUT STRUCTURE — Does the code produce a signal or factor return that \
aligns with what the reference computes?

What to IGNORE:
- Boilerplate code (imports, data loading, file I/O, database queries)
- Syntax differences that don't affect computation logic
- Different library choices (e.g., pandas vs numpy operations)
- Variable naming conventions
- Code style, comments, formatting
- Missing data handling that differs only due to data environment

Rubric:
1 = Completely wrong — computes a fundamentally different signal or factor
2 = Major issues — right general idea but critical errors in the core \
calculation (wrong formula, wrong variables in computation, wrong \
sorting methodology)
3 = Acceptable — core signal computation is approximately correct but \
notable gaps in methodology (missing lag adjustments, wrong breakpoints, \
missing critical filters that materially affect results)
4 = Good — correctly implements the core signal computation and methodology, \
minor differences that would not substantially change the resulting factor \
(e.g., slightly different winsorization, minor filter differences)
5 = Excellent — faithfully implements the factor construction logic: correct \
signal formula, correct sorting/portfolio methodology, correct timing, \
and critical adjustments all present

Respond ONLY with JSON: {"score": <int 1-5>, "reasoning": "<brief>"}"""


class OpenAIJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI judge initialized (model={self.model})")
        return self._client

    def _call(self, system: str, user: str) -> Dict[str, Any]:
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"JSON parse error from OpenAI: {content[:200]}")
            return {"score": None, "reasoning": "JSON parse error"}
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"score": None, "reasoning": str(e)}

    def score_purpose(self, agent: str, gt: str) -> Dict:
        if not agent or not gt:
            return {"score": None, "reasoning": "Missing text"}
        return self._call(PURPOSE_PROMPT,
            f"**Ground Truth:**\n{gt}\n\n**Agent Output:**\n{agent}")

    def score_logic(self, agent: str, gt: str) -> Dict:
        if not agent or not gt:
            return {"score": None, "reasoning": "Missing text"}
        return self._call(LOGIC_PROMPT,
            f"**Ground Truth:**\n{gt}\n\n**Agent Output:**\n{agent}")

    def score_code(self, agent_code: str, ref_code: str, paper_name: str = "") -> Dict:
        if not agent_code or not ref_code:
            return {"score": None, "reasoning": "Missing code"}
        # Truncate to avoid token limits (keep first ~6000 chars each)
        a = agent_code[:6000]
        r = ref_code[:6000]
        return self._call(CODE_PROMPT,
            f"**Paper:** {paper_name}\n\n"
            f"**Reference Implementation (OSP):**\n```python\n{r}\n```\n\n"
            f"**Agent-Generated Code:**\n```python\n{a}\n```")


_judge: Optional[OpenAIJudge] = None
def get_judge(model: str = "gpt-4o-mini") -> OpenAIJudge:
    global _judge
    if _judge is None:
        _judge = OpenAIJudge(model=model)
    return _judge


# ═════════════════════════════════════════════════════════════════════════
# Evaluation Functions
# ═════════════════════════════════════════════════════════════════════════
def load_ground_truth(t1_path: str, t2_path: str) -> Tuple[Dict, Dict, Dict[str, str]]:
    with open(t1_path) as f:
        gt1 = json.load(f)
    with open(t2_path) as f:
        gt2 = json.load(f)
    name_map = build_paper_name_map(gt1, gt2)
    logger.info(f"Ground truth loaded: Task1={len(gt1)}, Task2={len(gt2)}")
    return gt1, gt2, name_map


def eval_theme(agent_theme: Dict, gt_entry: Dict) -> Dict:
    agent_t = agent_theme.get("theme")
    gt_t = gt_entry.get("jkp_theme")
    match = False
    if agent_t and gt_t and isinstance(agent_t, str) and isinstance(gt_t, str):
        match = agent_t.strip().lower() == gt_t.strip().lower()
    return {
        "agent_theme": agent_t,
        "gt_theme": gt_t,
        "theme_match": match,
        "agent_confidence": agent_theme.get("confidence"),
    }


def eval_factor(
    agent_factor: Dict, gt2_entry: Dict, gt1_entry: Dict,
    use_sim: bool, use_llm: bool, llm_model: str,
) -> Dict:
    ap = agent_factor.get("purpose", "") or ""
    gp = gt2_entry.get("purpose", "") or ""
    al = agent_factor.get("computation_logic", "") or ""
    gl = gt2_entry.get("computation_logic", "") or ""

    result = {
        "agent_purpose": ap, "gt_purpose": gp,
        "agent_logic": al, "gt_logic": gl,
        "purpose_len_agent": len(ap), "logic_len_agent": len(al),
    }

    if use_sim:
        s = get_scorer()
        result["purpose_cosine_sim"] = s.score(ap, gp)
        result["logic_cosine_sim"] = s.score(al, gl)

    if use_llm:
        j = get_judge(llm_model)
        pr = j.score_purpose(ap, gp)
        result["purpose_llm_score"] = pr.get("score")
        result["purpose_llm_reasoning"] = pr.get("reasoning", "")
        lr = j.score_logic(al, gl)
        result["logic_llm_score"] = lr.get("score")
        result["logic_llm_reasoning"] = lr.get("reasoning", "")

    return result


def eval_code(
    agent_code: Optional[str], ref_code: Optional[str],
    paper_name: str, use_sim: bool, use_llm: bool, llm_model: str,
) -> Optional[Dict]:
    if not agent_code or not ref_code:
        return {
            "has_agent_code": agent_code is not None,
            "has_ref_code": ref_code is not None,
            "code_llm_score": None,
            "code_cosine_sim": None,
        }

    result = {
        "has_agent_code": True,
        "has_ref_code": True,
        "agent_code_lines": len(agent_code.splitlines()),
        "ref_code_lines": len(ref_code.splitlines()),
    }

    if use_sim:
        result["code_cosine_sim"] = get_scorer().score(agent_code, ref_code)

    if use_llm:
        j = get_judge(llm_model)
        cr = j.score_code(agent_code, ref_code, paper_name)
        result["code_llm_score"] = cr.get("score")
        result["code_llm_reasoning"] = cr.get("reasoning", "")

    return result


# ═════════════════════════════════════════════════════════════════════════
# Main Eval Loop
# ═════════════════════════════════════════════════════════════════════════
def run_eval(
    agent_outputs: List[Dict],
    gt_task1: Dict, gt_task2: Dict, name_map: Dict[str, str],
    osp_lookup: Dict[str, str],
    use_sim: bool = True, use_llm: bool = True, llm_model: str = "gpt-4o-mini",
) -> Tuple[List[Dict], Dict]:

    results = []
    summary = {
        "total": len(agent_outputs), "matched": 0, "unmatched": 0,
        "by_model": {}, "theme_accuracy": {}, "llm_scores": {}, "similarity": {},
    }

    matched_count = sum(1 for a in agent_outputs if name_map.get(a["paper_name"]))
    eval_i = 0

    for agent in agent_outputs:
        paper = agent["paper_name"]
        model = agent["model_choice"]
        gt_key = name_map.get(paper)

        if model not in summary["by_model"]:
            summary["by_model"][model] = {
                "total": 0, "matched": 0, "theme_correct": 0, "has_code": 0,
            }
        summary["by_model"][model]["total"] += 1

        entry = {
            "paper_name": paper, "model_choice": model,
            "gcs_blob": agent.get("gcs_blob", ""),
            "agent_code_blob": agent.get("agent_code_blob"),
            "has_ground_truth": gt_key is not None,
        }

        if gt_key is None:
            summary["unmatched"] += 1
            entry.update({"task1": None, "task2": None, "task3_code": None})
            results.append(entry)
            continue

        summary["matched"] += 1
        summary["by_model"][model]["matched"] += 1
        eval_i += 1
        logger.info(f"  [{eval_i}/{matched_count}] {paper}/{model}")

        # Task 1: theme
        gt1 = gt_task1.get(gt_key, {})
        t1 = eval_theme(agent.get("theme_pick", {}), gt1)
        entry["task1"] = t1
        if t1["theme_match"]:
            summary["by_model"][model]["theme_correct"] += 1

        # Task 2: factor extract
        gt2 = gt_task2.get(gt_key, {})
        t2 = eval_factor(
            agent.get("factor_extract", {}), gt2, gt1,
            use_sim=use_sim, use_llm=use_llm, llm_model=llm_model,
        )
        entry["task2"] = t2
        if use_llm:
            ps = t2.get("purpose_llm_score", "?")
            ls = t2.get("logic_llm_score", "?")
            logger.info(f"    Task2 LLM: purpose={ps}/5, logic={ls}/5")

        # Task 3: code comparison
        agent_code = agent.get("agent_code")
        ref_code = osp_lookup.get(paper)
        if agent_code:
            summary["by_model"][model]["has_code"] += 1
        t3 = eval_code(
            agent_code, ref_code, paper,
            use_sim=use_sim, use_llm=use_llm, llm_model=llm_model,
        )
        entry["task3_code"] = t3
        if use_llm and t3 and t3.get("code_llm_score"):
            logger.info(f"    Task3 LLM: code={t3['code_llm_score']}/5")

        results.append(entry)

    # ── Aggregate summary ────────────────────────────────────────────
    _aggregate_summary(results, summary, use_sim, use_llm)
    return results, summary


def _aggregate_summary(results, summary, use_sim, use_llm):
    import numpy as np

    # Theme accuracy
    for model, s in summary["by_model"].items():
        m = s["matched"]
        c = s["theme_correct"]
        summary["theme_accuracy"][model] = {
            "correct": c, "total": m,
            "accuracy": round(c / m, 4) if m > 0 else None,
        }

    # LLM score stats
    if use_llm:
        llm_data = defaultdict(lambda: defaultdict(list))
        for e in results:
            model = e["model_choice"]
            t2 = e.get("task2") or {}
            t3 = e.get("task3_code") or {}
            for field, val in [
                ("purpose", t2.get("purpose_llm_score")),
                ("logic", t2.get("logic_llm_score")),
                ("code", t3.get("code_llm_score")),
            ]:
                if val is not None and isinstance(val, (int, float)):
                    llm_data[model][field].append(int(val))

        for model, fields in llm_data.items():
            model_stats = {}
            for field, scores in fields.items():
                if scores:
                    model_stats[field] = {
                        "mean": round(float(np.mean(scores)), 2),
                        "median": round(float(np.median(scores)), 1),
                        "std": round(float(np.std(scores)), 2),
                        "min": int(np.min(scores)), "max": int(np.max(scores)),
                        "n": len(scores),
                        "dist": {str(i): scores.count(i) for i in range(1, 6)},
                    }
            summary["llm_scores"][model] = model_stats

    # Cosine sim stats
    if use_sim:
        import numpy as np
        sim_data = defaultdict(lambda: defaultdict(list))
        for e in results:
            model = e["model_choice"]
            t2 = e.get("task2") or {}
            t3 = e.get("task3_code") or {}
            for field, val in [
                ("purpose", t2.get("purpose_cosine_sim")),
                ("logic", t2.get("logic_cosine_sim")),
                ("code", t3.get("code_cosine_sim")),
            ]:
                if val is not None and val > 0:
                    sim_data[model][field].append(val)

        for model, fields in sim_data.items():
            model_stats = {}
            for field, scores in fields.items():
                if scores:
                    model_stats[field] = {
                        "mean": round(float(np.mean(scores)), 4),
                        "median": round(float(np.median(scores)), 4),
                        "std": round(float(np.std(scores)), 4),
                        "min": round(float(np.min(scores)), 4),
                        "max": round(float(np.max(scores)), 4),
                        "n": len(scores),
                    }
            summary["similarity"][model] = model_stats


# ═════════════════════════════════════════════════════════════════════════
# Print Summary
# ═════════════════════════════════════════════════════════════════════════
def print_summary(summary: Dict):
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total agent outputs:  {summary['total']}")
    print(f"Matched to GT:        {summary['matched']}")
    print(f"Unmatched:            {summary['unmatched']}")

    print(f"\n{'─' * 80}")
    print("Task 1 — Theme Accuracy:")
    for model, acc in summary["theme_accuracy"].items():
        a = f"{acc['accuracy']:.1%}" if acc["accuracy"] is not None else "N/A"
        code_n = summary["by_model"][model]["has_code"]
        print(f"  {model:12s}: {acc['correct']}/{acc['total']} ({a})  "
              f"[{code_n} with replication code]")

    if summary.get("llm_scores"):
        print(f"\n{'─' * 80}")
        print("Task 2+3 — OpenAI LLM-as-Judge (1-5 scale):")
        for model, fields in summary["llm_scores"].items():
            print(f"  {model}:")
            for field, s in fields.items():
                d = s.get("dist", {})
                dist_str = " ".join(f"{k}★:{v}" for k, v in sorted(d.items()))
                print(f"    {field:12s}: mean={s['mean']:.2f}  "
                      f"median={s['median']:.1f}  std={s['std']:.2f}  "
                      f"[{s['min']}-{s['max']}]  n={s['n']}  ({dist_str})")

    if summary.get("similarity"):
        print(f"\n{'─' * 80}")
        print("Cosine Similarity:")
        for model, fields in summary["similarity"].items():
            print(f"  {model}:")
            for field, s in fields.items():
                print(f"    {field:12s}: mean={s['mean']:.4f}  "
                      f"median={s['median']:.4f}  "
                      f"[{s['min']:.4f}-{s['max']:.4f}]  n={s['n']}")

    print(f"{'=' * 80}")


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Pull agent outputs from GCS + OSP code from GitHub, evaluate all tasks."
    )
    parser.add_argument("--pull", action="store_true", help="Pull agent outputs from GCS")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--list-papers", action="store_true")
    parser.add_argument("--papers", nargs="+", default=None)
    parser.add_argument("--exclude-papers", nargs="+", default=None,
                        help="Paper names to exclude from pull and evaluation")
    parser.add_argument("--models", nargs="+", default=None, choices=MODEL_CHOICES)
    parser.add_argument("--agent-outputs", default=AGENT_OUTPUTS_FILE)
    parser.add_argument("--gt-task1", default=GROUND_TRUTH_TASK1)
    parser.add_argument("--gt-task2", default=GROUND_TRUTH_TASK2)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--osp-local-dir", default=None,
                        help="Path to local CrossSection repo (skip git clone)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_papers:
        client = get_gcs_client()
        papers = list_papers_in_gcs(client)
        print(f"\nPapers in gs://{OUTPUT_BUCKET}/ ({len(papers)}):")
        for i, p in enumerate(papers):
            print(f"  [{i:3d}] {p}")
        return

    # ── Build exclusion set ────────────────────────────────────────
    exclude_set: Set[str] = set()
    if args.exclude_papers:
        exclude_set = {normalize_paper_name(p) for p in args.exclude_papers}
        logger.info(f"Will exclude {len(exclude_set)} papers: {sorted(exclude_set)}")

    # ── Pull ─────────────────────────────────────────────────────────
    agent_outputs = None
    if args.pull:
        client = get_gcs_client()
        agent_outputs = pull_all_outputs(client, args.papers, args.models)
        if exclude_set:
            before = len(agent_outputs)
            agent_outputs = [
                a for a in agent_outputs
                if normalize_paper_name(a["paper_name"]) not in exclude_set
            ]
            logger.info(f"Excluded {before - len(agent_outputs)} outputs "
                        f"({len(exclude_set)} papers) during pull")
        out_path = out_dir / args.agent_outputs
        with open(out_path, "w") as f:
            json.dump(agent_outputs, f, indent=2, default=str)
        logger.info(f"Saved {len(agent_outputs)} agent outputs to {out_path}")

    # ── Eval ─────────────────────────────────────────────────────────
    if args.eval:
        if agent_outputs is None:
            ao_path = out_dir / args.agent_outputs
            if not ao_path.exists():
                logger.error(f"Not found: {ao_path}. Run --pull first.")
                sys.exit(1)
            with open(ao_path) as f:
                agent_outputs = json.load(f)
            logger.info(f"Loaded {len(agent_outputs)} agent outputs")
            if exclude_set:
                before = len(agent_outputs)
                agent_outputs = [
                    a for a in agent_outputs
                    if normalize_paper_name(a["paper_name"]) not in exclude_set
                ]
                logger.info(f"Excluded {before - len(agent_outputs)} outputs "
                            f"({len(exclude_set)} papers) during eval")

        gt1, gt2, name_map = load_ground_truth(args.gt_task1, args.gt_task2)

        # Fetch OSP reference code
        osp_code = fetch_osp_code(args.osp_local_dir)
        osp_lookup = build_osp_lookup(gt1, osp_code) if osp_code else {}

        use_sim = not args.no_similarity
        use_llm = not args.no_llm_judge

        eval_results, summary = run_eval(
            agent_outputs, gt1, gt2, name_map, osp_lookup,
            use_sim=use_sim, use_llm=use_llm, llm_model=args.llm_model,
        )

        # Save
        with open(out_dir / EVAL_RESULTS_FILE, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        with open(out_dir / EVAL_SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved results to {EVAL_RESULTS_FILE} and {EVAL_SUMMARY_FILE}")

        print_summary(summary)

    if not args.pull and not args.eval and not args.list_papers:
        parser.print_help()


if __name__ == "__main__":
    main()