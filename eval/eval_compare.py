#!/usr/bin/env python3
"""
eval_compare.py -- Compare framework (multi-agent) vs one-shot outputs
                   for frontier LLM providers against ground truth.

Pulls outputs from GCS for both pipelines and evaluates side-by-side:
    Framework:  gs://<bucket>/<paper>/<provider>/<ts>.json
    One-shot:   gs://<bucket>/<paper>/<provider>/oneshot/<ts>_oneshot.json

Each provider x pipeline combination is treated as a distinct "model_choice":
    openai_framework, openai_oneshot,
    anthropic_framework, anthropic_oneshot,
    gemini_framework, gemini_oneshot

Three evaluation tasks (same as eval_all.py):
  Task 1: theme_pick -- exact match vs ground truth jkp_theme
  Task 2: factor_extract -- OpenAI LLM-as-Judge 1-5 scoring of purpose & logic
  Task 3: replication code -- OpenAI LLM-as-Judge 1-5 scoring vs clean_code

Head-to-head: when scores are tied, an LLM tiebreaker picks the winner.
No ties are possible.

Usage:
    python eval_compare.py --pull --eval
    python eval_compare.py --eval --agent-outputs compare_outputs.json
    python eval_compare.py --eval --no-llm-judge
    python eval_compare.py --pull --eval --providers openai anthropic
    python eval_compare.py --list-papers
"""

import argparse
import json
import logging
import os
import re
import sys
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_compare")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GCS_PROJECT_ID = os.getenv("GCS_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
GCS_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", os.getenv("GCS_CREDENTIALS_PATH", "")
)
OUTPUT_BUCKET = os.getenv("GCS_OUTPUT_BUCKET", "investment-research-output")

PROVIDERS = ["openai", "anthropic", "gemini"]
PIPELINES = ["framework", "oneshot"]

GROUND_TRUTH_FILE = os.getenv("GROUND_TRUTH_FILE", "osp_eval_policy_v003.json")

AGENT_OUTPUTS_FILE = "compare_outputs.json"
EVAL_RESULTS_FILE = "compare_eval_results.json"
EVAL_SUMMARY_FILE = "compare_eval_summary.json"


# ---------------------------------------------------------------------------
# GCS Client
# ---------------------------------------------------------------------------
def get_gcs_client() -> storage.Client:
    creds = GCS_CREDENTIALS_PATH
    if creds and Path(creds).exists():
        return storage.Client.from_service_account_json(creds)
    if GCS_PROJECT_ID:
        return storage.Client(project=GCS_PROJECT_ID)
    return storage.Client()


def normalize_paper_name(name: str) -> str:
    return name.replace(".pdf", "")


# ---------------------------------------------------------------------------
# GCS Discovery
# ---------------------------------------------------------------------------
def list_papers_in_gcs(client: storage.Client) -> List[str]:
    bucket = client.bucket(OUTPUT_BUCKET)
    paper_names = set()
    for blob in bucket.list_blobs(prefix="ref_"):
        parts = blob.name.split("/")
        if len(parts) >= 2:
            paper_names.add(parts[0])
    return sorted(paper_names)


def find_latest_framework_json(client, paper_name, provider):
    bucket = client.bucket(OUTPUT_BUCKET)
    prefix = f"{paper_name}/{provider}/"
    blobs = []
    for b in bucket.list_blobs(prefix=prefix):
        name = b.name
        if f"/{provider}/oneshot/" in name:
            continue
        if not name.endswith(".json"):
            continue
        if name.endswith("_replication.json"):
            continue
        if "_oneshot" in name:
            continue
        blobs.append(b)
    if not blobs:
        return None
    blobs.sort(key=lambda b: b.time_created or b.name, reverse=True)
    return blobs[0]


def find_latest_framework_code(client, paper_name, provider):
    bucket = client.bucket(OUTPUT_BUCKET)
    prefix = f"{paper_name}/{provider}/"
    blobs = []
    for b in bucket.list_blobs(prefix=prefix):
        if f"/{provider}/oneshot/" in b.name:
            continue
        if b.name.endswith("_code.py") and "_oneshot" not in b.name:
            blobs.append(b)
    if not blobs:
        return None
    blobs.sort(key=lambda b: b.time_created or b.name, reverse=True)
    return blobs[0]


def find_latest_oneshot_json(client, paper_name, provider):
    bucket = client.bucket(OUTPUT_BUCKET)
    prefix = f"{paper_name}/{provider}/oneshot/"
    blobs = [
        b for b in bucket.list_blobs(prefix=prefix)
        if b.name.endswith("_oneshot.json")
    ]
    if not blobs:
        return None
    blobs.sort(key=lambda b: b.time_created or b.name, reverse=True)
    return blobs[0]


def find_latest_oneshot_code(client, paper_name, provider):
    bucket = client.bucket(OUTPUT_BUCKET)
    prefix = f"{paper_name}/{provider}/oneshot/"
    blobs = [
        b for b in bucket.list_blobs(prefix=prefix)
        if b.name.endswith("_oneshot_code.py")
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


# ---------------------------------------------------------------------------
# Pull All Outputs (Framework + Oneshot)
# ---------------------------------------------------------------------------
def pull_all_outputs(client, paper_filter=None, provider_filter=None):
    providers = provider_filter or PROVIDERS
    papers = paper_filter or list_papers_in_gcs(client)
    logger.info(
        f"Pulling outputs for {len(papers)} papers x "
        f"{len(providers)} providers x 2 pipelines"
    )

    all_outputs = []
    stats = defaultdict(int)

    for paper_name in papers:
        for provider in providers:
            for pipeline in PIPELINES:
                try:
                    if pipeline == "framework":
                        blob = find_latest_framework_json(client, paper_name, provider)
                        code_blob = find_latest_framework_code(client, paper_name, provider)
                    else:
                        blob = find_latest_oneshot_json(client, paper_name, provider)
                        code_blob = find_latest_oneshot_code(client, paper_name, provider)

                    if blob is None:
                        stats[f"{provider}_{pipeline}_missing"] += 1
                        continue

                    raw = json.loads(blob.download_as_text())
                    raw = normalise_upstream(raw)

                    model_choice = f"{provider}_{pipeline}"

                    entry = {
                        "paper_name": paper_name,
                        "provider": provider,
                        "pipeline": pipeline,
                        "model_choice": model_choice,
                        "gcs_blob": blob.name,
                        "theme_pick": raw.get("theme_pick", {}),
                        "factor_extract": raw.get("factor_extract", {}),
                    }

                    if code_blob is not None:
                        entry["agent_code"] = code_blob.download_as_text()
                        entry["agent_code_blob"] = code_blob.name
                        stats[f"{provider}_{pipeline}_code"] += 1
                    elif raw.get("generated_code"):
                        entry["agent_code"] = raw["generated_code"]
                        entry["agent_code_blob"] = f"{blob.name}#generated_code"
                        stats[f"{provider}_{pipeline}_code"] += 1
                    else:
                        entry["agent_code"] = None
                        entry["agent_code_blob"] = None

                    all_outputs.append(entry)
                    stats[f"{provider}_{pipeline}_found"] += 1

                    theme = entry["theme_pick"].get("theme", "?")
                    code_mark = "+code" if entry["agent_code"] else "no-code"
                    logger.info(
                        f"  + {paper_name}/{provider}/{pipeline} "
                        f"(theme={theme}, {code_mark})"
                    )

                except Exception as e:
                    logger.error(f"  x {paper_name}/{provider}/{pipeline}: {e}")
                    stats[f"{provider}_{pipeline}_errors"] += 1

    logger.info(f"\nPull stats:")
    for k, v in sorted(stats.items()):
        logger.info(f"  {k}: {v}")
    logger.info(f"Total outputs: {len(all_outputs)}")
    return all_outputs


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------
def load_ground_truth(gt_path):
    with open(gt_path) as f:
        gt = json.load(f)
    name_map = {}
    for key in gt.keys():
        name_map[normalize_paper_name(key)] = key
    logger.info(f"Ground truth loaded: {len(gt)} papers from {gt_path}")
    return gt, name_map


def build_code_lookup(gt):
    lookup = {}
    for gt_key, entry in gt.items():
        code = entry.get("clean_code", "")
        if code:
            paper = normalize_paper_name(gt_key)
            lookup[paper] = code
    logger.info(f"Reference code lookup: {len(lookup)}/{len(gt)} papers have clean_code")
    return lookup


# ---------------------------------------------------------------------------
# Cosine Similarity Scorer
# ---------------------------------------------------------------------------
class SimilarityScorer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)

    def score(self, text_a, text_b):
        if not text_a or not text_b:
            return 0.0
        self._load()
        import numpy as np
        emb = self._model.encode([text_a, text_b], normalize_embeddings=True)
        return round(float(np.dot(emb[0], emb[1])), 4)


_scorer = None
def get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = SimilarityScorer()
    return _scorer


# ---------------------------------------------------------------------------
# OpenAI LLM-as-Judge (identical prompts to eval_all.py)
# ---------------------------------------------------------------------------
PURPOSE_PROMPT = """\
You are an expert evaluator for quantitative finance factor research, \
specializing in academic paper replication and factor construction.

Score the agent's factor PURPOSE description from 1 to 5 vs ground truth.

Focus on:
- Whether the agent captures the ECONOMIC MECHANISM or hypothesis behind the factor
- Whether the agent identifies the correct ASSET PRICING CONTEXT
- Whether the agent NAMES THE SPECIFIC VARIABLES or signals used
- Whether the agent explains the DIRECTION of the predicted relationship
- Nuance: theoretical motivation from the paper > generic description

Rubric:
1 = Completely wrong -- different factor or entirely wrong mechanism
2 = Partially relevant -- right area but misses core mechanism
3 = Acceptable -- general idea correct but lacks specificity
4 = Good -- accurate purpose, mechanism, variables, direction. Minor omissions.
5 = Excellent -- fully captures purpose, mechanism, variables, direction, novelty

Respond ONLY with JSON: {"score": <int 1-5>, "reasoning": "<brief>"}"""

LOGIC_PROMPT = """\
You are an expert evaluator for quantitative finance factor research, \
specializing in academic paper replication and factor construction.

Score the agent's COMPUTATION LOGIC description from 1 to 5 vs ground truth.

Key evaluation criteria (in order of importance):
1. VARIABLE CONSTRUCTION -- correct input variables, transformations, signal formula
2. SORTING/PORTFOLIO METHODOLOGY -- correct sorting procedure, breakpoints
3. TIMING AND LAGS -- correct measurement timing, rebalance frequency, holding period
4. SAMPLE FILTERS AND ADJUSTMENTS -- critical filters mentioned
5. COMPLETENESS -- covers signal, universe, timing, AND portfolio construction
6. EVIDENCE GROUNDING -- distinguishes paper-stated vs assumed conventions

Rubric:
1 = Completely wrong -- different computation entirely
2 = Major gaps -- some correct elements but missing critical steps
3 = Acceptable -- general approach correct but vague on specifics. <3 of 4 areas.
4 = Good -- main construction steps correct, >=3 of 4 areas. Minor omissions.
5 = Excellent -- comprehensive with correct variables, sorting, timing, adjustments

Respond ONLY with JSON: {"score": <int 1-5>, "reasoning": "<brief>"}"""

CODE_PROMPT = """\
You are an expert evaluator for quantitative finance factor replication code.

Compare agent-generated Python replication code against the reference implementation.

IMPORTANT: Do NOT penalize for data loading differences, file paths, column names, \
imports, or variable naming. DO focus on core factor calculation logic.

Score from 1 to 5:
1. SIGNAL COMPUTATION -- correct formula and transformation (most critical)
2. SORTING/PORTFOLIO LOGIC -- correct breakpoints, quantiles, long-short
3. TIMING AND LAGS -- correct temporal alignment
4. CRITICAL FILTERS -- essential sample restrictions
5. ASSUMPTIONS AND ROBUSTNESS -- documented assumptions, sanity checks
6. OUTPUT STRUCTURE -- produces compatible signal/return output

Rubric:
1 = Completely wrong signal/factor
2 = Right idea but critical calculation errors
3 = Core signal approximately correct but methodology gaps
4 = Correct signal and methodology, minor differences
5 = Faithful implementation with documented assumptions and validation

Respond ONLY with JSON: {"score": <int 1-5>, "reasoning": "<brief>"}"""


class OpenAIJudge:
    def __init__(self, model="gpt-4o-mini"):
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

    def _call(self, system, user):
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
            logger.error(f"JSON parse error from judge: {content[:200]}")
            return {"score": None, "reasoning": "JSON parse error"}
        except Exception as e:
            logger.error(f"Judge API error: {e}")
            return {"score": None, "reasoning": str(e)}

    def score_purpose(self, agent, gt):
        if not agent or not gt:
            return {"score": None, "reasoning": "Missing text"}
        return self._call(PURPOSE_PROMPT,
            f"**Ground Truth:**\n{gt}\n\n**Agent Output:**\n{agent}")

    def score_logic(self, agent, gt):
        if not agent or not gt:
            return {"score": None, "reasoning": "Missing text"}
        return self._call(LOGIC_PROMPT,
            f"**Ground Truth:**\n{gt}\n\n**Agent Output:**\n{agent}")

    def score_code(self, agent_code, ref_code, paper_name=""):
        if not agent_code or not ref_code:
            return {"score": None, "reasoning": "Missing code"}
        a = agent_code[:6000]
        r = ref_code[:6000]
        return self._call(CODE_PROMPT,
            f"**Paper:** {paper_name}\n\n"
            f"**Reference Implementation:**\n```python\n{r}\n```\n\n"
            f"**Agent-Generated Code:**\n```python\n{a}\n```")


_judge = None
def get_judge(model="gpt-4o-mini"):
    global _judge
    if _judge is None:
        _judge = OpenAIJudge(model=model)
    return _judge


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------
def eval_theme(agent_theme, gt_entry):
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


def eval_factor(agent_factor, gt_entry, use_sim, use_llm, llm_model):
    ap = agent_factor.get("purpose", "") or ""
    al = agent_factor.get("computation_logic", "") or ""

    flp = gt_entry.get("factor_logic_in_paper", {})
    gp = flp.get("purpose", "") or ""
    gl = flp.get("computation_logic", "") or ""

    result = {
        "agent_purpose": ap, "gt_purpose": gp,
        "agent_logic": al, "gt_logic": gl,
        "purpose_len_agent": len(ap), "logic_len_agent": len(al),
    }

    al_lower = al.lower()
    topic_coverage = {
        "signal": any(w in al_lower for w in [
            "ratio", "signal", "variable", "formula", "computed",
            "numerator", "denominator", "transform", "standardiz",
            "winsoriz", "log ", "difference", "change in",
        ]),
        "universe": any(w in al_lower for w in [
            "universe", "filter", "exclud", "financials", "utilities",
            "common stock", "nyse", "amex", "nasdaq", "exchange",
            "market cap", "price filter", "penny",
        ]),
        "timing": any(w in al_lower for w in [
            "rebalanc", "holding period", "formation", "lag",
            "monthly", "quarterly", "annual", "june",
            "look-ahead", "fiscal year",
        ]),
        "portfolio": any(w in al_lower for w in [
            "sort", "decile", "quintile", "tercile", "quantile",
            "breakpoint", "long-short", "long short", "top", "bottom",
            "equal-weight", "value-weight", "portfolio",
        ]),
    }
    result["topic_coverage"] = topic_coverage
    result["topic_coverage_count"] = sum(topic_coverage.values())

    provenance_signals = {
        "evidence_tags": len(re.findall(r'\[EVIDENCE:', al, re.IGNORECASE)),
        "convention_tags": len(re.findall(r'\[CONVENTION:', al, re.IGNORECASE)),
        "missing_tags": len(re.findall(r'\[MISSING:', al, re.IGNORECASE)),
        "convention_notes": len(re.findall(r'standard convention:', al_lower)),
        "not_specified_notes": len(re.findall(r'not specified', al_lower)),
    }
    result["provenance_signals"] = provenance_signals
    result["has_provenance"] = any(v > 0 for v in provenance_signals.values())

    paragraphs = [p.strip() for p in al.split('\n\n') if p.strip() and len(p.strip()) > 20]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in al.split('\n') if p.strip() and len(p.strip()) > 40]
    result["paragraph_count"] = len(paragraphs)
    result["logic_word_count"] = len(re.findall(r'[A-Za-z]+', al))
    result["purpose_word_count"] = len(re.findall(r'[A-Za-z]+', ap))

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


def eval_code(agent_code, ref_code, paper_name, use_sim, use_llm, llm_model):
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

    code_lower = agent_code.lower()
    assumption_lines = [
        line for line in agent_code.splitlines()
        if re.search(r'#\s*(assumption|assume|convention|paper does not|not specified)',
                      line, re.IGNORECASE)
    ]
    result["assumption_count"] = len(assumption_lines)
    result["has_assumptions_block"] = bool(re.search(
        r'#\s*ASSUMPTIONS?\s*:', agent_code, re.IGNORECASE
    ))
    result["assert_count"] = len(re.findall(r'\bassert\b', agent_code))
    result["has_usecols"] = "usecols" in code_lower
    result["has_shift_lag"] = ".shift(" in code_lower
    result["has_pivot"] = "pivot" in code_lower

    if use_sim:
        result["code_cosine_sim"] = get_scorer().score(agent_code, ref_code)

    if use_llm:
        j = get_judge(llm_model)
        cr = j.score_code(agent_code, ref_code, paper_name)
        result["code_llm_score"] = cr.get("score")
        result["code_llm_reasoning"] = cr.get("reasoning", "")

    return result


# ---------------------------------------------------------------------------
# Main Eval Loop
# ---------------------------------------------------------------------------
def run_eval(agent_outputs, gt, name_map, code_lookup,
             use_sim=True, use_llm=True, llm_model="gpt-4o-mini"):

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
            "paper_name": paper,
            "model_choice": model,
            "provider": agent.get("provider", ""),
            "pipeline": agent.get("pipeline", ""),
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

        gt_entry = gt.get(gt_key, {})

        t1 = eval_theme(agent.get("theme_pick", {}), gt_entry)
        entry["task1"] = t1
        if t1["theme_match"]:
            summary["by_model"][model]["theme_correct"] += 1

        t2 = eval_factor(
            agent.get("factor_extract", {}), gt_entry,
            use_sim=use_sim, use_llm=use_llm, llm_model=llm_model,
        )
        entry["task2"] = t2
        if use_llm:
            ps = t2.get("purpose_llm_score", "?")
            ls = t2.get("logic_llm_score", "?")
            logger.info(f"    Task2 LLM: purpose={ps}/5, logic={ls}/5")

        agent_code = agent.get("agent_code")
        ref_code = code_lookup.get(paper)
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

    _aggregate_summary(results, summary, use_sim, use_llm)
    summary["head_to_head"] = _build_head_to_head(results, use_llm)

    return results, summary


# ---------------------------------------------------------------------------
# Aggregate Summary
# ---------------------------------------------------------------------------
def _aggregate_summary(results, summary, use_sim, use_llm):
    import numpy as np

    for model, s in summary["by_model"].items():
        m = s["matched"]
        c = s["theme_correct"]
        summary["theme_accuracy"][model] = {
            "correct": c, "total": m,
            "accuracy": round(c / m, 4) if m > 0 else None,
        }

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

    structural = defaultdict(lambda: {
        "topic_coverage": [], "provenance_pct": 0, "provenance_total": 0,
        "logic_word_count": [], "purpose_word_count": [],
        "paragraph_count": [],
        "assert_count": [], "assumption_count": [],
        "has_assumptions_block": 0, "code_n": 0,
    })

    for e in results:
        model = e["model_choice"]
        t2 = e.get("task2") or {}
        t3 = e.get("task3_code") or {}

        if t2.get("topic_coverage_count") is not None:
            structural[model]["topic_coverage"].append(t2["topic_coverage_count"])
        if t2.get("has_provenance") is not None:
            structural[model]["provenance_total"] += 1
            if t2["has_provenance"]:
                structural[model]["provenance_pct"] += 1
        if t2.get("logic_word_count") is not None:
            structural[model]["logic_word_count"].append(t2["logic_word_count"])
        if t2.get("purpose_word_count") is not None:
            structural[model]["purpose_word_count"].append(t2["purpose_word_count"])
        if t2.get("paragraph_count") is not None:
            structural[model]["paragraph_count"].append(t2["paragraph_count"])

        if t3 and t3.get("has_agent_code"):
            structural[model]["code_n"] += 1
            if t3.get("assert_count") is not None:
                structural[model]["assert_count"].append(t3["assert_count"])
            if t3.get("assumption_count") is not None:
                structural[model]["assumption_count"].append(t3["assumption_count"])
            if t3.get("has_assumptions_block"):
                structural[model]["has_assumptions_block"] += 1

    summary["structural_quality"] = {}
    for model, s in structural.items():
        import numpy as np
        entry = {}
        for field in ("topic_coverage", "logic_word_count", "purpose_word_count",
                       "paragraph_count", "assert_count", "assumption_count"):
            vals = s[field]
            if vals:
                entry[field] = {
                    "mean": round(float(np.mean(vals)), 2),
                    "median": round(float(np.median(vals)), 1),
                }
        if s["provenance_total"] > 0:
            entry["provenance_pct"] = round(
                s["provenance_pct"] / s["provenance_total"] * 100, 1
            )
        if s["code_n"] > 0:
            entry["assumptions_block_pct"] = round(
                s["has_assumptions_block"] / s["code_n"] * 100, 1
            )
        summary["structural_quality"][model] = entry


# ---------------------------------------------------------------------------
# LLM Tiebreaker -- when scores are equal, ask the judge to pick a winner
# ---------------------------------------------------------------------------
TIEBREAK_PROMPT = """\
You are an expert evaluator for quantitative finance factor research.

Two systems analyzed the same academic paper. Their outputs are tied on numeric scores.
You MUST pick a winner -- no ties allowed.

Compare the two outputs holistically across theme classification, factor purpose, \
and computation logic. Pick whichever output is more accurate, specific, and faithful \
to what a finance researcher would extract from this paper.

Respond ONLY with JSON: {"winner": "A" or "B", "reasoning": "<one sentence>"}"""


def _llm_tiebreak(fw_result, os_result, paper, provider):
    """Ask the LLM judge to pick a winner when scores are tied."""
    try:
        j = get_judge()

        fw_t2 = fw_result.get("task2") or {}
        os_t2 = os_result.get("task2") or {}

        output_a = (
            f"Theme: {(fw_result.get('task1') or {}).get('agent_theme', '?')}\n"
            f"Purpose: {fw_t2.get('agent_purpose', '')[:500]}\n"
            f"Logic: {fw_t2.get('agent_logic', '')[:800]}"
        )
        output_b = (
            f"Theme: {(os_result.get('task1') or {}).get('agent_theme', '?')}\n"
            f"Purpose: {os_t2.get('agent_purpose', '')[:500]}\n"
            f"Logic: {os_t2.get('agent_logic', '')[:800]}"
        )

        user_msg = (
            f"Paper: {paper}\n\n"
            f"**Output A (multi-agent framework):**\n{output_a}\n\n"
            f"**Output B (one-shot):**\n{output_b}"
        )

        resp = j._call(TIEBREAK_PROMPT, user_msg)
        pick = resp.get("winner", "A").strip().upper()
        logger.info(
            f"  [tiebreak] {paper}/{provider}: LLM picked "
            f"{'framework' if pick == 'A' else 'oneshot'} "
            f"-- {resp.get('reasoning', '')[:80]}"
        )
        return "framework" if pick == "A" else "oneshot"
    except Exception as e:
        logger.warning(
            f"  [tiebreak] LLM call failed for {paper}/{provider}: {e}, "
            f"defaulting to framework"
        )
        return "framework"


# ---------------------------------------------------------------------------
# Head-to-Head: Framework vs Oneshot per provider per paper
# ---------------------------------------------------------------------------
def _build_head_to_head(results, use_llm):
    """For each provider, compare framework vs oneshot paper-by-paper."""

    lookup = {}
    for e in results:
        if not e.get("has_ground_truth"):
            continue
        key = (e["paper_name"], e.get("provider", ""), e.get("pipeline", ""))
        lookup[key] = e

    h2h = {}
    for provider in PROVIDERS:
        wins = {"framework": 0, "oneshot": 0}
        comparisons = []

        fw_papers = {k[0] for k in lookup if k[1] == provider and k[2] == "framework"}
        os_papers = {k[0] for k in lookup if k[1] == provider and k[2] == "oneshot"}
        both = sorted(fw_papers & os_papers)

        for paper in both:
            fw = lookup.get((paper, provider, "framework"))
            os_ = lookup.get((paper, provider, "oneshot"))
            if not fw or not os_:
                continue

            comp = {"paper": paper}

            fw_theme = (fw.get("task1") or {}).get("theme_match", False)
            os_theme = (os_.get("task1") or {}).get("theme_match", False)
            comp["theme_fw"] = fw_theme
            comp["theme_os"] = os_theme

            if use_llm:
                fw_t2 = fw.get("task2") or {}
                os_t2 = os_.get("task2") or {}
                fw_t3 = fw.get("task3_code") or {}
                os_t3 = os_.get("task3_code") or {}

                for field, fw_val, os_val in [
                    ("purpose", fw_t2.get("purpose_llm_score"),
                     os_t2.get("purpose_llm_score")),
                    ("logic", fw_t2.get("logic_llm_score"),
                     os_t2.get("logic_llm_score")),
                    ("code", fw_t3.get("code_llm_score"),
                     os_t3.get("code_llm_score")),
                ]:
                    comp[f"{field}_fw"] = fw_val
                    comp[f"{field}_os"] = os_val

                fw_scores = [v for v in [
                    fw_t2.get("purpose_llm_score"),
                    fw_t2.get("logic_llm_score"),
                    fw_t3.get("code_llm_score"),
                ] if isinstance(v, (int, float))]
                os_scores = [v for v in [
                    os_t2.get("purpose_llm_score"),
                    os_t2.get("logic_llm_score"),
                    os_t3.get("code_llm_score"),
                ] if isinstance(v, (int, float))]

                fw_total = sum(fw_scores) if fw_scores else 0
                os_total = sum(os_scores) if os_scores else 0
                comp["total_fw"] = fw_total
                comp["total_os"] = os_total

                if fw_total != os_total:
                    comp["winner"] = "framework" if fw_total > os_total else "oneshot"
                else:
                    # Tied on scores -- ask the LLM to pick
                    comp["winner"] = _llm_tiebreak(fw, os_, paper, provider)
                    comp["tiebreaker"] = "llm_preference"

                wins[comp["winner"]] += 1

            comparisons.append(comp)

        h2h[provider] = {
            "papers_compared": len(both),
            "wins": wins,
            "comparisons": comparisons,
        }

    return h2h


# ---------------------------------------------------------------------------
# Print Summary
# ---------------------------------------------------------------------------
def print_summary(summary):
    print(f"\n{'=' * 80}")
    print("FRAMEWORK vs ONE-SHOT COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total outputs:   {summary['total']}")
    print(f"Matched to GT:   {summary['matched']}")
    print(f"Unmatched:       {summary['unmatched']}")

    print(f"\n{'-' * 80}")
    print("Task 1 -- Theme Accuracy:")
    for provider in PROVIDERS:
        fw_key = f"{provider}_framework"
        os_key = f"{provider}_oneshot"
        fw = summary["theme_accuracy"].get(fw_key, {})
        os_ = summary["theme_accuracy"].get(os_key, {})

        fw_acc = f"{fw['accuracy']:.1%}" if fw.get("accuracy") is not None else "N/A"
        os_acc = f"{os_['accuracy']:.1%}" if os_.get("accuracy") is not None else "N/A"
        fw_n = f"{fw.get('correct', 0)}/{fw.get('total', 0)}"
        os_n = f"{os_.get('correct', 0)}/{os_.get('total', 0)}"

        print(f"  {provider:12s}:  framework={fw_n} ({fw_acc})  "
              f"oneshot={os_n} ({os_acc})")

    if summary.get("llm_scores"):
        print(f"\n{'-' * 80}")
        print("Task 2+3 -- LLM-as-Judge Scores (1-5 scale):")
        for provider in PROVIDERS:
            fw_key = f"{provider}_framework"
            os_key = f"{provider}_oneshot"
            fw_scores = summary["llm_scores"].get(fw_key, {})
            os_scores = summary["llm_scores"].get(os_key, {})

            if not fw_scores and not os_scores:
                continue

            print(f"\n  {provider}:")
            all_fields = sorted(set(list(fw_scores.keys()) + list(os_scores.keys())))
            for field in all_fields:
                fw_s = fw_scores.get(field, {})
                os_s = os_scores.get(field, {})
                fw_str = (
                    f"mean={fw_s['mean']:.2f} med={fw_s['median']:.1f} n={fw_s['n']}"
                    if fw_s else "N/A"
                )
                os_str = (
                    f"mean={os_s['mean']:.2f} med={os_s['median']:.1f} n={os_s['n']}"
                    if os_s else "N/A"
                )
                delta_str = ""
                if fw_s and os_s:
                    delta = os_s["mean"] - fw_s["mean"]
                    arrow = ">" if delta > 0 else "<" if delta < 0 else "="
                    delta_str = f"  {arrow} {abs(delta):+.2f}"
                print(f"    {field:10s}:  framework: {fw_str}  |  "
                      f"oneshot: {os_str}{delta_str}")

    h2h = summary.get("head_to_head", {})
    if h2h:
        print(f"\n{'-' * 80}")
        print("Head-to-Head (Framework vs One-Shot, NO TIES):")
        for provider in PROVIDERS:
            ph = h2h.get(provider, {})
            if not ph:
                continue
            wins = ph.get("wins", {})
            n = ph.get("papers_compared", 0)
            print(
                f"  {provider:12s}:  {n} papers compared  |  "
                f"Framework wins: {wins.get('framework', 0)}  "
                f"Oneshot wins: {wins.get('oneshot', 0)}"
            )

    if summary.get("similarity"):
        print(f"\n{'-' * 80}")
        print("Cosine Similarity:")
        for provider in PROVIDERS:
            fw_key = f"{provider}_framework"
            os_key = f"{provider}_oneshot"
            fw_sim = summary["similarity"].get(fw_key, {})
            os_sim = summary["similarity"].get(os_key, {})
            if not fw_sim and not os_sim:
                continue
            print(f"\n  {provider}:")
            all_fields = sorted(set(list(fw_sim.keys()) + list(os_sim.keys())))
            for field in all_fields:
                fw_s = fw_sim.get(field, {})
                os_s = os_sim.get(field, {})
                fw_str = f"mean={fw_s['mean']:.4f}" if fw_s else "N/A"
                os_str = f"mean={os_s['mean']:.4f}" if os_s else "N/A"
                print(f"    {field:10s}:  fw={fw_str}  os={os_str}")

    print(f"\n{'=' * 80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare framework vs one-shot outputs for frontier providers."
    )
    parser.add_argument("--pull", action="store_true", help="Pull outputs from GCS")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--list-papers", action="store_true")
    parser.add_argument("--papers", nargs="+", default=None)
    parser.add_argument("--exclude-papers", nargs="+", default=None)
    parser.add_argument("--providers", nargs="+", default=None,
                        choices=PROVIDERS,
                        help=f"Providers to compare (default: all three)")
    parser.add_argument("--agent-outputs", default=AGENT_OUTPUTS_FILE)
    parser.add_argument("--gt", default=GROUND_TRUTH_FILE,
                        help="Path to ground truth policy JSON")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--no-similarity", action="store_true")
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
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

    exclude_set = set()
    if args.exclude_papers:
        exclude_set = {normalize_paper_name(p) for p in args.exclude_papers}
        logger.info(f"Will exclude {len(exclude_set)} papers")

    agent_outputs = None
    if args.pull:
        client = get_gcs_client()
        agent_outputs = pull_all_outputs(client, args.papers, args.providers)
        if exclude_set:
            before = len(agent_outputs)
            agent_outputs = [
                a for a in agent_outputs
                if normalize_paper_name(a["paper_name"]) not in exclude_set
            ]
            logger.info(f"Excluded {before - len(agent_outputs)} outputs")
        out_path = out_dir / args.agent_outputs
        with open(out_path, "w") as f:
            json.dump(agent_outputs, f, indent=2, default=str)
        logger.info(f"Saved {len(agent_outputs)} outputs to {out_path}")

    if args.eval:
        if agent_outputs is None:
            ao_path = out_dir / args.agent_outputs
            if not ao_path.exists():
                logger.error(f"Not found: {ao_path}. Run --pull first.")
                sys.exit(1)
            with open(ao_path) as f:
                agent_outputs = json.load(f)
            logger.info(f"Loaded {len(agent_outputs)} outputs")
            if exclude_set:
                before = len(agent_outputs)
                agent_outputs = [
                    a for a in agent_outputs
                    if normalize_paper_name(a["paper_name"]) not in exclude_set
                ]
                logger.info(f"Excluded {before - len(agent_outputs)} outputs")

        gt, name_map = load_ground_truth(args.gt)
        code_lookup = build_code_lookup(gt)

        use_sim = not args.no_similarity
        use_llm = not args.no_llm_judge

        eval_results, summary = run_eval(
            agent_outputs, gt, name_map, code_lookup,
            use_sim=use_sim, use_llm=use_llm, llm_model=args.llm_model,
        )

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_file = f"compare_eval_results_{ts}.json"
        summary_file = f"compare_eval_summary_{ts}.json"
        with open(out_dir / results_file, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        with open(out_dir / summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        with open(out_dir / EVAL_RESULTS_FILE, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        with open(out_dir / EVAL_SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved to {results_file} and {summary_file}")

        print_summary(summary)

    if not args.pull and not args.eval and not args.list_papers:
        parser.print_help()


if __name__ == "__main__":
    main()