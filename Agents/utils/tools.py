from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import spearmanr

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from langchain_core.tools import tool

@tool
def decile_sorter(
    df: pd.DataFrame,
    signal_col: str = "signal",
    date_col: str = "date",
    asset_col: str = "asset",
    n_deciles: int = 10,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Ranks stocks into quantiles (deciles by default) based on a signal and
    assigns each (date, asset) pair a decile label.

    Parameters
    ----------
    df : DataFrame
        Must contain columns [date_col, asset_col, signal_col].
    signal_col : str
        Name of the signal column to sort on.
    date_col : str
        Column indicating rebalance date.
    asset_col : str
        Column indicating asset identifier.
    n_deciles : int
        Number of quantiles/deciles.
    ascending : bool
        If True, smallest signal -> lowest decile. If False, reverse.

    Returns
    -------
    DataFrame
        Original df with an added integer column 'decile' in [1, n_deciles].
    """

    df = df.copy()

    def _assign_deciles(group: pd.DataFrame) -> pd.DataFrame:
        # Rank to break ties
        ranked = group[signal_col].rank(method="first", ascending=ascending)
        try:
            labels = list(range(1, n_deciles + 1))
            group["decile"] = pd.qcut(
                ranked,
                q=n_deciles,
                labels=labels,
                duplicates="drop",
            ).astype(int)
        except ValueError:
            # Not enough unique observations; fall back to fewer buckets
            uniq = ranked.nunique()
            if uniq == 0:
                group["decile"] = 1
            else:
                labels = list(range(1, uniq + 1))
                group["decile"] = pd.qcut(
                    ranked,
                    q=uniq,
                    labels=labels,
                    duplicates="drop",
                ).astype(int)
        return group

    df = df.groupby(date_col, group_keys=False).apply(_assign_deciles)
    return df

@tool  
def zscore_normalizer(
    df: pd.DataFrame,
    signal_col: str = "signal",
    date_col: str = "date",
    clip: Optional[float] = None,
    out_col: str = "signal_z",
) -> pd.DataFrame:
    """
    Cross-sectional z-score standardization per rebalance date:
    transforms signal to mean 0, std 1 on each date.

    Parameters
    ----------
    df : DataFrame
    signal_col : str
        Name of the signal column.
    date_col : str
    clip : float, optional
        If provided, z-scores are winsorized to +/- clip.
    out_col : str
        Name of standardized signal column.

    Returns
    -------
    DataFrame
        df with added column out_col.
    """
    df = df.copy()

    def _z(group: pd.DataFrame) -> pd.DataFrame:
        s = group[signal_col].astype(float)
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            z = s * 0.0
        else:
            z = (s - mu) / sigma
        if clip is not None:
            z = z.clip(-clip, clip)
        group[out_col] = z
        return group

    df = df.groupby(date_col, group_keys=False).apply(_z)
    return df

@tool
def industry_neutralizer(
    df: pd.DataFrame,
    signal_col: str = "signal",
    industry_col: str = "industry",
    date_col: str = "date",
    out_col: str = "signal_ind_neutral",
) -> pd.DataFrame:
    """
    Removes industry/sector effects from signal values by running a
    cross-sectional regression of signal on industry dummies each date and
    using residuals as neutralized signal.

    signal_it = a_t + sum_k b_{k,t} * I(industry=k) + e_it

    Parameters
    ----------
    df : DataFrame
        Must contain [date_col, signal_col, industry_col].
    signal_col : str
    industry_col : str
    date_col : str
    out_col : str

    Returns
    -------
    DataFrame
        df with added column out_col containing industry-neutral residuals.
    """
    df = df.copy()

    def _neutralize(group: pd.DataFrame) -> pd.DataFrame:
        if group[industry_col].nunique() <= 1:
            # Nothing to neutralize
            group[out_col] = group[signal_col] - group[signal_col].mean()
            return group

        dummies = pd.get_dummies(group[industry_col], drop_first=True)
        X = sm.add_constant(dummies)
        y = group[signal_col].astype(float).values
        try:
            model = sm.OLS(y, X).fit()
            residuals = y - model.fittedvalues
        except Exception:
            residuals = y - y.mean()
        group[out_col] = residuals
        return group

    df = df.groupby(date_col, group_keys=False).apply(_neutralize)
    return df


def cross_sectional_backtester(
    df: pd.DataFrame,
    ret_col: str = "ret",
    decile_col: str = "decile",
    date_col: str = "date",
    long_deciles: Sequence[int] = (10,),
    short_deciles: Sequence[int] = (1,),
) -> pd.DataFrame:
    """
    Computes equal-weighted portfolio returns for each decile and a
    long-short portfolio across rebalances.

    Parameters
    ----------
    df : DataFrame
        Must contain [date_col, decile_col, ret_col].
    ret_col : str
        Forward return column.
    decile_col : str
    date_col : str
    long_deciles : sequence of int
        Deciles to go long (e.g., (10,) for top decile).
    short_deciles : sequence of int
        Deciles to short.

    Returns
    -------
    DataFrame
        Index: date
        Columns: decile_1 ... decile_K and 'long_short'.
    """
    grp = df.groupby([date_col, decile_col])[ret_col].mean()
    port = grp.unstack(decile_col).sort_index(axis=1)
    port.columns = [f"decile_{int(c)}" for c in port.columns]

    # Long-short portfolio
    long_cols = [f"decile_{d}" for d in long_deciles if f"decile_{d}" in port.columns]
    short_cols = [f"decile_{d}" for d in short_deciles if f"decile_{d}" in port.columns]

    if long_cols and short_cols:
        long_ret = port[long_cols].mean(axis=1)
        short_ret = port[short_cols].mean(axis=1)
        port["long_short"] = long_ret - short_ret

    port.index.name = date_col
    return port


def robust_tstat_calc(
    series: pd.Series,
    max_lags: int = 3,
) -> Dict[str, float]:
    """
    Computes the t-statistic for the mean of a time series using
    Newey-West (HAC) robust standard errors.

    Parameters
    ----------
    series : Series
        Time series of returns or factor premia.
    max_lags : int
        Number of lags for HAC covariance.

    Returns
    -------
    dict
        {'alpha': mean, 'tstat': t-statistic, 'se': robust_se}
    """
    y = series.dropna().values
    if len(y) < 3:
        return {"alpha": np.nan, "tstat": np.nan, "se": np.nan}

    X = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit()
    cov = cov_hac(model, maxlags=max_lags)
    se = np.sqrt(cov[0, 0])
    alpha = model.params[0]
    tstat = alpha / se if se > 0 else np.nan
    return {"alpha": alpha, "tstat": tstat, "se": se}


@dataclass
class FamaMacBethResult:
    risk_premia: pd.Series
    t_stats: pd.Series
    betas_ts: pd.DataFrame  # time series of betas per date


def fama_macbeth_ols(
    df: pd.DataFrame,
    ret_col: str,
    factor_cols: Sequence[str],
    date_col: str = "date",
    max_lags: int = 3,
) -> FamaMacBethResult:
    """
    Two-step Fama-MacBeth regression for factor risk premia estimation.

    Step 1: For each date t, run cross-sectional OLS:
        r_it = a_t + b_t' f_it + e_it
    Step 2: Average b_t over time and compute Newey-West t-stats.

    Parameters
    ----------
    df : DataFrame
        Must contain [date_col, ret_col] + factor_cols.
    ret_col : str
    factor_cols : list of str
    date_col : str
    max_lags : int
        Lags for Newey-West when computing t-stats on betas.

    Returns
    -------
    FamaMacBethResult
    """
    betas_list = []
    dates = []

    for dt, group in df.groupby(date_col):
        if len(group) <= len(factor_cols) + 1:
            continue

        y = group[ret_col].values
        X = group[list(factor_cols)]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        betas_list.append(model.params)
        dates.append(dt)

    betas_ts = pd.DataFrame(betas_list, index=pd.Index(dates, name=date_col))
    risk_premia = betas_ts.mean()

    t_stats = {}
    for col in betas_ts.columns:
        info = robust_tstat_calc(betas_ts[col], max_lags=max_lags)
        t_stats[col] = info["tstat"]

    t_stats = pd.Series(t_stats)
    return FamaMacBethResult(
        risk_premia=risk_premia,
        t_stats=t_stats,
        betas_ts=betas_ts,
    )


def event_study_engine(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset",
    ret_col: str = "ret",
    event_date_col: str = "event_date",
    window: Tuple[int, int] = (-5, 5),
) -> Dict[str, pd.DataFrame]:
    """
    Pools returns into events and evaluates the response around events.

    Parameters
    ----------
    returns : DataFrame
        Columns: [date_col, asset_col, ret_col].
    events : DataFrame
        Columns: [asset_col, event_date_col].
    window : (int, int)
        Event window in periods (e.g., days or months).
        (-5, 5) -> from T-5 to T+5.

    Returns
    -------
    dict with:
      'event_panel' : stacked event-time returns
      'average_path' : avg return by event-time (tau) with cumulative return
    """
    returns = returns.copy()
    returns.set_index([asset_col, date_col], inplace=True)
    records = []

    min_tau, max_tau = window

    for _, ev in events.iterrows():
        a = ev[asset_col]
        t0 = pd.to_datetime(ev[event_date_col])

        for tau in range(min_tau, max_tau + 1):
            dt = t0 + (tau - 0) * (returns.index.levels[1].freq or pd.offsets.Day())
            # Fallback: if no freq, we just look for exact dt
            try:
                r = returns.loc[(a, dt), ret_col]
            except KeyError:
                continue

            records.append(
                {
                    asset_col: a,
                    "event_date": t0,
                    "tau": tau,
                    "ret": float(r),
                }
            )

    event_panel = pd.DataFrame(records)
    if event_panel.empty:
        return {"event_panel": event_panel, "average_path": event_panel}

    avg = (
        event_panel.groupby("tau")["ret"]
        .mean()
        .reset_index()
        .sort_values("tau")
        .reset_index(drop=True)
    )
    avg["cum_ret"] = (1.0 + avg["ret"]).cumprod() - 1.0

    return {"event_panel": event_panel, "average_path": avg}


def term_structure_analyzer(
    residual_returns: pd.Series,
    freq_per_year: int = 12,
) -> Dict[str, float]:
    """
    Evaluates the signal's horizon by estimating an AR(1) on portfolio-level
    (residual) returns and computing the half-life.

    Parameters
    ----------
    residual_returns : Series
        Time series of portfolio residual returns.
    freq_per_year : int
        Observations per year (12 for monthly, 252 for daily).

    Returns
    -------
    dict
        {
          'phi1': AR(1) coefficient,
          'half_life_periods': half-life in periods,
          'half_life_years': half-life in years
        }
    """
    y = residual_returns.dropna()
    if len(y) < 10:
        return {"phi1": np.nan, "half_life_periods": np.nan, "half_life_years": np.nan}

    model = AutoReg(y, lags=1, old_names=False).fit()
    # The first lag coefficient (may be named 'y.L1' or similar)
    phi1 = float(model.params[1])

    if 0 < phi1 < 1:
        half_life = -np.log(2) / np.log(phi1)
        half_life_years = half_life / freq_per_year
    else:
        half_life = np.inf
        half_life_years = np.inf

    return {
        "phi1": phi1,
        "half_life_periods": half_life,
        "half_life_years": half_life_years,
    }


def decile_return_analyzer(
    port_rets: pd.DataFrame,
    periods_per_year: int = 12,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Calculates average returns by decile and basic diagnostics for
    monotonicity and jump behavior.

    Parameters
    ----------
    port_rets : DataFrame
        Output from cross_sectional_backtester (columns decile_1,...).
    periods_per_year : int
        For annualizing Sharpe ratios.

    Returns
    -------
    stats_df : DataFrame
        Columns: ['decile', 'mean_ret', 'std_ret', 'ann_sharpe', 'delta_vs_prev']
    diagnostics : dict
        {'spearman_rho': ..., 'spearman_pvalue': ...}
    """
    decile_cols = [c for c in port_rets.columns if c.startswith("decile_")]
    stats = []

    for c in decile_cols:
        d = int(c.split("_")[1])
        r = port_rets[c].dropna()
        if len(r) == 0:
            mean_r = std_r = sharpe = np.nan
        else:
            mean_r = r.mean()
            std_r = r.std(ddof=1)
            sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else np.nan
        stats.append(
            {
                "decile": d,
                "mean_ret": mean_r,
                "std_ret": std_r,
                "ann_sharpe": sharpe,
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("decile").reset_index(drop=True)
    stats_df["delta_vs_prev"] = stats_df["mean_ret"].diff()

    # Monotonicity via Spearman rank correlation of decile index vs mean_ret
    if stats_df["mean_ret"].notna().sum() >= 3:
        rho, p = spearmanr(stats_df["decile"], stats_df["mean_ret"])
    else:
        rho = p = np.nan

    diagnostics = {"spearman_rho": rho, "spearman_pvalue": p}
    return stats_df, diagnostics


def decile_sharpe_analyzer(
    port_rets: pd.DataFrame,
    ls_col: str = "long_short",
    periods_per_year: int = 12,
) -> Dict[str, Any]:
    """
    Evaluates stability of the long-minus-short Sharpe ratio across time.

    Parameters
    ----------
    port_rets : DataFrame
        Output from cross_sectional_backtester, must contain ls_col.
        Index should be a DatetimeIndex.
    ls_col : str
        Column name for high-minus-low portfolio.
    periods_per_year : int
        E.g. 12 for monthly, 252 for daily.

    Returns
    -------
    dict
        {
          'overall_sharpe': float,
          'by_year': DataFrame with annual Sharpe ratios
        }
    """
    r = port_rets[ls_col].dropna()
    if len(r) == 0:
        return {"overall_sharpe": np.nan, "by_year": pd.DataFrame()}

    overall_mean = r.mean()
    overall_std = r.std(ddof=1)
    overall_sharpe = (
        (overall_mean / overall_std) * np.sqrt(periods_per_year) if overall_std > 0 else np.nan
    )

    by_year = (
        r.groupby(r.index.year)
        .agg(["mean", "std", "count"])
        .rename_axis(index="year")
    )
    by_year["ann_sharpe"] = (
        (by_year["mean"] / by_year["std"]) * np.sqrt(periods_per_year)
    )

    return {"overall_sharpe": overall_sharpe, "by_year": by_year}


def panel_analysis_runner(
    df: pd.DataFrame,
    ret_col: str = "ret",
    signal_col: str = "signal",
    conditioning_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Runs cross-sectional regressions with conditioning variables (including
    dummy variables such as size/style buckets) for each date.

    Example model for each date t:
        r_it = a_t + b_t * signal_it + sum_k c_{k,t} * X_{k,it} + e_it

    Parameters
    ----------
    df : DataFrame
        Panel data with at least [date_col, ret_col, signal_col].
    ret_col : str
    signal_col : str
    conditioning_cols : sequence of str, optional
        Additional regressors (numeric or categorical).
    categorical_cols : sequence of str, optional
        Subset of conditioning_cols treated as categorical (dummied).
    date_col : str

    Returns
    -------
    DataFrame
        Index: date
        Columns: ['beta_signal'] + optionally other betas if needed.
    """
    if conditioning_cols is None:
        conditioning_cols = []
    if categorical_cols is None:
        categorical_cols = []

    results = []

    for dt, g in df.groupby(date_col):
        if len(g) <= 3:
            continue

        # Build formula string
        terms = [signal_col]
        for col in conditioning_cols:
            if col in categorical_cols:
                terms.append(f"C({col})")
            else:
                terms.append(col)
        rhs = " + ".join(terms)
        formula = f"{ret_col} ~ {rhs}"

        try:
            model = smf.ols(formula, data=g).fit()
        except Exception:
            continue

        row = {"date": dt}
        if signal_col in model.params:
            row["beta_signal"] = model.params[signal_col]
        else:
            row["beta_signal"] = np.nan

        # Optionally capture more betas if you want
        results.append(row)

    betas = pd.DataFrame(results).set_index("date").sort_index()
    return betas


_theme_model: Optional[SentenceTransformer] = None


def _get_theme_model() -> SentenceTransformer:
    global _theme_model
    if _theme_model is None:
        _theme_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _theme_model


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
    )


def theme_similarity_retriever(
    query: str,
    collection_name: str = "qr_themes",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieves similar CoT traces, research themes, or signals from a vector DB
    (Qdrant) given a natural language query. Used as a memory / recall tool
    for the agent.

    Assumes vectors were stored with payload fields like:
      {'id': ..., 'text': ..., 'meta': {...}}

    Parameters
    ----------
    query : str
    collection_name : str
    top_k : int

    Returns
    -------
    list of dict
        Each item includes similarity score and stored payload.
    """
    model = _get_theme_model()
    client = _get_qdrant_client()

    q_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=False,
        show_progress_bar=False,
    )[0]

    res = client.search(
        collection_name=collection_name,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )

    out = []
    for p in res:
        out.append(
            {
                "score": float(p.score),
                "payload": p.payload,
            }
        )
    return out


@tool
def search_research_themes(query: str, top_k: int = 5) -> str:
    """
    Search for similar research themes, signals, or analysis from the vector database.

    Args:
        query: Natural language query describing what you're looking for
        top_k: Number of results to return (default: 5)

    Returns:
        String representation of the search results with scores and content
    """
    results = theme_similarity_retriever(query, top_k=top_k)

    output = f"Found {len(results)} similar research themes:\n\n"
    for i, result in enumerate(results, 1):
        output += f"{i}. Score: {result['score']:.3f}\n"
        output += f"   Content: {result['payload']}\n\n"

    return output


TOOLS = [search_research_themes]
