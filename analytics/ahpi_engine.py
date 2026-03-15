"""
AHPI (Asymmetric Heterogeneous Pairwise Interactions) Ranking Engine.

Implements an EM algorithm to estimate latent skill scores S_{k,m} for law firms
per case type, defendant biases epsilon_m per case type, and valence probabilities q_m.

Mathematical model:
  rho_n(A) = sigmoid(S_A_m - (S_B_m + epsilon_m))
  p_n(A)   = q_m * rho_n(A) + (1 - q_m) * (1 - rho_n(A))

Time decay: interactions are weighted by w_t = exp(-lambda * (T - t) / 365.25)
so that older cases have less influence on the fitted scores.
"""

import numpy as np
from scipy.special import expit  # sigmoid
from scipy.optimize import minimize
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import warnings


CASE_TYPES = ["Civil Rights", "Contracts", "Labor", "Torts", "Other"]

# Fitted defendant bias initializations from the US legal system
EPSILON_INIT = {
    "Civil Rights": 2.03,
    "Contracts":    1.66,
    "Labor":        1.99,
    "Torts":        0.32,
    "Other":        1.90,
}

Q_INIT = {ct: 0.5 for ct in CASE_TYPES}

Q_FACTOR_MIN = 30  # Minimum interactions per firm for inclusion


@dataclass
class Interaction:
    plaintiff_firm: str
    defendant_firm: str
    case_type: str
    outcome: int  # 1 = plaintiff wins, 0 = defendant wins
    decision_date: Optional[str] = None  # "YYYY-MM-DD"


@dataclass
class AHPIModel:
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    # {firm_name: {case_type: score}}
    epsilon: dict[str, float] = field(default_factory=lambda: dict(EPSILON_INIT))
    q: dict[str, float] = field(default_factory=lambda: dict(Q_INIT))


def apply_q_filter(interactions: list[Interaction]) -> list[Interaction]:
    """
    Filter interactions to only include firms with >= Q_FACTOR_MIN total
    appearances (as plaintiff or defendant) in the dataset.
    """
    counts: dict[str, int] = {}
    for itx in interactions:
        counts[itx.plaintiff_firm] = counts.get(itx.plaintiff_firm, 0) + 1
        counts[itx.defendant_firm] = counts.get(itx.defendant_firm, 0) + 1

    qualified = {firm for firm, cnt in counts.items() if cnt >= Q_FACTOR_MIN}
    filtered = [
        itx for itx in interactions
        if itx.plaintiff_firm in qualified and itx.defendant_firm in qualified
    ]
    removed = len(interactions) - len(filtered)
    if removed:
        print(f"Q-filter: removed {removed} interactions ({len(qualified)} firms qualify).")
    return filtered


def _rho(s_a: float, s_b: float, eps: float) -> float:
    return float(expit(s_a - s_b - eps))


def _win_prob(s_a: float, s_b: float, eps: float, q: float) -> float:
    r = _rho(s_a, s_b, eps)
    return q * r + (1 - q) * (1 - r)


def _neg_log_likelihood(
    params: np.ndarray,
    interactions: list[Interaction],
    firms: list[str],
    case_types: list[str],
    firm_idx: dict[str, int],
    ct_idx: dict[str, int],
    n_firms: int,
    weights: Optional[np.ndarray],
) -> float:
    n_ct = len(case_types)
    scores_2d = params[:n_firms * n_ct].reshape(n_firms, n_ct)
    epsilons = params[n_firms * n_ct: n_firms * n_ct + n_ct]
    qs = params[n_firms * n_ct + n_ct:]

    # Clamp q to (0,1) for stability
    qs = np.clip(qs, 1e-6, 1 - 1e-6)

    nll = 0.0
    for idx, itx in enumerate(interactions):
        i = firm_idx[itx.plaintiff_firm]
        j = firm_idx[itx.defendant_firm]
        m = ct_idx[itx.case_type]
        p = _win_prob(scores_2d[i, m], scores_2d[j, m], epsilons[m], qs[m])
        p = np.clip(p, 1e-9, 1 - 1e-9)
        w = weights[idx] if weights is not None else 1.0
        nll -= w * (itx.outcome * np.log(p) + (1 - itx.outcome) * np.log(1 - p))

    return nll


def fit(
    interactions: list[Interaction],
    max_iter: int = 200,
    tol: float = 1e-6,
    apply_filter: bool = True,
    decay_lambda: float = 0.0,
) -> AHPIModel:
    """
    Fit the AHPI model via joint optimization (quasi-EM via L-BFGS-B).

    Args:
        interactions: List of observed case outcomes.
        max_iter: Maximum optimizer iterations.
        tol: Convergence tolerance.
        apply_filter: Whether to apply Q-factor firm filtering.
        decay_lambda: Time decay rate (years^-1). 0 disables decay.

    Returns:
        Fitted AHPIModel with per-case-type scores, epsilon, and q.
    """
    if apply_filter:
        interactions = apply_q_filter(interactions)

    if not interactions:
        raise ValueError("No interactions remain after Q-factor filtering.")

    firms = sorted({itx.plaintiff_firm for itx in interactions} |
                   {itx.defendant_firm for itx in interactions})
    case_types = sorted({itx.case_type for itx in interactions})

    firm_idx = {f: i for i, f in enumerate(firms)}
    ct_idx = {ct: i for i, ct in enumerate(case_types)}
    n_firms = len(firms)
    n_ct = len(case_types)

    # Compute time decay weights
    if decay_lambda > 0.0 and any(itx.decision_date for itx in interactions):
        today = date.today()
        weights = np.array([
            np.exp(-decay_lambda * max(
                (today - date.fromisoformat(itx.decision_date)).days / 365.25, 0
            )) if itx.decision_date else 1.0
            for itx in interactions
        ])
    else:
        weights = None

    # Initial params: scores=0, epsilon from EPSILON_INIT, q=0.5
    x0 = np.concatenate([
        np.zeros(n_firms * n_ct),
        np.array([EPSILON_INIT.get(ct, 1.0) for ct in case_types]),
        np.array([Q_INIT.get(ct, 0.5) for ct in case_types]),
    ])

    # Bounds: scores unbounded, epsilon capped at 4.0, q in (0,1).
    # Known fitted values for the US legal system are all < 2.5; 4.0 gives
    # headroom while preventing degenerate solutions on sparse/imbalanced data.
    score_bounds = [(None, None)] * (n_firms * n_ct)
    eps_bounds = [(0.0, 4.0)] * n_ct
    q_bounds = [(1e-6, 1 - 1e-6)] * n_ct
    bounds = score_bounds + eps_bounds + q_bounds

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(interactions, firms, case_types, firm_idx, ct_idx, n_firms, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol},
    )

    if not result.success:
        warnings.warn(f"Optimizer did not fully converge: {result.message}")

    params = result.x
    scores_2d = params[:n_firms * n_ct].reshape(n_firms, n_ct)
    epsilons_arr = params[n_firms * n_ct: n_firms * n_ct + n_ct]
    qs_arr = params[n_firms * n_ct + n_ct:]

    # Normalize scores per case type so mean = 0
    for m in range(n_ct):
        scores_2d[:, m] -= scores_2d[:, m].mean()

    # Only keep (firm, case_type) pairs that had at least one observed interaction.
    # Firms with no cases in a given case type receive zero gradient and stay at
    # 0.0000 — including them produces misleading predictions.
    observed_pairs = (
        {(itx.plaintiff_firm, itx.case_type) for itx in interactions}
        | {(itx.defendant_firm, itx.case_type) for itx in interactions}
    )

    model = AHPIModel(
        scores={
            firm: {
                ct: float(scores_2d[i, m])
                for m, ct in enumerate(case_types)
                if (firm, ct) in observed_pairs
            }
            for i, firm in enumerate(firms)
        },
        epsilon={ct: float(epsilons_arr[ct_idx[ct]]) for ct in case_types},
        q={ct: float(np.clip(qs_arr[ct_idx[ct]], 1e-6, 1 - 1e-6)) for ct in case_types},
    )
    return model


def predict_plaintiff_win(
    model: AHPIModel,
    plaintiff_firm: str,
    defendant_firm: str,
    case_type: str,
) -> dict:
    """
    Predict the probability that the plaintiff firm wins.

    Returns a dict with:
        - propensity (rho): raw skill-differential win probability
        - win_probability (p): final probability accounting for valence q
    """
    s_a = model.scores.get(plaintiff_firm, {}).get(case_type)
    s_b = model.scores.get(defendant_firm, {}).get(case_type)

    if s_a is None:
        raise ValueError(
            f"Plaintiff firm '{plaintiff_firm}' has no score for case type '{case_type}'."
        )
    if s_b is None:
        raise ValueError(
            f"Defendant firm '{defendant_firm}' has no score for case type '{case_type}'."
        )

    eps = model.epsilon.get(case_type, EPSILON_INIT.get(case_type, 1.0))
    q = model.q.get(case_type, 0.5)

    rho = _rho(s_a, s_b, eps)
    p = _win_prob(s_a, s_b, eps, q)

    return {
        "plaintiff_firm": plaintiff_firm,
        "defendant_firm": defendant_firm,
        "case_type": case_type,
        "plaintiff_score": s_a,
        "defendant_score": s_b,
        "defendant_bias": eps,
        "valence_q": q,
        "propensity": rho,
        "win_probability": p,
    }


def rank_firms(
    model: AHPIModel,
    case_type: str = "Civil Rights",
    top_n: Optional[int] = None,
) -> list[tuple[str, float]]:
    """Return firms sorted by descending latent skill score for the given case type."""
    ranked = sorted(
        [(firm, ct_scores.get(case_type, 0.0)) for firm, ct_scores in model.scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n] if top_n else ranked
