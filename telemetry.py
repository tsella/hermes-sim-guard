"""Rolling Pre/Post Attribution Engine & Append-Only Telemetry Store.

Persists typed JSONL events under ``<plugin_data_dir>/telemetry.jsonl`` and computes
rolling Pre/Post attribution across the **5 core quantitative metrics** for any rolling
window size ``W`` (default 20 eligible turns before/after each promoted skill edit):

  1. Checklist Obedience & Strict Pass Rate (Delta S_bar, Delta S_strict_bar)
  2. Directional Regret & Soft-HURT Gate Health (mean Delta_hurt, hurt/help ratio,
     accept / repair / reject / rollback rates)
  3. SIM Prediction Calibration & Transfer Retention (predicted->fixed precision,
     diagnosis reuse rate, cross-model/cross-domain retention delta)
  4. Execution Efficiency Delta (Pre vs Post tokens/turn, cache-read ratio, tool iters/turn)
  5. Skill Hygiene & Reuse (reuse_after_patch rate, skill byte-size vs B_W compliance,
     churn/rollback ratio)
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _delta(post: Optional[float], pre: Optional[float]) -> Optional[float]:
    if post is None or pre is None:
        return None
    return round(post - pre, 3)


def _pct_delta(post: Optional[float], pre: Optional[float]) -> Optional[float]:
    if post is None or pre is None or pre == 0:
        return None
    return round(100.0 * (post - pre) / pre, 2)


class TelemetryStore:
    """Thread-safe append-only JSONL telemetry store for SIM+GUARD attribution."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "ts": _now_iso(),
            "event": event_type,
            **payload,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_events(self, *, event_type: Optional[str] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and (event_type is None or obj.get("event") == event_type):
                    rows.append(obj)
        except OSError:
            return []
        if limit is not None and limit > 0:
            return rows[-limit:]
        return rows


def compute_rolling_attribution(
    events: list[dict[str, Any]],
    *,
    sim_stats: Optional[dict[str, Any]] = None,
    window: int = 20,
    skill_filter: Optional[str] = None,
    playbook_max_chars: int = 2600,
) -> dict[str, Any]:
    """Compute rolling Pre vs Post attribution across all 5 core quantitative metrics."""
    window = max(1, int(window))

    # Index turn_graded events in chronological order alongside edit epochs.
    turns: list[tuple[int, dict[str, Any]]] = []
    epochs: list[tuple[int, dict[str, Any]]] = []
    gates: list[dict[str, Any]] = []
    skill_events: list[dict[str, Any]] = []
    curator_runs: list[dict[str, Any]] = []

    for idx, ev in enumerate(events):
        etype = ev.get("event")
        if etype == "turn_graded":
            if skill_filter:
                loaded = set(ev.get("skills_loaded") or [])
                if skill_filter not in loaded:
                    continue
            turns.append((idx, ev))
        elif etype == "skill_edit_epoch":
            if skill_filter:
                touched = set(ev.get("skills") or [])
                if skill_filter not in touched:
                    continue
            epochs.append((idx, ev))
        elif etype == "gate_decision":
            if skill_filter and ev.get("skill") != skill_filter:
                continue
            gates.append(ev)
        elif etype == "skill_lifecycle":
            if skill_filter and ev.get("skill_name") != skill_filter:
                continue
            skill_events.append(ev)
        elif etype == "curator_run":
            curator_runs.append(ev)

    # Per-epoch Pre/Post slices (each epoch compares up to W turns strictly before vs after).
    epoch_attributions: list[dict[str, Any]] = []
    all_pre_scores: list[float] = []
    all_post_scores: list[float] = []
    all_pre_strict: list[float] = []
    all_post_strict: list[float] = []
    all_pre_tokens: list[float] = []
    all_post_tokens: list[float] = []
    all_pre_iters: list[float] = []
    all_post_iters: list[float] = []
    all_pre_cache: list[float] = []
    all_post_cache: list[float] = []

    for ep_idx, ep in epochs:
        ep_skills = set(ep.get("skills") or [])
        # Prefer skill-matched turns when available, falling back to all turns in rolling window.
        scoped_turns = [
            (t_idx, t) for t_idx, t in turns
            if not ep_skills or bool(set(t.get("skills_loaded") or []) & ep_skills)
        ] or turns
        pre_slice = [t for t_idx, t in scoped_turns if t_idx < ep_idx][-window:]
        post_slice = [t for t_idx, t in scoped_turns if t_idx > ep_idx][:window]

        pre_s = [float(t.get("score", 0.0)) for t in pre_slice]
        post_s = [float(t.get("score", 0.0)) for t in post_slice]
        pre_st = [float(t.get("strict_score", 0.0)) for t in pre_slice]
        post_st = [float(t.get("strict_score", 0.0)) for t in post_slice]
        pre_tok = [float(t.get("total_tokens", 0)) for t in pre_slice if t.get("total_tokens") is not None]
        post_tok = [float(t.get("total_tokens", 0)) for t in post_slice if t.get("total_tokens") is not None]
        pre_it = [float(t.get("tool_iterations", 0)) for t in pre_slice]
        post_it = [float(t.get("tool_iterations", 0)) for t in post_slice]
        pre_cr = [float(t["cache_read_ratio"]) for t in pre_slice if t.get("cache_read_ratio") is not None]
        post_cr = [float(t["cache_read_ratio"]) for t in post_slice if t.get("cache_read_ratio") is not None]

        all_pre_scores.extend(pre_s)
        all_post_scores.extend(post_s)
        all_pre_strict.extend(pre_st)
        all_post_strict.extend(post_st)
        all_pre_tokens.extend(pre_tok)
        all_post_tokens.extend(post_tok)
        all_pre_iters.extend(pre_it)
        all_post_iters.extend(post_it)
        all_pre_cache.extend(pre_cr)
        all_post_cache.extend(post_cr)

        epoch_attributions.append({
            "ts": ep.get("ts"),
            "round": ep.get("round"),
            "skills": list(ep.get("skills") or []),
            "mode": ep.get("mode", ""),
            "problem_id": ep.get("problem_id", ""),
            "pre_n": len(pre_slice),
            "post_n": len(post_slice),
            "pre_score": _safe_mean(pre_s),
            "post_score": _safe_mean(post_s),
            "delta_score": _delta(_safe_mean(post_s), _safe_mean(pre_s)),
            "pre_strict": _safe_mean(pre_st),
            "post_strict": _safe_mean(post_st),
            "delta_strict": _delta(_safe_mean(post_st), _safe_mean(pre_st)),
            "delta_tokens_pct": _pct_delta(_safe_mean(post_tok), _safe_mean(pre_tok)),
            "delta_iters": _delta(_safe_mean(post_it), _safe_mean(pre_it)),
        })

    # If no edit epochs yet, compare first half vs second half of the rolling window as baseline trend.
    if not epochs and len(turns) >= 2:
        recent = [t for _, t in turns[-2 * window :]]
        mid = len(recent) // 2
        pre_half, post_half = recent[:mid], recent[mid:]
        all_pre_scores = [float(t.get("score", 0.0)) for t in pre_half]
        all_post_scores = [float(t.get("score", 0.0)) for t in post_half]
        all_pre_strict = [float(t.get("strict_score", 0.0)) for t in pre_half]
        all_post_strict = [float(t.get("strict_score", 0.0)) for t in post_half]
        all_pre_tokens = [float(t.get("total_tokens", 0)) for t in pre_half if t.get("total_tokens") is not None]
        all_post_tokens = [float(t.get("total_tokens", 0)) for t in post_half if t.get("total_tokens") is not None]
        all_pre_iters = [float(t.get("tool_iterations", 0)) for t in pre_half]
        all_post_iters = [float(t.get("tool_iterations", 0)) for t in post_half]
        all_pre_cache = [float(t["cache_read_ratio"]) for t in pre_half if t.get("cache_read_ratio") is not None]
        all_post_cache = [float(t["cache_read_ratio"]) for t in post_half if t.get("cache_read_ratio") is not None]

    # --- Metric 1: Checklist Obedience & Strict Pass Rate ---
    overall_scores = [float(t.get("score", 0.0)) for _, t in turns]
    overall_strict = [float(t.get("strict_score", 0.0)) for _, t in turns]
    pre_s_mean, post_s_mean = _safe_mean(all_pre_scores), _safe_mean(all_post_scores)
    pre_st_mean, post_st_mean = _safe_mean(all_pre_strict), _safe_mean(all_post_strict)
    metric_1 = {
        "name": "Checklist Obedience & Strict Pass Rate",
        "turns_graded": len(turns),
        "overall_score": _safe_mean(overall_scores),
        "overall_strict": _safe_mean(overall_strict),
        "pre_score": pre_s_mean,
        "post_score": post_s_mean,
        "delta_score": _delta(post_s_mean, pre_s_mean),
        "pre_strict": pre_st_mean,
        "post_strict": post_st_mean,
        "delta_strict": _delta(post_st_mean, pre_st_mean),
    }

    # --- Metric 2: Directional Regret & Soft-HURT Gate Health ---
    gate_total = len(gates)
    gate_promoted = sum(1 for g in gates if g.get("promoted"))
    gate_repaired = sum(1 for g in gates if g.get("repaired"))
    gate_rejected = gate_total - gate_promoted
    gate_rollbacks = sum(1 for g in gates if g.get("rolled_back"))
    total_hurt_ep = sum(int(g.get("hurt_count", 0)) for g in gates)
    total_help_ep = sum(int(g.get("helped_count", 0)) for g in gates)
    hurt_penalties = [float(g.get("hurt_penalty", 0.0)) for g in gates]
    delta_hurts = [float(t["mean_delta_hurt"]) for _, t in turns if t.get("mean_delta_hurt") is not None]
    metric_2 = {
        "name": "Directional Regret & Soft-HURT Gate Health",
        "gate_evaluations": gate_total,
        "promoted": gate_promoted,
        "repaired": gate_repaired,
        "rejected": gate_rejected,
        "rollbacks": gate_rollbacks,
        "accept_rate": round(gate_promoted / gate_total, 4) if gate_total else None,
        "rollback_rate": round(gate_rollbacks / gate_total, 4) if gate_total else None,
        "mean_hurt_penalty": _safe_mean(hurt_penalties),
        "mean_delta_hurt": _safe_mean(delta_hurts),
        "hurt_to_help_ratio": round(total_hurt_ep / total_help_ep, 4) if total_help_ep > 0 else (float(total_hurt_ep) if total_hurt_ep else 0.0),
    }

    # --- Metric 3: SIM Prediction Calibration & Transfer Retention ---
    cross_model_delta = _compute_transfer_retention_delta([t for _, t in turns], key="model")
    cross_domain_delta = _compute_transfer_retention_delta([t for _, t in turns], key="platform")
    sim_data = dict(sim_stats or {})
    metric_3 = {
        "name": "SIM Prediction Calibration & Transfer Retention",
        "problem_count": sim_data.get("problem_count", 0),
        "attempt_count": sim_data.get("attempt_count", 0),
        "prediction_precision": sim_data.get("prediction_precision"),
        "diagnosis_reuse_rate": sim_data.get("diagnosis_reuse_rate", 0.0),
        "cross_model_continuations": sim_data.get("cross_model_continuations", 0),
        "cross_model_retention_delta": cross_model_delta,
        "cross_domain_retention_delta": cross_domain_delta,
        "models_seen": sim_data.get("models_seen", []),
    }

    # --- Metric 4: Execution Efficiency Delta ---
    pre_tok_mean, post_tok_mean = _safe_mean(all_pre_tokens), _safe_mean(all_post_tokens)
    pre_it_mean, post_it_mean = _safe_mean(all_pre_iters), _safe_mean(all_post_iters)
    pre_cr_mean, post_cr_mean = _safe_mean(all_pre_cache), _safe_mean(all_post_cache)
    metric_4 = {
        "name": "Execution Efficiency Delta",
        "pre_tokens_per_turn": pre_tok_mean,
        "post_tokens_per_turn": post_tok_mean,
        "delta_tokens_pct": _pct_delta(post_tok_mean, pre_tok_mean),
        "pre_cache_read_ratio": pre_cr_mean,
        "post_cache_read_ratio": post_cr_mean,
        "delta_cache_read_ratio": _delta(post_cr_mean, pre_cr_mean),
        "pre_tool_iters": pre_it_mean,
        "post_tool_iters": post_it_mean,
        "delta_tool_iters": _delta(post_it_mean, pre_it_mean),
    }

    # --- Metric 5: Skill Hygiene & Reuse ---
    loaded_events = [e for e in skill_events if e.get("action") == "loaded"]
    reuse_after_patch_hits = sum(1 for e in loaded_events if e.get("reuse_after_patch"))
    patch_events = [e for e in skill_events if e.get("action") in {"patched", "patch", "created", "create", "edited", "edit"}]
    sizes = [int(ep["skill_chars"]) for _, ep in epochs if isinstance(ep.get("skill_chars"), int)]
    within_budget = sum(1 for sz in sizes if sz <= playbook_max_chars)
    total_mutations = max(len(epochs), len(patch_events))
    metric_5 = {
        "name": "Skill Hygiene & Reuse",
        "skill_loads": len(loaded_events),
        "reuse_after_patch_hits": reuse_after_patch_hits,
        "reuse_after_patch_rate": round(reuse_after_patch_hits / len(loaded_events), 4) if loaded_events else None,
        "edit_epochs": len(epochs),
        "mean_skill_chars": _safe_mean([float(s) for s in sizes]),
        "budget_compliance_rate": round(within_budget / len(sizes), 4) if sizes else None,
        "playbook_max_chars": playbook_max_chars,
        "churn_rollback_ratio": round(gate_rollbacks / total_mutations, 4) if total_mutations > 0 else 0.0,
        "curator_runs_observed": len(curator_runs),
    }

    return {
        "generated_at": _now_iso(),
        "window": window,
        "skill_filter": skill_filter,
        "event_count": len(events),
        "metrics": {
            "1_checklist_obedience": metric_1,
            "2_directional_regret_gate": metric_2,
            "3_sim_calibration_transfer": metric_3,
            "4_execution_efficiency": metric_4,
            "5_skill_hygiene_reuse": metric_5,
        },
        "epochs": epoch_attributions[-20:],
    }


def _compute_transfer_retention_delta(turns: list[dict[str, Any]], *, key: str) -> Optional[float]:
    """Mean score delta across adjacent turns where ``key`` (model or platform) changed."""
    deltas: list[float] = []
    prev_val = None
    prev_scores: list[float] = []
    for idx, t in enumerate(turns):
        val = str(t.get(key) or "").strip()
        if not val:
            continue
        score = float(t.get("score", 0.0))
        if prev_val is None:
            prev_val = val
            prev_scores = [score]
            continue
        if val != prev_val and prev_scores:
            next_scores = [float(x.get("score", 0.0)) for x in turns[idx : idx + 5] if str(x.get(key) or "").strip() == val]
            if next_scores:
                deltas.append((sum(next_scores) / len(next_scores)) - (sum(prev_scores[-5:]) / len(prev_scores[-5:])))
            prev_val = val
            prev_scores = [score]
        else:
            prev_scores.append(score)
    return _safe_mean(deltas)
