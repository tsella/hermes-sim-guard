"""GUARD §4.2 — Paired Regret Ranking (Analyze Stage) & Trajectory Contrasts.

Computes criterion obedience ``obey(c; W)``, directional regret ``Delta_hurt,t(c)``,
incumbent shortfall ``Delta_gap,t(c)``, and ``Priority_t(c) = Delta_hurt + Delta_gap in [0, 2]``.
Partitions criteria into:
  1. Regressed (Delta_hurt > 0) -> wrap offending rules in ``[iff <trigger>]`` preconditions
  2. Unsolved Gap (Delta_hurt == 0 and Delta_gap > 0) -> add method + verification check
  3. Mastered (Priority == 0) -> pin against overwriting
Also pairs high/low trajectories ``(tau+, tau-)`` per Eq. (5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CriterionRegret:
    criterion_id: str
    desc: str
    obey_prev: Optional[float]
    obey_curr: float
    delta_hurt: float
    delta_gap: float
    priority: float
    bucket: str  # "regressed" | "unsolved_gap" | "mastered"


@dataclass(frozen=True)
class TrajectoryContrast:
    task_signature: str
    top_score: float
    top_strict: float
    bottom_score: float
    bottom_strict: float
    top_summary: str
    bottom_summary: str


@dataclass(frozen=True)
class RegretBundle:
    regressed: tuple[CriterionRegret, ...]
    unsolved_gap: tuple[CriterionRegret, ...]
    mastered: tuple[CriterionRegret, ...]
    contrasts: tuple[TrajectoryContrast, ...]
    mean_delta_hurt: float
    mean_delta_gap: float

    def to_prompt_block(
        self,
        *,
        rejection_entries: Optional[list[dict[str, Any]]] = None,
        playbook_max_chars: int = 2600,
    ) -> str:
        """Render the bounded GUARD Regret & Contrast Bundle (Box B.2a)."""
        lines = [
            "### GUARD REGRET & CONTRAST BUNDLE (arXiv:2611.23476)",
            f"Budget: keep updated SKILL.md body concise (<= {playbook_max_chars} chars for rules).",
        ]
        if self.regressed:
            lines.append("\n1. REGRESSED (Delta_hurt > 0 — wrap offending rules in [iff <precondition>] guards, do NOT discard specialized fixes):")
            for item in self.regressed[:8]:
                lines.append(
                    f"   - [{item.criterion_id}] Priority={item.priority:.2f} (hurt={item.delta_hurt:.2f}, gap={item.delta_gap:.2f}) — {item.desc}"
                )
        if self.unsolved_gap:
            lines.append("\n2. UNSOLVED GAP (Delta_hurt = 0, Delta_gap > 0 — add targeted procedure + check):")
            for item in self.unsolved_gap[:8]:
                lines.append(
                    f"   - [{item.criterion_id}] Priority={item.priority:.2f} (obey={item.obey_curr:.2f}) — {item.desc}"
                )
        if self.mastered:
            lines.append("\n3. MASTERED (Priority = 0 — PINNED invariants; DO NOT overwrite or break these):")
            for item in self.mastered[:10]:
                lines.append(f"   - [{item.criterion_id}] obey=1.00 — {item.desc}")
        if self.contrasts:
            lines.append("\n4. TRAJECTORY CONTRASTS (tau+ vs tau-):")
            for idx, c in enumerate(self.contrasts[:4], start=1):
                lines.append(
                    f"   - Contrast #{idx}: top(strict={c.top_strict:.1f}, S={c.top_score:.1f}) {c.top_summary[:140]} "
                    f"VS bottom(strict={c.bottom_strict:.1f}, S={c.bottom_score:.1f}) {c.bottom_summary[:140]}"
                )
        if rejection_entries:
            lines.append("\n5. ACTIVE REJECTION LEDGER (do NOT re-emit these rejected edits verbatim; narrow with [iff] if revisiting):")
            for entry in rejection_entries:
                lines.append(
                    f"   - [r{entry.get('round', '?')}] skill={entry.get('skill', '?')} "
                    f"hurt={entry.get('hurt', 0):.2f} gain={entry.get('net_gain', 0):.2f} "
                    f"reason={entry.get('reason', '')} excerpt={str(entry.get('excerpt', ''))[:140]}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        def _row(r: CriterionRegret) -> dict[str, Any]:
            return {
                "criterion_id": r.criterion_id,
                "desc": r.desc,
                "obey_prev": r.obey_prev,
                "obey_curr": r.obey_curr,
                "delta_hurt": r.delta_hurt,
                "delta_gap": r.delta_gap,
                "priority": r.priority,
                "bucket": r.bucket,
            }

        return {
            "regressed": [_row(r) for r in self.regressed],
            "unsolved_gap": [_row(r) for r in self.unsolved_gap],
            "mastered": [_row(r) for r in self.mastered],
            "mean_delta_hurt": round(self.mean_delta_hurt, 4),
            "mean_delta_gap": round(self.mean_delta_gap, 4),
            "contrast_count": len(self.contrasts),
        }


def compute_criterion_obedience(graded_turns: list[dict[str, Any]]) -> dict[str, tuple[float, str]]:
    """Eq. (3): ``obey(c; W) = (1/|T_c(W)|) * sum r(tau, c)`` mapped with latest description."""
    totals: dict[str, int] = {}
    passes: dict[str, int] = {}
    descs: dict[str, str] = {}
    for turn in graded_turns:
        for v in turn.get("verdicts") or []:
            cid = str(v.get("id") or "").strip()
            if not cid:
                continue
            totals[cid] = totals.get(cid, 0) + 1
            if v.get("passed"):
                passes[cid] = passes.get(cid, 0) + 1
            if v.get("desc"):
                descs[cid] = str(v["desc"])
    return {cid: (passes.get(cid, 0) / count, descs.get(cid, cid)) for cid, count in totals.items() if count > 0}


def build_regret_bundle(
    *,
    current_turns: list[dict[str, Any]],
    previous_turns: Optional[list[dict[str, Any]]] = None,
    rejected_turns: Optional[list[dict[str, Any]]] = None,
    delta_con: float = 10.0,
) -> RegretBundle:
    """GUARD §4.2 Eq. (4)–(5): compute paired regret ranking and trajectory contrasts.

    When ``rejected_turns`` is provided (after a rejected gate round), ``Delta_hurt`` compares
    ``current_turns`` (incumbent W_{t-1}) against ``rejected_turns`` (rejected winner W~_{t-1}),
    while ``Delta_gap`` and Mastered pinning stay anchored to the incumbent.
    """
    curr_obey = compute_criterion_obedience(current_turns)
    prev_obey = compute_criterion_obedience(previous_turns or [])
    rej_obey = compute_criterion_obedience(rejected_turns or []) if rejected_turns else None

    regressed: list[CriterionRegret] = []
    unsolved: list[CriterionRegret] = []
    mastered: list[CriterionRegret] = []

    for cid, (obey_curr, desc) in curr_obey.items():
        if rej_obey is not None and cid in rej_obey:
            # Post-rejection: measure what the rejected edit broke relative to the incumbent.
            delta_hurt = max(0.0, obey_curr - rej_obey[cid][0])
            obey_prev: Optional[float] = rej_obey[cid][0]
        elif cid in prev_obey:
            obey_prev = prev_obey[cid][0]
            delta_hurt = max(0.0, obey_prev - obey_curr)
        else:
            obey_prev = None
            delta_hurt = 0.0

        delta_gap = max(0.0, 1.0 - obey_curr)
        priority = delta_hurt + delta_gap

        if delta_hurt > 0.0:
            bucket = "regressed"
        elif delta_gap > 0.0:
            bucket = "unsolved_gap"
        else:
            bucket = "mastered"

        item = CriterionRegret(
            criterion_id=cid,
            desc=desc,
            obey_prev=round(obey_prev, 4) if obey_prev is not None else None,
            obey_curr=round(obey_curr, 4),
            delta_hurt=round(delta_hurt, 4),
            delta_gap=round(delta_gap, 4),
            priority=round(priority, 4),
            bucket=bucket,
        )
        if bucket == "regressed":
            regressed.append(item)
        elif bucket == "unsolved_gap":
            unsolved.append(item)
        else:
            mastered.append(item)

    sort_key = lambda r: (-r.priority, -r.delta_hurt, r.criterion_id)
    regressed.sort(key=sort_key)
    unsolved.sort(key=sort_key)
    mastered.sort(key=lambda r: r.criterion_id)

    contrasts = _extract_trajectory_contrasts(current_turns, delta_con=delta_con)
    all_items = regressed + unsolved + mastered
    mean_hurt = sum(r.delta_hurt for r in all_items) / len(all_items) if all_items else 0.0
    mean_gap = sum(r.delta_gap for r in all_items) / len(all_items) if all_items else 0.0
    return RegretBundle(
        regressed=tuple(regressed),
        unsolved_gap=tuple(unsolved),
        mastered=tuple(mastered),
        contrasts=tuple(contrasts),
        mean_delta_hurt=mean_hurt,
        mean_delta_gap=mean_gap,
    )


def _extract_trajectory_contrasts(
    graded_turns: list[dict[str, Any]], *, delta_con: float = 10.0,
) -> list[TrajectoryContrast]:
    """Pair top/bottom trajectories per skill/group when Eq. (5) holds."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for t in graded_turns:
        skills = (t.get("meta") or {}).get("skills_loaded") or []
        key = str(skills[0]) if skills else str(t.get("task_signature") or "general").split("|", 1)[0]
        by_group.setdefault(key, []).append(t)

    contrasts: list[TrajectoryContrast] = []
    for key, group in by_group.items():
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda x: (float(x.get("strict_score", 0.0)), float(x.get("score", 0.0))),
        )
        bottom, top = ordered[0], ordered[-1]
        d_strict = float(top.get("strict_score", 0.0)) - float(bottom.get("strict_score", 0.0))
        d_cont = float(top.get("score", 0.0)) - float(bottom.get("score", 0.0))
        if d_strict > 0.0 or d_cont >= delta_con:
            contrasts.append(TrajectoryContrast(
                task_signature=key,
                top_score=float(top.get("score", 0.0)),
                top_strict=float(top.get("strict_score", 0.0)),
                bottom_score=float(bottom.get("score", 0.0)),
                bottom_strict=float(bottom.get("strict_score", 0.0)),
                top_summary=_summarize_verdicts(top.get("verdicts") or []),
                bottom_summary=_summarize_verdicts(bottom.get("verdicts") or []),
            ))
    return contrasts


def _summarize_verdicts(verdicts: list[dict[str, Any]]) -> str:
    failed = [f"{v.get('id')}: {v.get('why','')}" for v in verdicts if not v.get("passed")]
    if failed:
        return "failed[" + "; ".join(failed[:3]) + "]"
    return f"all {len(verdicts)} criteria passed"
