"""GUARD §4.3–4.4 — Group-Relative Candidate Pooling, ContrastEvolve, Soft-HURT Gate & Rejection Ledger.

Implements:
  - Uniform strategy prompt pool P_{t,g} seeded with s_0 (SHARED_SEED_PROMPT, Box B.3)
  - M >= 2 parallel/sequential candidate distillation via ``sim_evolver`` auxiliary slot
  - PAST-Bench four-criterion joint critic scoring (Box B.1c) + Eq. (7)-(8) fitness R_m & A^_m
  - ContrastEvolve (Eq. (9)-(10), Box B.2b) with task/session ID scrubbing -> (s+_{t+1}, s-_{t+1})
  - Soft-HURT promotion gate (Eq. (11)-(13)) + at most one in-round [iff] scope-guard repair
  - Bounded deduplicated JSONL rejection ledger L_g + automatic rollback via ``tools.skill_ledger``
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

try:
    from .config import AUX_TASK_EVOLVER, AUX_TASK_REVIEWER, SimGuardConfig
except ImportError:
    from config import AUX_TASK_EVOLVER, AUX_TASK_REVIEWER, SimGuardConfig

logger = logging.getLogger(__name__)

# Box B.3 (GUARD Appendix B.4) — domain-agnostic uniform seed prompt s_0.
SHARED_SEED_PROMPT = (
    "Diagnose the incumbent skill's dominant behavioral failure mode from its execution traces and "
    "regret bundle before designing anything. Then propose exactly ONE narrow, high-leverage skill "
    "refinement (SKILL.md rule or procedure patch) that directly removes that failure mode. "
    "Guard the rule with an explicit [iff <precondition>] trigger scope so it cannot fire on tasks "
    "the incumbent already solves, and preserve every pinned Mastered invariant."
)

_DISTILL_INSTRUCTIONS = (
    "You are a GUARD Harness Distiller. Given the strategy prompt, incumbent SKILL.md content, "
    "SIM root-cause history, and GUARD Regret & Contrast Bundle, produce a single improved SKILL.md "
    "candidate within the character budget. Return strict JSON: "
    '{"root_cause": "<hypothesis>", "problem_id": "<existing P# or empty>", '
    '"what": "<concise edit rationale>", "predicted": ["<criterion_id>"], "skill_content": "<full SKILL.md>"}.'
)

_CRITIC_INSTRUCTIONS = (
    "You are a strict GUARD Four-Criterion Critic (arXiv:2611.23476 Box B.1c). "
    "Compare the INCUMBENT skill against each CANDIDATE skill over the recent episode summaries across "
    "four criteria: (1) grounding in real session history without hallucinated preferences, "
    "(2) exact procedure/SOP step fidelity, (3) explicit [iff <precondition>] scope boundaries avoiding "
    "over-triggering or episodic pollution, and (4) complete replacement of superseded rules without bloat. "
    "For each episode index (0..N-1), score INCUMBENT and every CANDIDATE on [0, 100] (continuous and strict). "
    'Return strict JSON: {"incumbent_scores": [{"episode": 0, "score": 80.0, "strict": 40.0}], '
    '"candidates": [{"index": 0, "scores": [{"episode": 0, "score": 90.0, "strict": 100.0}], "summary": "..."}]}.'
)

_CONTRAST_EVOLVE_INSTRUCTIONS = (
    "You are GUARD ContrastEvolve (arXiv:2611.23476 §4.3 Box B.2b). "
    "Given the strategy prompt, excerpt, and metrics (R_m, |Q+|, |Q-|) of the BEST candidate (m+) and "
    "WORST candidate (m-), extract causal learnings explaining why m+ outperformed m- and rewrite BOTH "
    "prompts: updated_best_prompt amplifies m+'s scoped structure while preserving pinned rules; "
    "updated_worst_prompt repairs m-'s over-broad triggers into strict [iff <precondition>] guards. "
    'Return strict JSON: {"learnings": ["..."], "updated_best_prompt": "...", "updated_worst_prompt": "..."}.'
)

_IFF_REPAIR_INSTRUCTIONS = (
    "You are GUARD In-Round [iff] Scope-Guard Repair (arXiv:2611.23476 §4.4). "
    "The candidate skill edit helped on target tasks but caused isolated regressions on the listed hurt "
    "episodes because a new rule fired too broadly. Narrow ONLY the offending rule(s) with explicit "
    "[iff <precondition>] triggers so hurt tasks retain incumbent behavior while keeping the fix. "
    'Return strict JSON: {"what": "<repair note>", "skill_content": "<repaired SKILL.md>"}.'
)

_TASK_ID_SCRUB_RE = re.compile(r"\b(?:sess|task|turn|run|req)[_-][a-zA-Z0-9_-]{3,}\b")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def scrub_ids(text: str) -> str:
    """Scrub task/session identifiers to ``<TASK_ID>`` before ContrastEvolve (§4.3)."""
    return _TASK_ID_SCRUB_RE.sub("<TASK_ID>", text or "")


@dataclass(frozen=True)
class CandidateEvaluation:
    index: int
    strategy_prompt: str
    root_cause: str
    problem_id: str
    what: str
    predicted: tuple[str, ...]
    skill_content: str
    mean_score: float
    mean_strict: float
    incumbent_mean_score: float
    helped_episodes: tuple[int, ...]
    hurt_episodes: tuple[int, ...]
    hurt_penalty: float
    net_gain: float
    fitness: float
    advantage: float = 0.0


@dataclass(frozen=True)
class GateDecision:
    promoted: bool
    repaired: bool
    reason: str
    net_gain: float
    hurt_penalty: float
    helped_count: int
    hurt_count: int
    rolled_back: bool = False


def evaluate_candidate_metrics(
    *,
    incumbent_scores: list[dict[str, float]],
    candidate_scores: list[dict[str, float]],
    delta_noise: float,
    beta_hurt: float,
    gamma_help: float,
) -> dict[str, Any]:
    """Compute GUARD Eq. (7), (8), and (11): Q+, Q-, Hurt(W~, W), Perf, and R_m."""
    inc_by_ep = {int(r.get("episode", i)): r for i, r in enumerate(incumbent_scores)}
    cand_by_ep = {int(r.get("episode", i)): r for i, r in enumerate(candidate_scores)}
    episodes = sorted(set(inc_by_ep) & set(cand_by_ep))
    if not episodes:
        return {
            "mean_score": 0.0,
            "mean_strict": 0.0,
            "incumbent_mean_score": 0.0,
            "helped": [],
            "hurt": [],
            "hurt_penalty": 0.0,
            "net_gain": 0.0,
            "fitness": 0.0,
        }
    inc_s = [float(inc_by_ep[e].get("score", 0.0)) for e in episodes]
    cand_s = [float(cand_by_ep[e].get("score", 0.0)) for e in episodes]
    cand_strict = [float(cand_by_ep[e].get("strict", cand_by_ep[e].get("score", 0.0))) for e in episodes]

    helped: list[int] = []
    hurt: list[int] = []
    hurt_sum = 0.0
    for ep, s_old, s_new in zip(episodes, inc_s, cand_s):
        diff = s_new - s_old
        if diff > delta_noise:
            helped.append(ep)
        elif -diff > delta_noise:
            hurt.append(ep)
            hurt_sum += s_old - s_new

    n = len(episodes)
    mean_inc = sum(inc_s) / n
    mean_cand = sum(cand_s) / n
    mean_strict = sum(cand_strict) / n
    hurt_penalty = hurt_sum / n
    net_gain = mean_cand - mean_inc
    perf = mean_strict + 0.25 * mean_cand
    fitness = perf - beta_hurt * len(hurt) + gamma_help * len(helped)
    return {
        "mean_score": round(mean_cand, 3),
        "mean_strict": round(mean_strict, 3),
        "incumbent_mean_score": round(mean_inc, 3),
        "helped": helped,
        "hurt": hurt,
        "hurt_penalty": round(hurt_penalty, 3),
        "net_gain": round(net_gain, 3),
        "fitness": round(fitness, 3),
    }


def standardize_advantages(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    """GUARD Eq. (8): ``A^_m = (R_m - mu_R) / (sqrt(var) + 1e-6)``."""
    if not candidates:
        return []
    rewards = [c.fitness for c in candidates]
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(var) + 1e-6
    return [
        CandidateEvaluation(
            index=c.index,
            strategy_prompt=c.strategy_prompt,
            root_cause=c.root_cause,
            problem_id=c.problem_id,
            what=c.what,
            predicted=c.predicted,
            skill_content=c.skill_content,
            mean_score=c.mean_score,
            mean_strict=c.mean_strict,
            incumbent_mean_score=c.incumbent_mean_score,
            helped_episodes=c.helped_episodes,
            hurt_episodes=c.hurt_episodes,
            hurt_penalty=c.hurt_penalty,
            net_gain=c.net_gain,
            fitness=c.fitness,
            advantage=round((c.fitness - mu) / sigma, 4),
        )
        for c in candidates
    ]


def check_soft_hurt_gate(
    *,
    net_gain: float,
    hurt_penalty: float,
    helped_count: int,
    hurt_count: int,
    h_max: int,
    mu_min: float,
    is_repair: bool = False,
) -> GateDecision:
    """GUARD §4.4 Eq. (12)–(13): Soft-HURT promotion gate."""
    eff_mu_min = 0.0 if is_repair else mu_min
    if hurt_count > h_max:
        return GateDecision(
            promoted=False,
            repaired=is_repair,
            reason=f"hurt_count({hurt_count}) > h_max({h_max})",
            net_gain=net_gain,
            hurt_penalty=hurt_penalty,
            helped_count=helped_count,
            hurt_count=hurt_count,
        )
    if hurt_count > 0 and helped_count < 1:
        return GateDecision(
            promoted=False,
            repaired=is_repair,
            reason="regression_without_helped_task",
            net_gain=net_gain,
            hurt_penalty=hurt_penalty,
            helped_count=helped_count,
            hurt_count=hurt_count,
        )
    threshold = max(hurt_penalty, eff_mu_min if hurt_count > 0 else 0.0)
    if net_gain <= threshold:
        return GateDecision(
            promoted=False,
            repaired=is_repair,
            reason=f"net_gain({net_gain:.2f}) <= threshold({threshold:.2f})",
            net_gain=net_gain,
            hurt_penalty=hurt_penalty,
            helped_count=helped_count,
            hurt_count=hurt_count,
        )
    return GateDecision(
        promoted=True,
        repaired=is_repair,
        reason="soft_hurt_passed",
        net_gain=net_gain,
        hurt_penalty=hurt_penalty,
        helped_count=helped_count,
        hurt_count=hurt_count,
    )


class RejectionLedger:
    """Bounded, deduplicated on-disk JSONL rejection ledger L_g (§4.4)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def _fingerprint(skill: str, excerpt: str) -> str:
        norm = re.sub(r"\s+", " ", f"{skill.strip().lower()}::{excerpt.strip().lower()[:300]}")
        return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

    def read_recent(self, *, limit: int = 12, skill: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and (skill is None or item.get("skill") == skill):
                    rows.append(item)
        except OSError:
            return []
        return rows[-limit:]

    def record(
        self,
        *,
        round_num: int,
        skill: str,
        reason: str,
        net_gain: float,
        hurt: float,
        excerpt: str,
        helped_count: int = 0,
        hurt_count: int = 0,
    ) -> bool:
        """Append deduplicated active rejection entry; returns True when newly recorded."""
        fp = self._fingerprint(skill, excerpt)
        recent = self.read_recent(limit=200)
        if any(r.get("fingerprint") == fp for r in recent):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": _now_iso(),
            "round": int(round_num),
            "skill": skill,
            "fingerprint": fp,
            "reason": reason,
            "net_gain": round(net_gain, 3),
            "hurt": round(hurt, 3),
            "helped_count": int(helped_count),
            "hurt_count": int(hurt_count),
            "excerpt": scrub_ids(excerpt)[:400],
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True


class StrategyPoolStore:
    """Persisted strategy prompt pool P_{t,g} per skill group (Eq. (6), (10))."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"groups": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"groups": {}}
        except (OSError, ValueError):
            return {"groups": {}}

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def get_prompts(self, group: str, pool_size: int) -> list[str]:
        data = self.load()
        entry = (data.get("groups") or {}).get(group)
        if isinstance(entry, dict) and isinstance(entry.get("prompts"), list) and len(entry["prompts"]) >= 2:
            prompts = [str(p) for p in entry["prompts"] if isinstance(p, str) and p.strip()]
            if len(prompts) >= pool_size:
                return prompts[:pool_size]
            while len(prompts) < pool_size:
                prompts.append(prompts[-1] if prompts else SHARED_SEED_PROMPT)
            return prompts
        return [SHARED_SEED_PROMPT for _ in range(max(2, pool_size))]

    def update_prompts(self, group: str, *, best_prompt: str, worst_prompt: str, learnings: list[str], pool_size: int) -> list[str]:
        """Eq. (10): ``P_{t+1,g} = {s+}^{floor(M/2)} U {s-}^{M - floor(M/2)}``."""
        m = max(2, pool_size)
        n_best = m // 2
        n_worst = m - n_best
        next_pool = [best_prompt or SHARED_SEED_PROMPT] * n_best + [worst_prompt or SHARED_SEED_PROMPT] * n_worst
        data = self.load()
        groups = data.setdefault("groups", {})
        prev_learnings = list((groups.get(group) or {}).get("learnings") or [])
        groups[group] = {
            "prompts": next_pool,
            "learnings": (prev_learnings + list(learnings or []))[-24:],
            "updated_at": _now_iso(),
        }
        self.save(data)
        return next_pool


def critic_score_candidates(
    *,
    llm: Any,
    skill_name: str,
    incumbent_content: str,
    candidate_contents: list[str],
    episodes: list[dict[str, Any]],
    cfg: SimGuardConfig,
) -> tuple[list[dict[str, float]], list[list[dict[str, float]]]]:
    """Score incumbent and M candidates jointly using GUARD's four-criterion critic (Box B.1c).

    Falls back to deterministic size/hygiene/trigger scoring when ``llm`` is unavailable.
    """
    if not episodes:
        episodes = [{"episode": 0, "summary": "baseline skill usage", "score": 75.0, "strict": 50.0}]

    if llm is not None and cfg.enable_llm_review:
        try:
            payload = json.dumps({
                "skill_name": skill_name,
                "playbook_max_chars": cfg.playbook_max_chars,
                "incumbent": incumbent_content[: cfg.playbook_max_chars + 400],
                "candidates": [
                    {"index": idx, "content": text[: cfg.playbook_max_chars + 400]}
                    for idx, text in enumerate(candidate_contents)
                ],
                "episodes": [
                    {
                        "episode": i,
                        "score": ep.get("score", 75.0),
                        "strict": ep.get("strict_score", 50.0),
                        "verdicts": (ep.get("verdicts") or [])[:5],
                    }
                    for i, ep in enumerate(episodes[:8])
                ],
            }, ensure_ascii=False)
            res = llm.complete_structured(
                instructions=_CRITIC_INSTRUCTIONS,
                input=[{"type": "text", "text": payload}],
                json_mode=True,
                temperature=0.0,
                max_tokens=1024,
                timeout=30.0,
                task=AUX_TASK_REVIEWER,
                purpose="guard_four_criterion_critic",
            )
            parsed = getattr(res, "parsed", None)
            if isinstance(parsed, dict) and isinstance(parsed.get("incumbent_scores"), list) and isinstance(parsed.get("candidates"), list):
                inc_scores = [s for s in parsed["incumbent_scores"] if isinstance(s, dict)]
                by_idx = {
                    int(c.get("index", i)): [s for s in (c.get("scores") or []) if isinstance(s, dict)]
                    for i, c in enumerate(parsed["candidates"])
                    if isinstance(c, dict)
                }
                if inc_scores and all(i in by_idx and by_idx[i] for i in range(len(candidate_contents))):
                    return inc_scores, [by_idx[i] for i in range(len(candidate_contents))]
        except Exception as exc:
            logger.debug("GUARD critic pass fell back to deterministic heuristic: %s", exc)

    # Deterministic fallback: checks non-empty body, budget compliance B_W, and [iff] scoping.
    inc_scores = [
        {"episode": i, "score": float(ep.get("score", 75.0)), "strict": float(ep.get("strict_score", 50.0))}
        for i, ep in enumerate(episodes[:8])
    ]
    cand_lists: list[list[dict[str, float]]] = []
    for text in candidate_contents:
        body = (text or "").strip()
        scoped = "[iff" in body.lower() or "when " in body.lower()
        within_budget = 0 < len(body) <= cfg.playbook_max_chars
        delta = (4.0 if scoped else -3.0) + (2.0 if within_budget else -6.0)
        if body == (incumbent_content or "").strip():
            delta = 0.0
        scored = []
        for row in inc_scores:
            ns = max(0.0, min(100.0, row["score"] + delta))
            nstrict = 100.0 if ns >= 99.5 else (ns / 2.0 if ns > 0 else 0.0)
            scored.append({"episode": row["episode"], "score": round(ns, 2), "strict": round(nstrict, 2)})
        cand_lists.append(scored)
    return inc_scores, cand_lists


def run_override_evolution_round(
    *,
    llm: Any,
    skill_name: str,
    incumbent_content: str,
    regret_prompt_block: str,
    sim_prompt_block: str,
    episodes: list[dict[str, Any]],
    strategy_pool: StrategyPoolStore,
    rejection_ledger: RejectionLedger,
    cfg: SimGuardConfig,
    round_num: int,
) -> dict[str, Any]:
    """Execute one GUARD Algorithm 1 Propose + ContrastEvolve + Soft-HURT Gate round."""
    prompts = strategy_pool.get_prompts(skill_name, cfg.pool_size)
    raw_candidates: list[dict[str, Any]] = []
    for idx, strat_prompt in enumerate(prompts):
        cand = _distill_candidate(
            llm=llm,
            strategy_prompt=strat_prompt,
            skill_name=skill_name,
            incumbent_content=incumbent_content,
            regret_block=regret_prompt_block,
            sim_block=sim_prompt_block,
            max_chars=cfg.playbook_max_chars,
            temperature=0.65 if idx == 0 else 0.8,
        )
        if cand is not None:
            raw_candidates.append({"strategy_prompt": strat_prompt, **cand})

    if not raw_candidates:
        return {"promoted": False, "reason": "no_candidates_generated", "candidates": []}

    inc_scores, cand_score_lists = critic_score_candidates(
        llm=llm,
        skill_name=skill_name,
        incumbent_content=incumbent_content,
        candidate_contents=[c["skill_content"] for c in raw_candidates],
        episodes=episodes,
        cfg=cfg,
    )

    evaluated: list[CandidateEvaluation] = []
    for idx, (cand, c_scores) in enumerate(zip(raw_candidates, cand_score_lists)):
        m = evaluate_candidate_metrics(
            incumbent_scores=inc_scores,
            candidate_scores=c_scores,
            delta_noise=cfg.delta_noise,
            beta_hurt=cfg.beta_hurt,
            gamma_help=cfg.gamma_help,
        )
        evaluated.append(CandidateEvaluation(
            index=idx,
            strategy_prompt=cand["strategy_prompt"],
            root_cause=str(cand.get("root_cause") or "skill refinement"),
            problem_id=str(cand.get("problem_id") or ""),
            what=str(cand.get("what") or "GUARD candidate edit"),
            predicted=tuple(str(p) for p in (cand.get("predicted") or []) if p),
            skill_content=str(cand.get("skill_content") or ""),
            mean_score=m["mean_score"],
            mean_strict=m["mean_strict"],
            incumbent_mean_score=m["incumbent_mean_score"],
            helped_episodes=tuple(m["helped"]),
            hurt_episodes=tuple(m["hurt"]),
            hurt_penalty=m["hurt_penalty"],
            net_gain=m["net_gain"],
            fitness=m["fitness"],
        ))

    evaluated = standardize_advantages(evaluated)
    ranked = sorted(
        evaluated,
        key=lambda c: (-c.fitness, -c.mean_score, len(c.hurt_episodes), c.index),
    )
    winner = ranked[0]

    gate = check_soft_hurt_gate(
        net_gain=winner.net_gain,
        hurt_penalty=winner.hurt_penalty,
        helped_count=len(winner.helped_episodes),
        hurt_count=len(winner.hurt_episodes),
        h_max=cfg.h_max,
        mu_min=cfg.mu_min,
        is_repair=False,
    )

    final_content = winner.skill_content
    final_what = winner.what

    # In-round [iff] scope-guard repair when 1 <= |Q-| <= h_rep (§4.4)
    if not gate.promoted and 1 <= len(winner.hurt_episodes) <= cfg.h_rep:
        repaired = _attempt_iff_repair(
            llm=llm,
            skill_name=skill_name,
            candidate_content=winner.skill_content,
            hurt_episodes=list(winner.hurt_episodes),
            episodes=episodes,
            max_chars=cfg.playbook_max_chars,
        )
        if repaired is not None:
            r_inc, r_cands = critic_score_candidates(
                llm=llm,
                skill_name=skill_name,
                incumbent_content=incumbent_content,
                candidate_contents=[repaired["skill_content"]],
                episodes=episodes,
                cfg=cfg,
            )
            rm = evaluate_candidate_metrics(
                incumbent_scores=r_inc,
                candidate_scores=r_cands[0],
                delta_noise=cfg.delta_noise,
                beta_hurt=cfg.beta_hurt,
                gamma_help=cfg.gamma_help,
            )
            rep_gate = check_soft_hurt_gate(
                net_gain=rm["net_gain"],
                hurt_penalty=rm["hurt_penalty"],
                helped_count=len(rm["helped"]),
                hurt_count=len(rm["hurt"]),
                h_max=cfg.h_max,
                mu_min=cfg.mu_min,
                is_repair=True,
            )
            if rep_gate.promoted:
                gate = rep_gate
                final_content = repaired["skill_content"]
                final_what = f"{winner.what} [repaired: {repaired['what']}]"

    if not gate.promoted:
        rejection_ledger.record(
            round_num=round_num,
            skill=skill_name,
            reason=gate.reason,
            net_gain=gate.net_gain,
            hurt=gate.hurt_penalty,
            excerpt=winner.what or winner.skill_content[:300],
            helped_count=gate.helped_count,
            hurt_count=gate.hurt_count,
        )

    # ContrastEvolve (Eq. (9)-(10)): contrast distinct best m+ and worst m-
    if len(ranked) >= 2:
        best_c, worst_c = ranked[0], ranked[-1]
        _run_contrast_evolve(
            llm=llm,
            group=skill_name,
            best=best_c,
            worst=worst_c,
            strategy_pool=strategy_pool,
            pool_size=cfg.pool_size,
        )

    return {
        "promoted": gate.promoted,
        "repaired": gate.repaired,
        "reason": gate.reason,
        "gate": gate,
        "winner": winner,
        "final_content": final_content,
        "final_what": final_what,
        "candidates": evaluated,
    }


def _distill_candidate(
    *,
    llm: Any,
    strategy_prompt: str,
    skill_name: str,
    incumbent_content: str,
    regret_block: str,
    sim_block: str,
    max_chars: int,
    temperature: float,
) -> Optional[dict[str, Any]]:
    if llm is None:
        return None
    try:
        user_text = (
            f"### DISTILLER STRATEGY PROMPT\n{strategy_prompt}\n\n"
            f"### TARGET SKILL: {skill_name}\n### INCUMBENT SKILL.md\n{incumbent_content}\n\n"
            f"{regret_block}\n\n{sim_block}"
        )
        res = llm.complete_structured(
            instructions=_DISTILL_INSTRUCTIONS,
            input=[{"type": "text", "text": user_text}],
            json_mode=True,
            temperature=temperature,
            max_tokens=2048,
            timeout=35.0,
            task=AUX_TASK_EVOLVER,
            purpose="guard_distill_candidate",
        )
        parsed = getattr(res, "parsed", None)
        if not isinstance(parsed, dict):
            return None
        content = str(parsed.get("skill_content") or "").strip()
        if not content:
            return None
        if len(content) > max_chars + 1200:
            content = content[: max_chars + 1200]
        return {
            "root_cause": str(parsed.get("root_cause") or "skill refinement"),
            "problem_id": str(parsed.get("problem_id") or ""),
            "what": str(parsed.get("what") or "distilled refinement"),
            "predicted": list(parsed.get("predicted") or []),
            "skill_content": content,
        }
    except Exception as exc:
        logger.debug("GUARD distill candidate failed: %s", exc)
        return None


def _attempt_iff_repair(
    *,
    llm: Any,
    skill_name: str,
    candidate_content: str,
    hurt_episodes: list[int],
    episodes: list[dict[str, Any]],
    max_chars: int,
) -> Optional[dict[str, str]]:
    if llm is None:
        return None
    try:
        hurt_details = [episodes[i] for i in hurt_episodes if 0 <= i < len(episodes)]
        payload = json.dumps({
            "skill_name": skill_name,
            "playbook_max_chars": max_chars,
            "candidate_skill": candidate_content[: max_chars + 600],
            "hurt_episodes": hurt_details,
        }, ensure_ascii=False)
        res = llm.complete_structured(
            instructions=_IFF_REPAIR_INSTRUCTIONS,
            input=[{"type": "text", "text": payload}],
            json_mode=True,
            temperature=0.25,
            max_tokens=2048,
            timeout=30.0,
            task=AUX_TASK_EVOLVER,
            purpose="guard_iff_repair",
        )
        parsed = getattr(res, "parsed", None)
        if isinstance(parsed, dict) and str(parsed.get("skill_content") or "").strip():
            return {
                "what": str(parsed.get("what") or "narrowed [iff] guard"),
                "skill_content": str(parsed["skill_content"]).strip(),
            }
    except Exception as exc:
        logger.debug("GUARD [iff] repair failed: %s", exc)
    return None


def _run_contrast_evolve(
    *,
    llm: Any,
    group: str,
    best: CandidateEvaluation,
    worst: CandidateEvaluation,
    strategy_pool: StrategyPoolStore,
    pool_size: int,
) -> None:
    if llm is None:
        return
    try:
        payload = json.dumps({
            "best": {
                "strategy_prompt": scrub_ids(best.strategy_prompt),
                "excerpt": scrub_ids(best.skill_content[:800]),
                "R_m": best.fitness,
                "helped": len(best.helped_episodes),
                "hurt": len(best.hurt_episodes),
            },
            "worst": {
                "strategy_prompt": scrub_ids(worst.strategy_prompt),
                "excerpt": scrub_ids(worst.skill_content[:800]),
                "R_m": worst.fitness,
                "helped": len(worst.helped_episodes),
                "hurt": len(worst.hurt_episodes),
            },
        }, ensure_ascii=False)
        res = llm.complete_structured(
            instructions=_CONTRAST_EVOLVE_INSTRUCTIONS,
            input=[{"type": "text", "text": payload}],
            json_mode=True,
            temperature=0.2,
            max_tokens=1024,
            timeout=25.0,
            task=AUX_TASK_EVOLVER,
            purpose="guard_contrast_evolve",
        )
        parsed = getattr(res, "parsed", None)
        if isinstance(parsed, dict):
            s_plus = str(parsed.get("updated_best_prompt") or "").strip()
            s_minus = str(parsed.get("updated_worst_prompt") or "").strip()
            learnings = [str(x) for x in (parsed.get("learnings") or []) if isinstance(x, str)]
            if s_plus and s_minus:
                strategy_pool.update_prompts(
                    group, best_prompt=s_plus, worst_prompt=s_minus, learnings=learnings, pool_size=pool_size,
                )
    except Exception as exc:
        logger.debug("GUARD ContrastEvolve failed: %s", exc)


def rollback_skill_via_ledger(skill_name: str, *, entry_id: Optional[str] = None) -> tuple[bool, str]:
    """Roll back the latest (or specified) mutation of ``skill_name`` via Hermes's ``tools.skill_ledger``."""
    try:
        from tools import skill_ledger

        target_id = entry_id
        if not target_id:
            entries = skill_ledger.list_entries(skill=skill_name, limit=5)
            if not entries:
                return False, f"no skill_ledger entry found for '{skill_name}'"
            target_id = str(entries[0].get("id") or "")
        if not target_id:
            return False, f"missing entry id for '{skill_name}'"
        return skill_ledger.rollback_entry(target_id)
    except Exception as exc:
        return False, f"rollback unavailable: {exc}"
