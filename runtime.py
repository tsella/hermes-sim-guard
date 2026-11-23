"""Stateful runtime coordinator wiring SIM + GUARD + Telemetry onto Hermes lifecycle hooks.

Profile-safe: every disk path is resolved dynamically from ``ctx.state.data_dir`` on each call,
never cached at ``register()`` time, so multiplexed multi-profile gateways stay isolated.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from .config import SimGuardConfig
    from .guard_checklist import ChecklistCriterion, grade_turn, summarize_tool_event
    from .guard_propose_gate import (
        RejectionLedger,
        StrategyPoolStore,
        check_soft_hurt_gate,
        critic_score_candidates,
        evaluate_candidate_metrics,
        rollback_skill_via_ledger,
        run_override_evolution_round,
    )
    from .guard_regret import build_regret_bundle
    from .sim_memory import SimMemoryStore
    from .telemetry import TelemetryStore
except ImportError:
    from config import SimGuardConfig
    from guard_checklist import ChecklistCriterion, grade_turn, summarize_tool_event
    from guard_propose_gate import (
        RejectionLedger,
        StrategyPoolStore,
        check_soft_hurt_gate,
        critic_score_candidates,
        evaluate_candidate_metrics,
        rollback_skill_via_ledger,
        run_override_evolution_round,
    )
    from guard_regret import build_regret_bundle
    from sim_memory import SimMemoryStore
    from telemetry import TelemetryStore

logger = logging.getLogger("hermes_sim_guard")


@dataclass
class _TurnAccumulator:
    """Ephemeral per-session accumulator for the turn currently in flight."""

    user_message: str = ""
    model: str = ""
    platform: str = ""
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)
    pre_checklist: list[ChecklistCriterion] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    api_calls: int = 0


def _read_skill_text(skill_name: str) -> Optional[str]:
    """Read a local skill's SKILL.md via Hermes's documented skill lookup API."""
    try:
        from tools.skills_tool import locate_skill, skill_search_dirs

        roots, _active_dir = skill_search_dirs()
        err, _skill_dir, skill_md = locate_skill(skill_name, None, roots)
        if err or skill_md is None:
            return None
        path = Path(skill_md)
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except Exception:
        return None


def _latest_ledger_before_text(skill_name: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(entry_id, before_skill_md_text)`` from the newest ``skill_ledger`` entry."""
    try:
        from tools import skill_ledger

        entries = skill_ledger.list_entries(skill=skill_name, limit=3)
        if not entries:
            return None, None
        entry = entries[0]
        entry_id = str(entry.get("id") or "") or None
        for item in entry.get("before") or []:
            if isinstance(item, dict) and str(item.get("path", "")).endswith("SKILL.md"):
                raw = skill_ledger.read_blob(str(item.get("sha256") or ""))
                if raw is not None:
                    return entry_id, raw.decode("utf-8", errors="replace")
        return entry_id, None
    except Exception:
        return None, None


class SimGuardRuntime:
    """Coordinates SIM memory, GUARD regret/propose/gate, and Pre/Post telemetry."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._lock = threading.Lock()
        self._turns: dict[str, _TurnAccumulator] = {}
        self._pending_review_snapshots: dict[str, dict[str, str]] = {}

    def _cfg(self) -> SimGuardConfig:
        return SimGuardConfig.from_ctx(self.ctx)

    def _stores(self) -> tuple[SimMemoryStore, TelemetryStore, RejectionLedger, StrategyPoolStore]:
        data_dir = Path(self.ctx.state.data_dir)
        return (
            SimMemoryStore(data_dir / "sim_memory.json"),
            TelemetryStore(data_dir / "telemetry.jsonl"),
            RejectionLedger(data_dir / "rejection_ledger.jsonl"),
            StrategyPoolStore(data_dir / "strategy_pool.json"),
        )

    def _turn_key(self, session_id: Optional[str], task_id: Optional[str] = None) -> str:
        return str(session_id or task_id or "default")

    def _get_or_create_turn(self, key: str) -> _TurnAccumulator:
        with self._lock:
            acc = self._turns.get(key)
            if acc is None:
                acc = _TurnAccumulator()
                self._turns[key] = acc
            return acc

    # --- Turn lifecycle hooks -------------------------------------------------

    def on_pre_llm_call(
        self,
        *,
        session_id: str = "",
        user_message: str = "",
        model: str = "",
        platform: str = "",
        **_kwargs: Any,
    ) -> None:
        key = self._turn_key(session_id)
        with self._lock:
            self._turns[key] = _TurnAccumulator(
                user_message=str(user_message or ""),
                model=str(model or ""),
                platform=str(platform or ""),
            )

    def on_post_tool_call(
        self,
        *,
        tool_name: str = "",
        args: Optional[dict[str, Any]] = None,
        result: str = "",
        task_id: str = "",
        session_id: str = "",
        duration_ms: int = 0,
        **_kwargs: Any,
    ) -> None:
        key = self._turn_key(session_id, task_id)
        acc = self._get_or_create_turn(key)
        with self._lock:
            acc.tool_trace.append(summarize_tool_event(tool_name, result, duration_ms))
            if tool_name == "skill_view" and isinstance(args, dict):
                sname = str(args.get("name") or "").strip()
                if sname and sname not in acc.skills_loaded:
                    acc.skills_loaded.append(sname)

    def on_post_api_request(
        self,
        *,
        session_id: str = "",
        task_id: str = "",
        usage: Optional[dict[str, Any]] = None,
        model: str = "",
        **_kwargs: Any,
    ) -> None:
        key = self._turn_key(session_id, task_id)
        acc = self._get_or_create_turn(key)
        u = usage if isinstance(usage, dict) else {}
        with self._lock:
            acc.api_calls += 1
            acc.input_tokens += int(u.get("input_tokens") or u.get("prompt_tokens") or 0)
            acc.output_tokens += int(u.get("output_tokens") or u.get("completion_tokens") or 0)
            acc.cache_read_tokens += int(u.get("cache_read_tokens") or u.get("cache_read_input_tokens") or 0)
            if model and not acc.model:
                acc.model = str(model)

    def on_post_llm_call(
        self,
        *,
        session_id: str = "",
        task_id: str = "",
        user_message: str = "",
        assistant_response: str = "",
        model: str = "",
        platform: str = "",
        **_kwargs: Any,
    ) -> None:
        cfg = self._cfg()
        key = self._turn_key(session_id, task_id)
        with self._lock:
            acc = self._turns.pop(key, None) or _TurnAccumulator()

        eff_user = acc.user_message or str(user_message or "")
        eff_model = str(model or acc.model or "")
        eff_platform = str(platform or acc.platform or "cli")
        tool_count = len(acc.tool_trace)
        skills_loaded = list(acc.skills_loaded)

        # Gate LLM checklist + Layer-2 self-review to non-trivial turns (tool calls or loaded skills).
        eligible_for_llm = (
            cfg.enable_llm_review
            and (tool_count >= cfg.checklist_min_tool_calls or bool(skills_loaded))
        )
        grade = grade_turn(
            user_message=eff_user,
            assistant_response=str(assistant_response or ""),
            tool_trace=acc.tool_trace,
            skills_loaded=skills_loaded,
            pre_checklist=acc.pre_checklist or None,
            failed=False,
            interrupted=False,
            llm=getattr(self.ctx, "llm", None) if eligible_for_llm else None,
            enable_llm=eligible_for_llm,
        )

        sim_store, telemetry, _, _ = self._stores()
        recent_turns = telemetry.read_events(event_type="turn_graded", limit=cfg.attribution_window * 2)
        prev_window = recent_turns[-2 * cfg.attribution_window : -cfg.attribution_window] if len(recent_turns) >= 2 else []
        curr_window = recent_turns[-cfg.attribution_window :] + [grade.to_dict()]
        bundle = build_regret_bundle(
            current_turns=curr_window,
            previous_turns=prev_window,
            delta_con=cfg.delta_contrast,
        )

        total_tokens = acc.input_tokens + acc.output_tokens
        # Hermes reports input_tokens as NON-cached prompt tokens, so the cached share of the
        # prompt is cache_read / (input + cache_read). Dividing by input alone reports ratios
        # far above 1 (observed up to 872x).
        prompt_total = acc.input_tokens + acc.cache_read_tokens
        cache_ratio = (acc.cache_read_tokens / prompt_total) if prompt_total > 0 else None
        telemetry.append(
            "turn_graded",
            {
                "session_id": str(session_id or ""),
                "model": eff_model,
                "platform": eff_platform,
                "skills_loaded": skills_loaded,
                "tool_iterations": tool_count,
                "api_calls": acc.api_calls,
                "total_tokens": total_tokens if total_tokens > 0 else None,
                "input_tokens": acc.input_tokens,
                "output_tokens": acc.output_tokens,
                "cache_read_tokens": acc.cache_read_tokens,
                "cache_read_ratio": round(cache_ratio, 4) if cache_ratio is not None else None,
                "mean_delta_hurt": round(bundle.mean_delta_hurt, 4),
                "mean_delta_gap": round(bundle.mean_delta_gap, 4),
                **grade.to_dict(),
            },
        )
        # Backfill SIM predicted outcomes from this live turn's checklist verdicts.
        sim_store.backfill_outcomes(
            skills_loaded=skills_loaded,
            passed_criteria=grade.passed_ids,
            failed_criteria=grade.failed_ids,
            model=eff_model,
            domain=eff_platform,
        )

    def on_session_end(
        self,
        *,
        session_id: str = "",
        task_id: str = "",
        failed: bool = False,
        interrupted: bool = False,
        model: str = "",
        platform: str = "",
        **_kwargs: Any,
    ) -> None:
        # Record failed/interrupted turns that never fired post_llm_call.
        key = self._turn_key(session_id, task_id)
        with self._lock:
            acc = self._turns.pop(key, None)
        if acc is None or (not failed and not interrupted):
            return
        _, telemetry, _, _ = self._stores()
        grade = grade_turn(
            user_message=acc.user_message,
            assistant_response="",
            tool_trace=acc.tool_trace,
            skills_loaded=acc.skills_loaded,
            failed=bool(failed),
            interrupted=bool(interrupted),
            llm=None,
            enable_llm=False,
        )
        total_tokens = acc.input_tokens + acc.output_tokens
        telemetry.append(
            "turn_graded",
            {
                "session_id": str(session_id or ""),
                "model": str(model or acc.model or ""),
                "platform": str(platform or acc.platform or "cli"),
                "skills_loaded": list(acc.skills_loaded),
                "tool_iterations": len(acc.tool_trace),
                "api_calls": acc.api_calls,
                "total_tokens": total_tokens if total_tokens > 0 else None,
                "failed": bool(failed),
                "interrupted": bool(interrupted),
                **grade.to_dict(),
            },
        )

    def on_skill_lifecycle(
        self,
        *,
        action: str = "",
        skill_name: str = "",
        provenance: str = "",
        use_count: Optional[int] = None,
        reused: Optional[bool] = None,
        reuse_after_patch: Optional[bool] = None,
        session_id: str = "",
        **_kwargs: Any,
    ) -> None:
        _, telemetry, _, _ = self._stores()
        telemetry.append(
            "skill_lifecycle",
            {
                "action": action,
                "skill_name": skill_name,
                "provenance": provenance,
                "use_count": use_count,
                "reused": reused,
                "reuse_after_patch": reuse_after_patch,
                "session_id": session_id,
            },
        )

    # --- Background review lifecycle hooks (SIM + GUARD) ----------------------

    def _compose_sim_guard_prompt_context(
        self,
        *,
        model: str,
        provider: str,
        domain: str,
        skills_hint: Optional[list[str]] = None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        cfg = self._cfg()
        sim_store, telemetry, rejection_ledger, _ = self._stores()
        turns = telemetry.read_events(event_type="turn_graded", limit=cfg.attribution_window * 2)
        if skills_hint:
            hint_set = set(skills_hint)
            scoped = [t for t in turns if set(t.get("skills_loaded") or []) & hint_set]
            if scoped:
                turns = scoped
        mid = max(1, len(turns) // 2) if len(turns) >= 2 else 0
        prev_turns = turns[:mid] if mid else []
        curr_turns = turns[mid:] if mid else turns

        bundle = build_regret_bundle(
            current_turns=curr_turns,
            previous_turns=prev_turns,
            delta_con=cfg.delta_contrast,
        )
        rejections = rejection_ledger.read_recent(limit=cfg.rejection_ledger_window)
        sim_block = sim_store.render_for_prompt(
            current_model=model,
            current_provider=provider,
            current_domain=domain,
        )
        regret_block = bundle.to_prompt_block(
            rejection_entries=rejections,
            playbook_max_chars=cfg.playbook_max_chars,
        )
        return sim_block, regret_block, curr_turns

    def on_pre_background_review(
        self,
        *,
        session_id: str = "",
        review_memory: bool = False,
        review_skills: bool = False,
        model: str = "",
        provider: str = "",
        skills_loaded: Optional[list[str]] = None,
        **_kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        if not review_skills:
            return None

        cfg = self._cfg()
        loaded = list(skills_loaded or [])
        sim_block, regret_block, recent_turns = self._compose_sim_guard_prompt_context(
            model=model, provider=provider, domain="background_review", skills_hint=loaded,
        )

        # Snapshot current loaded skills before review for post-hoc Soft-HURT diffing.
        snapshots = {name: text for name in loaded if (text := _read_skill_text(name)) is not None}
        if snapshots:
            with self._lock:
                self._pending_review_snapshots[str(session_id or "default")] = snapshots

        # Override mode: skip Hermes's built-in single-pass skill fork when memory review isn't
        # co-requested and run GUARD's full M>=2 ContrastEvolve + Soft-HURT pipeline instead.
        if cfg.mode == "override" and not review_memory and loaded:
            self._run_override_skill_pass(
                session_id=session_id,
                skills=loaded,
                model=model,
                provider=provider,
                sim_block=sim_block,
                regret_block=regret_block,
                episodes=recent_turns,
                cfg=cfg,
            )
            return {"action": "skip", "reason": "hermes-sim-guard override mode"}

        # Default 'augment+gate' mode: inject SIM + GUARD bundle into warm-cache review prompt.
        sections = [s for s in (sim_block, regret_block) if s]
        sections.append(
            "SIM+GUARD output contract: whenever you patch or create a skill, wrap new narrow fixes "
            "in explicit [iff <precondition>] triggers, keep pinned Mastered invariants intact, and "
            "state the root cause and predicted criteria fixed in your final summary."
        )
        return {"append_prompt": "\n\n".join(sections)}

    def _run_override_skill_pass(
        self,
        *,
        session_id: str,
        skills: list[str],
        model: str,
        provider: str,
        sim_block: str,
        regret_block: str,
        episodes: list[dict[str, Any]],
        cfg: SimGuardConfig,
    ) -> None:
        sim_store, telemetry, rejection_ledger, strategy_pool = self._stores()
        llm = getattr(self.ctx, "llm", None)
        sim_data = sim_store.load()
        next_round = int(sim_data.get("round", 0)) + 1

        for skill_name in skills[:2]:
            incumbent = _read_skill_text(skill_name)
            if incumbent is None:
                continue
            outcome = run_override_evolution_round(
                llm=llm,
                skill_name=skill_name,
                incumbent_content=incumbent,
                regret_prompt_block=regret_block,
                sim_prompt_block=sim_block,
                episodes=episodes,
                strategy_pool=strategy_pool,
                rejection_ledger=rejection_ledger,
                cfg=cfg,
                round_num=next_round,
            )
            gate = outcome.get("gate")
            winner = outcome.get("winner")
            if gate is None or winner is None:
                continue

            telemetry.append(
                "gate_decision",
                {
                    "session_id": session_id,
                    "mode": "override",
                    "skill": skill_name,
                    "round": next_round,
                    "promoted": gate.promoted,
                    "repaired": gate.repaired,
                    "rolled_back": False,
                    "reason": gate.reason,
                    "net_gain": gate.net_gain,
                    "hurt_penalty": gate.hurt_penalty,
                    "helped_count": gate.helped_count,
                    "hurt_count": gate.hurt_count,
                },
            )
            recorded = sim_store.record_attempt(
                root_cause=winner.root_cause,
                problem_id=winner.problem_id or None,
                what=outcome.get("final_what") or winner.what,
                skills=[skill_name],
                predicted=list(winner.predicted),
                accepted=gate.promoted,
                sample="override",
                domain="background_review",
                model=model,
                provider=provider,
            )
            if gate.promoted and outcome.get("final_content"):
                self._apply_skill_edit(skill_name, str(outcome["final_content"]))
                telemetry.append(
                    "skill_edit_epoch",
                    {
                        "session_id": session_id,
                        "mode": "override",
                        "round": recorded["round"],
                        "problem_id": recorded["problem_id"],
                        "skills": [skill_name],
                        "skill_chars": len(str(outcome["final_content"])),
                        "model": model,
                        "provider": provider,
                    },
                )

    def _apply_skill_edit(self, skill_name: str, new_content: str) -> bool:
        """Apply a gated skill rewrite via Hermes's ``skill_manage`` tool (preserving ledger/guards)."""
        try:
            raw = self.ctx.dispatch_tool(
                "skill_manage",
                {"action": "edit", "name": skill_name, "content": new_content},
            )
            data = json.loads(raw) if isinstance(raw, str) else raw
            return bool(isinstance(data, dict) and data.get("success"))
        except Exception as exc:
            logger.debug("GUARD override skill_manage dispatch failed for %s: %s", skill_name, exc)
            return False

    def on_post_background_review(
        self,
        *,
        session_id: str = "",
        review_skills: bool = False,
        outcome: str = "ok",
        skills_touched: Optional[list[str]] = None,
        actions: Optional[list[str]] = None,
        model: str = "",
        provider: str = "",
        **_kwargs: Any,
    ) -> None:
        with self._lock:
            snapshots = self._pending_review_snapshots.pop(str(session_id or "default"), {})
        if not review_skills or outcome != "ok" or not skills_touched:
            return

        cfg = self._cfg()
        sim_store, telemetry, rejection_ledger, _ = self._stores()
        recent_turns = telemetry.read_events(event_type="turn_graded", limit=cfg.attribution_window)
        llm = getattr(self.ctx, "llm", None)

        for skill_name in skills_touched:
            after_text = _read_skill_text(skill_name)
            entry_id, ledger_before = _latest_ledger_before_text(skill_name)
            before_text = ledger_before if ledger_before is not None else snapshots.get(skill_name, "")
            if after_text is None:
                continue

            episodes = [t for t in recent_turns if skill_name in (t.get("skills_loaded") or [])] or recent_turns[-6:]
            inc_scores, cand_lists = critic_score_candidates(
                llm=llm,
                skill_name=skill_name,
                incumbent_content=before_text,
                candidate_contents=[after_text],
                episodes=episodes,
                cfg=cfg,
            )
            metrics = evaluate_candidate_metrics(
                incumbent_scores=inc_scores,
                candidate_scores=cand_lists[0],
                delta_noise=cfg.delta_noise,
                beta_hurt=cfg.beta_hurt,
                gamma_help=cfg.gamma_help,
            )
            gate = check_soft_hurt_gate(
                net_gain=metrics["net_gain"],
                hurt_penalty=metrics["hurt_penalty"],
                helped_count=len(metrics["helped"]),
                hurt_count=len(metrics["hurt"]),
                h_max=cfg.h_max,
                mu_min=cfg.mu_min,
                is_repair=False,
            )

            rolled_back = False
            if not gate.promoted and cfg.auto_rollback and entry_id:
                ok_rb, rb_msg = rollback_skill_via_ledger(skill_name, entry_id=entry_id)
                rolled_back = bool(ok_rb)
                if not ok_rb:
                    logger.warning("Soft-HURT rollback for %s failed: %s", skill_name, rb_msg)

            what = "; ".join(actions or []) or f"background review update to {skill_name}"
            recorded = sim_store.record_attempt(
                root_cause=f"skill:{skill_name} workflow refinement",
                what=what,
                skills=[skill_name],
                predicted=[v.get("id") for ep in episodes[-2:] for v in (ep.get("verdicts") or []) if not v.get("passed")][:5],
                accepted=gate.promoted,
                sample="augment+gate",
                domain="background_review",
                model=model,
                provider=provider,
                ledger_entry_ids=[entry_id] if entry_id else [],
            )

            if not gate.promoted:
                rejection_ledger.record(
                    round_num=recorded["round"],
                    skill=skill_name,
                    reason=gate.reason,
                    net_gain=gate.net_gain,
                    hurt=gate.hurt_penalty,
                    excerpt=what,
                    helped_count=gate.helped_count,
                    hurt_count=gate.hurt_count,
                )

            telemetry.append(
                "gate_decision",
                {
                    "session_id": session_id,
                    "mode": "augment+gate",
                    "skill": skill_name,
                    "round": recorded["round"],
                    "promoted": gate.promoted,
                    "repaired": False,
                    "rolled_back": rolled_back,
                    "reason": gate.reason,
                    "net_gain": gate.net_gain,
                    "hurt_penalty": gate.hurt_penalty,
                    "helped_count": gate.helped_count,
                    "hurt_count": gate.hurt_count,
                },
            )
            if gate.promoted:
                telemetry.append(
                    "skill_edit_epoch",
                    {
                        "session_id": session_id,
                        "mode": "augment+gate",
                        "round": recorded["round"],
                        "problem_id": recorded["problem_id"],
                        "skills": [skill_name],
                        "skill_chars": len(after_text),
                        "model": model,
                        "provider": provider,
                    },
                )

    # --- Curator lifecycle hooks ----------------------------------------------

    def on_pre_curator_run(
        self,
        *,
        consolidate: bool = False,
        candidates: Optional[list[str]] = None,
        **_kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        if not consolidate or not candidates:
            return None
        sim_block, regret_block, _ = self._compose_sim_guard_prompt_context(
            model="", provider="", domain="curator", skills_hint=list(candidates),
        )
        sections = [s for s in (sim_block, regret_block) if s]
        if not sections:
            return None
        sections.append(
            "SIM+GUARD curator invariant: preserve all pinned Mastered rules and [iff <precondition>] "
            "guards when consolidating skills into umbrellas, and do not re-apply entries from the "
            "active rejection ledger."
        )
        return {"append_prompt": "\n\n".join(sections)}

    def on_post_curator_run(
        self,
        *,
        run_json_path: str = "",
        dry_run: bool = False,
        **_kwargs: Any,
    ) -> None:
        if dry_run or not run_json_path:
            return
        try:
            payload = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        sim_store, telemetry, _, _ = self._stores()
        counts = payload.get("counts") or {}
        consolidated = payload.get("consolidated") or []
        pruned = payload.get("pruned_names") or []
        telemetry.append(
            "curator_run",
            {
                "model": payload.get("model", ""),
                "provider": payload.get("provider", ""),
                "consolidated": len(consolidated),
                "pruned": len(pruned),
                "added": len(payload.get("added") or []),
                "delta": counts.get("delta", 0),
            },
        )
        if consolidated:
            targets = sorted({str(c.get("into") or "") for c in consolidated if isinstance(c, dict) and c.get("into")})
            sources = [str(c.get("name") or "") for c in consolidated if isinstance(c, dict) and c.get("name")]
            sim_store.record_attempt(
                root_cause="curator skill consolidation",
                what=f"Consolidated {', '.join(sources[:6])} -> {', '.join(targets[:4])}",
                skills=targets or sources[:4],
                accepted=True,
                sample="curator",
                domain="curator",
                model=str(payload.get("model") or ""),
                provider=str(payload.get("provider") or ""),
            )
