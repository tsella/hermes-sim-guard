"""Self-Improvement Memory (SIM) — arXiv:2611.18856 (§3.2, §3.3, Appendix D).

Groups skill/harness edits by recurring root-cause hypothesis ``m_i = (c_i, [(a_ij, o_ij, p_ij)]_j)``.
Preserves full attempt histories across model and domain transitions, deterministically merges
identical root causes while maintaining ``aliases``, and tracks predicted vs observed outcomes
for transfer calibration.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

SIM_SCHEMA_VERSION = 1

SIM_GUIDANCE_RULES = (
    "SIM (Self-Improvement Memory) usage guidelines:\n"
    "1. Treat entries as prior experience, with flexibility to choose a different edit.\n"
    "2. Check the domain, platform, and model/provider behind each entry.\n"
    "3. Start with an existing diagnosis when it explains the current failure.\n"
    "4. Reuse the rationale and reassess the edit after a model or domain change.\n"
    "5. Revise, replace, or override a skill edit when new evidence warrants it.\n"
    "6. Revisit a tested direction only when new evidence justifies it.\n"
    "7. Create a new root-cause entry when existing diagnoses do not explain the failure.\n"
    "8. Give task-specific evidence more weight in similar settings and less when both domain and model changed."
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _normalize_root_cause(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
    return cleaned


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class SimMemoryStore:
    """Profile-scoped JSON ledger for Self-Improvement Memory."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._empty()
        if not isinstance(raw, dict):
            return self._empty()
        raw.setdefault("schema_version", SIM_SCHEMA_VERSION)
        raw.setdefault("next_id", 1)
        raw.setdefault("round", 0)
        raw.setdefault("carried_from", None)
        raw.setdefault("aliases", {})
        raw.setdefault("problems", [])
        return raw

    def save(self, data: dict[str, Any]) -> None:
        _atomic_write_json(self.path, data)

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "schema_version": SIM_SCHEMA_VERSION,
            "next_id": 1,
            "round": 0,
            "carried_from": None,
            "aliases": {},
            "problems": [],
        }

    def resolve_problem_id(self, data: dict[str, Any], problem_id: Optional[str]) -> Optional[str]:
        if not problem_id:
            return None
        aliases = data.get("aliases") or {}
        seen: set[str] = set()
        current = str(problem_id).strip()
        while current in aliases and current not in seen:
            seen.add(current)
            current = str(aliases[current])
        return current

    def _find_problem(self, data: dict[str, Any], *, problem_id: Optional[str], root_cause: str) -> Optional[dict[str, Any]]:
        problems = data.get("problems") or []
        resolved_id = self.resolve_problem_id(data, problem_id)
        if resolved_id:
            for entry in problems:
                if entry.get("id") == resolved_id:
                    return entry
        norm = _normalize_root_cause(root_cause)
        if norm:
            for entry in problems:
                if _normalize_root_cause(entry.get("name", "")) == norm:
                    return entry
        return None

    def record_attempt(
        self,
        *,
        root_cause: str,
        what: str,
        skills: list[str],
        files: Optional[list[str]] = None,
        predicted: Optional[list[str]] = None,
        fixed: Optional[list[str]] = None,
        missed: Optional[list[str]] = None,
        passed: int = 0,
        total: int = 0,
        accepted: bool = True,
        sample: str = "augment+gate",
        domain: str = "cli",
        model: str = "",
        provider: str = "",
        problem_id: Optional[str] = None,
        ledger_entry_ids: Optional[list[str]] = None,
        backfilled: bool = False,
    ) -> dict[str, Any]:
        """Append an optimization attempt to an existing root-cause entry or open a new one."""
        data = self.load()
        data["round"] = int(data.get("round", 0)) + 1
        round_num = data["round"]

        # Track cross-model / cross-domain continuation metadata (carried_from).
        last_ctx = data.get("last_context") or {}
        cur_ctx = {"domain": domain or "cli", "model": model or "", "provider": provider or ""}
        if last_ctx and (
            (model and last_ctx.get("model") and last_ctx.get("model") != model)
            or (domain and last_ctx.get("domain") and last_ctx.get("domain") != domain)
        ):
            data["carried_from"] = {
                "from": dict(last_ctx),
                "into": dict(cur_ctx),
                "switched_at_round": round_num,
                "switched_at": _now_iso(),
            }
        data["last_context"] = cur_ctx

        entry = self._find_problem(data, problem_id=problem_id, root_cause=root_cause)
        if entry is None:
            pid = f"P{int(data.get('next_id', 1))}"
            data["next_id"] = int(data.get("next_id", 1)) + 1
            entry = {
                "id": pid,
                "name": (root_cause or "unspecified harness issue").strip(),
                "opened_round": round_num,
                "attempts": [],
            }
            data["problems"].append(entry)

        attempt = {
            "round": round_num,
            "sample": sample,
            "accepted": bool(accepted),
            "backfilled": bool(backfilled),
            "what": (what or "").strip()[:1000],
            "files": list(files or [f"{s}/SKILL.md" for s in skills]),
            "skills": list(dict.fromkeys(s for s in skills if s)),
            "predicted": list(dict.fromkeys(p for p in (predicted or []) if p)),
            "fixed": list(dict.fromkeys(f for f in (fixed or []) if f)),
            "missed": list(dict.fromkeys(m for m in (missed or []) if m)),
            "passed": int(passed),
            "total": int(total),
            "domain": domain or "cli",
            "model": model or "",
            "provider": provider or "",
            "ledger_entry_ids": list(ledger_entry_ids or []),
            "timestamp": _now_iso(),
        }
        entry["attempts"].append(attempt)
        self._deduplicate_problems(data)
        self.save(data)
        return {"problem_id": entry["id"], "round": round_num, "attempt": attempt}

    def _deduplicate_problems(self, data: dict[str, Any]) -> None:
        """Deterministically merge entries with identical normalized root-cause text, recording aliases."""
        by_norm: dict[str, dict[str, Any]] = {}
        merged: list[dict[str, Any]] = []
        aliases = data.setdefault("aliases", {})
        for entry in data.get("problems") or []:
            norm = _normalize_root_cause(entry.get("name", ""))
            if norm and norm in by_norm:
                canonical = by_norm[norm]
                canonical["attempts"].extend(entry.get("attempts") or [])
                aliases[entry["id"]] = canonical["id"]
            else:
                if norm:
                    by_norm[norm] = entry
                merged.append(entry)
        data["problems"] = merged

    def backfill_outcomes(
        self,
        *,
        skills_loaded: list[str],
        passed_criteria: list[str],
        failed_criteria: list[str],
        model: str = "",
        domain: str = "",
    ) -> int:
        """Resolve open predicted outcomes on the latest accepted attempts touching ``skills_loaded``."""
        if not skills_loaded and not passed_criteria and not failed_criteria:
            return 0
        data = self.load()
        skill_set = set(skills_loaded)
        passed_set = set(passed_criteria)
        failed_set = set(failed_criteria)
        updated = 0
        for problem in data.get("problems") or []:
            attempts = problem.get("attempts") or []
            if not attempts:
                continue
            latest = attempts[-1]
            if not latest.get("accepted"):
                continue
            attempt_skills = set(latest.get("skills") or [])
            if skill_set and attempt_skills and not (skill_set & attempt_skills):
                continue
            predicted = latest.get("predicted") or []
            if not predicted:
                continue
            fixed = set(latest.get("fixed") or [])
            missed = set(latest.get("missed") or [])
            changed = False
            for pred in predicted:
                if pred in passed_set and pred not in fixed:
                    fixed.add(pred)
                    missed.discard(pred)
                    changed = True
                elif pred in failed_set and pred not in missed and pred not in fixed:
                    missed.add(pred)
                    changed = True
            if changed:
                latest["fixed"] = sorted(fixed)
                latest["missed"] = sorted(missed)
                latest["passed"] = int(latest.get("passed", 0)) + len(passed_set)
                latest["total"] = int(latest.get("total", 0)) + len(passed_set) + len(failed_set)
                latest["backfilled"] = True
                latest["observed_context"] = {"model": model, "domain": domain}
                updated += 1
        if updated:
            self.save(data)
        return updated

    def render_for_prompt(
        self,
        *,
        current_model: str = "",
        current_provider: str = "",
        current_domain: str = "",
        max_chars: int = 6000,
    ) -> str:
        """Render the complete SIM ledger for proposer/selector/curator prompts (§3.2, App. D.2)."""
        data = self.load()
        problems = data.get("problems") or []
        if not problems:
            return ""
        lines = ["### SELF-IMPROVEMENT MEMORY (SIM — arXiv:2611.18856)", SIM_GUIDANCE_RULES]
        if carried := data.get("carried_from"):
            src = carried.get("from") or {}
            dst = carried.get("into") or {}
            lines.append(
                f"Context transfer detected at round {carried.get('switched_at_round')}: "
                f"{src.get('provider','')}/{src.get('model','')} ({src.get('domain','')}) -> "
                f"{dst.get('provider','')}/{dst.get('model','')} ({dst.get('domain','')})."
            )
        if current_model or current_domain:
            lines.append(f"Current setting: provider={current_provider or '?'} model={current_model or '?'} domain={current_domain or '?'}")
        lines.append("\nRecorded root-cause history:")
        for prob in problems:
            lines.append(f"- [{prob['id']}] {prob['name']} (opened round {prob.get('opened_round', '?')}):")
            for att in prob.get("attempts") or []:
                status = "ACCEPTED" if att.get("accepted") else "REJECTED"
                pred = ", ".join(att.get("predicted") or []) or "none"
                fixed = ", ".join(att.get("fixed") or []) or "none"
                missed = ", ".join(att.get("missed") or []) or "none"
                skills = ", ".join(att.get("skills") or []) or "none"
                lines.append(
                    f"    r{att.get('round')} [{status} | {att.get('provider','')}/{att.get('model','')} | {att.get('domain','')}] "
                    f"skills=({skills}) what={att.get('what','')[:220]} | "
                    f"predicted=({pred}) fixed=({fixed}) missed=({missed}) score={att.get('passed',0)}/{att.get('total',0)}"
                )
        rendered = "\n".join(lines)
        if len(rendered) > max_chars:
            return rendered[: max_chars - 32] + "\n... [SIM truncated to budget]"
        return rendered

    def summary_stats(self) -> dict[str, Any]:
        """Quantitative summary for SIM calibration & transfer reporting (Metric #3)."""
        data = self.load()
        problems = data.get("problems") or []
        total_attempts = 0
        accepted_attempts = 0
        reused_diagnoses = 0
        total_predicted = 0
        total_fixed = 0
        total_missed = 0
        models_seen: set[str] = set()
        domains_seen: set[str] = set()
        cross_model_attempts = 0
        for prob in problems:
            attempts = prob.get("attempts") or []
            if len(attempts) > 1:
                reused_diagnoses += len(attempts) - 1
            prev_model = None
            for att in attempts:
                total_attempts += 1
                if att.get("accepted"):
                    accepted_attempts += 1
                total_predicted += len(att.get("predicted") or [])
                total_fixed += len(att.get("fixed") or [])
                total_missed += len(att.get("missed") or [])
                m = str(att.get("model") or "").strip()
                d = str(att.get("domain") or "").strip()
                if m:
                    models_seen.add(m)
                    if prev_model and prev_model != m:
                        cross_model_attempts += 1
                    prev_model = m
                if d:
                    domains_seen.add(d)
        resolved = total_fixed + total_missed
        precision = (total_fixed / resolved) if resolved > 0 else None
        reuse_rate = (reused_diagnoses / total_attempts) if total_attempts > 0 else 0.0
        return {
            "problem_count": len(problems),
            "attempt_count": total_attempts,
            "accepted_attempts": accepted_attempts,
            "rejected_attempts": total_attempts - accepted_attempts,
            "diagnosis_reuse_count": reused_diagnoses,
            "diagnosis_reuse_rate": round(reuse_rate, 4),
            "predicted_total": total_predicted,
            "fixed_total": total_fixed,
            "missed_total": total_missed,
            "prediction_precision": round(precision, 4) if precision is not None else None,
            "models_seen": sorted(models_seen),
            "domains_seen": sorted(domains_seen),
            "cross_model_continuations": cross_model_attempts,
            "carried_from": data.get("carried_from"),
        }
