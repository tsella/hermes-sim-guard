"""GUARD §4.1 — Reference-Free Self-Review via Pre-Execution Task Checklists.

Derives atomic binary predicates C(q) strictly from the task prompt and visible skill/tool context
(never peeking at the assistant output), then grades trajectories in two layers:
  - Layer 1: deterministic trace/execution checks (clean completion, zero unresolved tool errors,
    no repeated identical tool-error loops, non-empty response)
  - Layer 2: per-criterion semantic self-review via ``ctx.llm.complete_structured`` on the
    ``sim_reviewer`` auxiliary slot (unparseable/unavailable falls back to Layer-1 checks)
Computes criterion score S(tau) and strict score S_strict(tau) per Eq. (2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from .config import AUX_TASK_REVIEWER
except ImportError:
    from config import AUX_TASK_REVIEWER

logger = logging.getLogger(__name__)

_CHECKLIST_DERIVATION_INSTRUCTIONS = (
    "You are a deterministic rubric architect. Given ONLY the user prompt and loaded skill/tool names "
    "(never a trajectory or assistant response), decompose the request into 2 to 5 concrete atomic binary "
    "sub-requirements covering explicit prompt constraints, output format, safety rules, and clean execution. "
    'Return strict JSON: {"criteria": [{"id": "req_1", "desc": "..."}]}.'
)

_LAYER2_GRADER_INSTRUCTIONS = (
    "You are a strict, reference-free Self-Reviewer with no access to hidden verifiers. "
    "Using ONLY user_prompt, tool_trace_summary, and final_response, grade every pre-registered criterion "
    "in frozen_criteria. Reply strictly with JSON: "
    '{"verdicts": [{"id": "req_k", "passed": true, "why": "<evidence from trace/output>"}]}.'
)

_ERROR_PATTERNS = (
    re.compile(r'"success"\s*:\s*false', re.IGNORECASE),
    re.compile(r'"error"\s*:\s*"[^"]+"', re.IGNORECASE),
    re.compile(r"\btraceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\bcommand failed\b|\bexit code [1-9]\d*\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class ChecklistCriterion:
    id: str
    desc: str
    layer: str = "semantic"  # "exec" | "semantic"


@dataclass(frozen=True)
class CriterionVerdict:
    id: str
    desc: str
    layer: str
    passed: bool
    why: str = ""


@dataclass(frozen=True)
class TurnGrade:
    task_signature: str
    verdicts: tuple[CriterionVerdict, ...]
    score: float
    strict_score: float
    blocking_error: bool
    used_llm_reviewer: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed_ids(self) -> list[str]:
        return [v.id for v in self.verdicts if v.passed]

    @property
    def failed_ids(self) -> list[str]:
        return [v.id for v in self.verdicts if not v.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_signature": self.task_signature,
            "score": round(self.score, 2),
            "strict_score": round(self.strict_score, 2),
            "blocking_error": self.blocking_error,
            "used_llm_reviewer": self.used_llm_reviewer,
            "verdicts": [
                {"id": v.id, "desc": v.desc, "layer": v.layer, "passed": v.passed, "why": v.why}
                for v in self.verdicts
            ],
            "meta": dict(self.meta),
        }


def compute_scores(verdicts: list[CriterionVerdict], *, blocking_error: bool) -> tuple[float, float]:
    """GUARD Eq. (2): criterion score S(tau) and strict score S^strict(tau)."""
    if not verdicts:
        return (0.0, 0.0) if blocking_error else (100.0, 100.0)
    passed = sum(1 for v in verdicts if v.passed)
    score = 100.0 * (passed / len(verdicts))
    if blocking_error:
        strict = 0.0
    elif passed == len(verdicts):
        strict = 100.0
    else:
        strict = score / 2.0
    return round(score, 2), round(strict, 2)


def task_signature_for(user_message: str, skills_loaded: list[str], tool_names: list[str]) -> str:
    """Stable coarse signature grouping comparable tasks/skill invocations."""
    skill_part = ",".join(sorted(set(skills_loaded))) or "no_skill"
    tool_part = ",".join(sorted(set(tool_names))[:4]) or "no_tools"
    digest = hashlib.sha1((user_message or "").strip().lower()[:240].encode("utf-8")).hexdigest()[:8]
    return f"{skill_part}|{tool_part}|{digest}"


def _looks_like_tool_error(result_text: str) -> bool:
    if not result_text:
        return False
    return any(pat.search(result_text[:2000]) for pat in _ERROR_PATTERNS)


def _layer1_verdicts(
    *,
    tool_trace: list[dict[str, Any]],
    assistant_response: str,
    failed: bool,
    interrupted: bool,
) -> tuple[list[CriterionVerdict], bool]:
    """Deterministic Layer-1 execution predicates grounded directly in the turn trace."""
    verdicts: list[CriterionVerdict] = []
    blocking_error = bool(failed)

    # 1. Clean completion (no failed turn)
    verdicts.append(CriterionVerdict(
        id="exec_clean_completion",
        desc="Turn completed cleanly without runtime failure or abort.",
        layer="exec",
        passed=not failed and not interrupted,
        why="ok" if (not failed and not interrupted) else ("failed" if failed else "interrupted"),
    ))

    # 2. Non-empty actionable final answer
    has_answer = bool((assistant_response or "").strip())
    verdicts.append(CriterionVerdict(
        id="exec_non_empty_response",
        desc="Final assistant response is non-empty.",
        layer="exec",
        passed=has_answer,
        why="non-empty response" if has_answer else "empty assistant response",
    ))

    # 3. Tool execution hygiene (no unresolved final tool errors or repeated identical failure loops)
    if tool_trace:
        failed_calls = [t for t in tool_trace if t.get("error")]
        # Check if a tool failed and was never followed by a successful call of the same tool
        unresolved: list[str] = []
        last_status: dict[str, bool] = {}
        repeat_fail_counts: dict[str, int] = {}
        for item in tool_trace:
            tname = str(item.get("tool_name") or "unknown")
            err = bool(item.get("error"))
            last_status[tname] = err
            if err:
                repeat_fail_counts[tname] = repeat_fail_counts.get(tname, 0) + 1
        for tname, err in last_status.items():
            if err:
                unresolved.append(tname)
        looped = [tname for tname, count in repeat_fail_counts.items() if count >= 3]
        if looped:
            blocking_error = True
        verdicts.append(CriterionVerdict(
            id="exec_tool_calls_clean",
            desc="Tool calls resolved cleanly without unresolved errors or retry loops.",
            layer="exec",
            passed=not unresolved and not looped,
            why=(
                "all tool calls resolved"
                if (not unresolved and not looped)
                else f"unresolved={unresolved} looped={looped} total_failed={len(failed_calls)}"
            ),
        ))
    return verdicts, blocking_error


def _heuristic_semantic_criteria(user_message: str, skills_loaded: list[str]) -> list[ChecklistCriterion]:
    """Deterministic pre-execution checklist fallback when LLM derivation is off or unavailable."""
    criteria = [
        ChecklistCriterion(
            id="sem_prompt_constraints",
            desc="Response directly addresses all explicit requirements in the user prompt.",
            layer="semantic",
        )
    ]
    if skills_loaded:
        criteria.append(ChecklistCriterion(
            id="sem_skill_fidelity",
            desc=f"Workflow adheres to loaded skill guidance ({', '.join(skills_loaded[:3])}) without over-triggering.",
            layer="semantic",
        ))
    if any(k in (user_message or "").lower() for k in ("json", "table", "markdown", "yaml", "diff", "patch")):
        criteria.append(ChecklistCriterion(
            id="sem_requested_format",
            desc="Output adheres to the explicitly requested structure or format.",
            layer="semantic",
        ))
    return criteria


def derive_pre_execution_checklist(
    *,
    user_message: str,
    skills_loaded: list[str],
    tool_names: list[str],
    llm: Any = None,
    enable_llm: bool = True,
) -> list[ChecklistCriterion]:
    """Derive C(q) strictly from pre-execution public inputs (prompt + skill/tool names)."""
    fallback = _heuristic_semantic_criteria(user_message, skills_loaded)
    if not enable_llm or llm is None or not (user_message or "").strip():
        return fallback
    try:
        payload = json.dumps({
            "user_prompt": (user_message or "")[:2000],
            "loaded_skills": list(skills_loaded[:8]),
            "visible_tools": list(tool_names[:20]),
        }, ensure_ascii=False)
        res = llm.complete_structured(
            instructions=_CHECKLIST_DERIVATION_INSTRUCTIONS,
            input=[{"type": "text", "text": payload}],
            json_mode=True,
            temperature=0.0,
            max_tokens=512,
            timeout=20.0,
            task=AUX_TASK_REVIEWER,
            purpose="guard_pre_execution_checklist",
        )
        parsed = getattr(res, "parsed", None)
        raw_items = parsed.get("criteria") if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else None)
        out: list[ChecklistCriterion] = []
        for idx, item in enumerate(raw_items or [], start=1):
            if not isinstance(item, dict):
                continue
            desc = str(item.get("desc") or "").strip()
            if not desc:
                continue
            cid = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(item.get("id") or f"req_{idx}").strip())[:48] or f"req_{idx}"
            out.append(ChecklistCriterion(id=cid, desc=desc[:240], layer="semantic"))
            if len(out) >= 5:
                break
        return out or fallback
    except Exception as exc:
        logger.debug("GUARD checklist derivation fell back to heuristics: %s", exc)
        return fallback


def grade_turn(
    *,
    user_message: str,
    assistant_response: str,
    tool_trace: list[dict[str, Any]],
    skills_loaded: list[str],
    pre_checklist: Optional[list[ChecklistCriterion]] = None,
    failed: bool = False,
    interrupted: bool = False,
    llm: Any = None,
    enable_llm: bool = True,
) -> TurnGrade:
    """Run GUARD two-layer self-review on a completed turn."""
    tool_names = [str(t.get("tool_name") or "") for t in tool_trace if t.get("tool_name")]
    sig = task_signature_for(user_message, skills_loaded, tool_names)
    layer1, blocking_error = _layer1_verdicts(
        tool_trace=tool_trace,
        assistant_response=assistant_response,
        failed=failed,
        interrupted=interrupted,
    )

    criteria = list(pre_checklist) if pre_checklist else derive_pre_execution_checklist(
        user_message=user_message,
        skills_loaded=skills_loaded,
        tool_names=tool_names,
        llm=llm,
        enable_llm=enable_llm,
    )

    layer2: list[CriterionVerdict] = []
    used_llm = False
    if enable_llm and llm is not None and criteria:
        try:
            trace_summary = [
                {
                    "tool": str(t.get("tool_name") or ""),
                    "error": bool(t.get("error")),
                    "duration_ms": t.get("duration_ms", 0),
                    "result_preview": str(t.get("result_preview") or "")[:240],
                }
                for t in tool_trace[:20]
            ]
            grade_input = json.dumps({
                "user_prompt": (user_message or "")[:2000],
                "loaded_skills": skills_loaded[:8],
                "tool_trace_summary": trace_summary,
                "final_response": (assistant_response or "")[:3000],
                "frozen_criteria": [{"id": c.id, "desc": c.desc} for c in criteria],
            }, ensure_ascii=False)
            res = llm.complete_structured(
                instructions=_LAYER2_GRADER_INSTRUCTIONS,
                input=[{"type": "text", "text": grade_input}],
                json_mode=True,
                temperature=0.0,
                max_tokens=768,
                timeout=25.0,
                task=AUX_TASK_REVIEWER,
                purpose="guard_layer2_self_review",
            )
            parsed = getattr(res, "parsed", None)
            raw_verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
            if isinstance(raw_verdicts, list):
                by_id = {str(v.get("id") or ""): v for v in raw_verdicts if isinstance(v, dict)}
                for crit in criteria:
                    match = by_id.get(crit.id)
                    if isinstance(match, dict) and isinstance(match.get("passed"), bool):
                        layer2.append(CriterionVerdict(
                            id=crit.id,
                            desc=crit.desc,
                            layer="semantic",
                            passed=match["passed"],
                            why=str(match.get("why") or "")[:240],
                        ))
                    else:
                        # Unparseable verdict falls back to conservative Layer-1 trace signal.
                        layer2.append(CriterionVerdict(
                            id=crit.id,
                            desc=crit.desc,
                            layer="semantic",
                            passed=not blocking_error and bool((assistant_response or "").strip()),
                            why="unparseable semantic verdict; fallback to Layer-1",
                        ))
                used_llm = True
        except Exception as exc:
            logger.debug("GUARD Layer-2 grading fell back to Layer-1: %s", exc)

    if not layer2 and criteria:
        # Deterministic fallback mirrors Layer-1 cleanliness when LLM reviewer is disabled/offline.
        clean = all(v.passed for v in layer1) and not blocking_error
        for crit in criteria:
            layer2.append(CriterionVerdict(
                id=crit.id,
                desc=crit.desc,
                layer="semantic",
                passed=clean,
                why="layer-1 deterministic proxy",
            ))

    all_verdicts = layer1 + layer2
    score, strict_score = compute_scores(all_verdicts, blocking_error=blocking_error)
    return TurnGrade(
        task_signature=sig,
        verdicts=tuple(all_verdicts),
        score=score,
        strict_score=strict_score,
        blocking_error=blocking_error,
        used_llm_reviewer=used_llm,
        meta={"tool_count": len(tool_trace), "skills_loaded": list(skills_loaded)},
    )


def summarize_tool_event(tool_name: str, result: str, duration_ms: int = 0) -> dict[str, Any]:
    """Compact tool trace item captured from ``post_tool_call``."""
    text = str(result or "")
    return {
        "tool_name": str(tool_name or ""),
        "error": _looks_like_tool_error(text),
        "duration_ms": int(duration_ms or 0),
        "result_preview": text[:240],
    }
