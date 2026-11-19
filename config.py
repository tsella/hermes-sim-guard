"""Typed configuration for the ``hermes-sim-guard`` plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ExecutionMode = Literal["augment+gate", "override"]

AUX_TASK_REVIEWER = "sim_reviewer"
AUX_TASK_EVOLVER = "sim_evolver"


def _as_int(val: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(val)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _as_float(val: Any, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(val)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _as_bool(val: Any, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
    return default


@dataclass(frozen=True)
class SimGuardConfig:
    """Immutable runtime parameters parsed from ``ctx.get_config``."""

    mode: ExecutionMode = "augment+gate"
    pool_size: int = 2
    beta_hurt: float = 3.5
    gamma_help: float = 1.5
    delta_noise: float = 2.0
    delta_contrast: float = 10.0
    h_max: int = 1
    h_rep: int = 2
    mu_min: float = 2.5
    playbook_max_chars: int = 2600
    rejection_ledger_window: int = 12
    attribution_window: int = 20
    enable_llm_review: bool = True
    auto_rollback: bool = True
    checklist_min_tool_calls: int = 1

    @classmethod
    def from_ctx(cls, ctx: Any) -> "SimGuardConfig":
        raw_mode = str(ctx.get_config("mode", "augment+gate") or "augment+gate").strip().lower()
        mode: ExecutionMode = "override" if raw_mode == "override" else "augment+gate"
        beta = _as_float(ctx.get_config("beta_hurt", 3.5), 3.5, minimum=0.1)
        gamma = _as_float(ctx.get_config("gamma_help", 1.5), 1.5, minimum=0.0)
        if beta <= gamma:
            beta = gamma + 0.5
        return cls(
            mode=mode,
            pool_size=_as_int(ctx.get_config("pool_size", 2), 2, minimum=2),
            beta_hurt=beta,
            gamma_help=gamma,
            delta_noise=_as_float(ctx.get_config("delta_noise", 2.0), 2.0, minimum=0.0),
            delta_contrast=_as_float(ctx.get_config("delta_contrast", 10.0), 10.0, minimum=0.0),
            h_max=_as_int(ctx.get_config("h_max", 1), 1, minimum=0),
            h_rep=_as_int(ctx.get_config("h_rep", 2), 2, minimum=0),
            mu_min=_as_float(ctx.get_config("mu_min", 2.5), 2.5, minimum=0.0),
            playbook_max_chars=_as_int(ctx.get_config("playbook_max_chars", 2600), 2600, minimum=400),
            rejection_ledger_window=_as_int(ctx.get_config("rejection_ledger_window", 12), 12, minimum=1),
            attribution_window=_as_int(ctx.get_config("attribution_window", 20), 20, minimum=1),
            enable_llm_review=_as_bool(ctx.get_config("enable_llm_review", True), True),
            auto_rollback=_as_bool(ctx.get_config("auto_rollback", True), True),
            checklist_min_tool_calls=_as_int(ctx.get_config("checklist_min_tool_calls", 1), 1, minimum=0),
        )
