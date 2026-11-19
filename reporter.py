"""Drop-in Telemetry CLI & Report Generator (``/hermes-sim-stats`` & ``hermes sim-stats``).

Works three ways:
  1. In-session slash command ``/hermes-sim-stats [subcommand] [flags]``
  2. Hermes CLI subcommand ``hermes sim-stats [subcommand] [flags]``
  3. Standalone drop-in CLI: ``python3 reporter.py --data-dir <path> [subcommand] [flags]``
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from .guard_propose_gate import RejectionLedger
    from .sim_memory import SimMemoryStore
    from .telemetry import TelemetryStore, compute_rolling_attribution
except ImportError:
    from guard_propose_gate import RejectionLedger
    from sim_memory import SimMemoryStore
    from telemetry import TelemetryStore, compute_rolling_attribution


def _fmt(val: Any, suffix: str = "", signed: bool = False) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        spec = f"+.2f" if signed else ".2f"
        return f"{val:{spec}}{suffix}"
    return f"{val}{suffix}"


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "n/a"
    return f"{val * 100.0:.1f}%"


def build_report_data(
    *,
    data_dir: Path,
    window: int = 20,
    skill: Optional[str] = None,
    playbook_max_chars: int = 2600,
) -> dict[str, Any]:
    telemetry = TelemetryStore(data_dir / "telemetry.jsonl")
    sim = SimMemoryStore(data_dir / "sim_memory.json")
    rejections = RejectionLedger(data_dir / "rejection_ledger.jsonl")
    events = telemetry.read_events()
    sim_stats = sim.summary_stats()
    attribution = compute_rolling_attribution(
        events,
        sim_stats=sim_stats,
        window=window,
        skill_filter=skill,
        playbook_max_chars=playbook_max_chars,
    )
    return {
        "data_dir": str(data_dir),
        "attribution": attribution,
        "sim": sim.load(),
        "sim_stats": sim_stats,
        "recent_rejections": rejections.read_recent(limit=12, skill=skill),
    }


def render_text_report(bundle: dict[str, Any], *, section: str = "summary") -> str:
    attr = bundle["attribution"]
    m = attr["metrics"]
    m1 = m["1_checklist_obedience"]
    m2 = m["2_directional_regret_gate"]
    m3 = m["3_sim_calibration_transfer"]
    m4 = m["4_execution_efficiency"]
    m5 = m["5_skill_hygiene_reuse"]

    header = f"Hermes SIM + GUARD Telemetry (rolling window W={attr['window']}"
    if attr.get("skill_filter"):
        header += f", skill={attr['skill_filter']}"
    header += f", events={attr['event_count']})"

    if section == "sim":
        return _render_sim_section(header, bundle["sim"], bundle["sim_stats"])
    if section == "guard":
        return _render_guard_section(header, m1, m2, bundle["recent_rejections"])
    if section == "attribution":
        return _render_epochs_section(header, attr.get("epochs") or [])

    lines = [
        header,
        "=" * len(header),
        "",
        "5 Core Quantitative Metrics (Rolling Pre -> Post Attribution):",
        f"  1. Checklist Obedience : S={_fmt(m1['overall_score'])} (Pre {_fmt(m1['pre_score'])} -> Post {_fmt(m1['post_score'])}, delta {_fmt(m1['delta_score'], signed=True)}) | "
        f"Strict={_fmt(m1['overall_strict'])} (delta {_fmt(m1['delta_strict'], signed=True)}) [n={m1['turns_graded']}]",
        f"  2. Directional Regret  : mean_delta_hurt={_fmt(m2['mean_delta_hurt'])} | hurt/help={_fmt(m2['hurt_to_help_ratio'])} | "
        f"Soft-HURT accept={_fmt_pct(m2['accept_rate'])} ({m2['promoted']}/{m2['gate_evaluations']}, repaired={m2['repaired']}, rollbacks={m2['rollbacks']})",
        f"  3. SIM Calibration     : precision={_fmt_pct(m3['prediction_precision'])} | reuse={_fmt_pct(m3['diagnosis_reuse_rate'])} | "
        f"problems={m3['problem_count']} attempts={m3['attempt_count']} | x-model delta={_fmt(m3['cross_model_retention_delta'], signed=True)}",
        f"  4. Execution Efficiency: tokens/turn Pre {_fmt(m4['pre_tokens_per_turn'])} -> Post {_fmt(m4['post_tokens_per_turn'])} ({_fmt(m4['delta_tokens_pct'], '%', signed=True)}) | "
        f"cache={_fmt_pct(m4['post_cache_read_ratio'])} | tool iters delta={_fmt(m4['delta_tool_iters'], signed=True)}",
        f"  5. Skill Hygiene/Reuse : reuse-after-patch={_fmt_pct(m5['reuse_after_patch_rate'])} ({m5['reuse_after_patch_hits']}/{m5['skill_loads']}) | "
        f"B_W compliance={_fmt_pct(m5['budget_compliance_rate'])} | churn/rollback={_fmt(m5['churn_rollback_ratio'])}",
    ]
    epochs = attr.get("epochs") or []
    if epochs:
        lines += ["", f"Recent Promoted Edit Epochs ({len(epochs)}):"]
        for ep in epochs[-5:]:
            skills = ", ".join(ep.get("skills") or []) or "general"
            lines.append(
                f"  - r{ep.get('round','?')} [{ep.get('mode','')}] {skills} (Pre n={ep['pre_n']} -> Post n={ep['post_n']}): "
                f"delta_S={_fmt(ep['delta_score'], signed=True)}, delta_strict={_fmt(ep['delta_strict'], signed=True)}, "
                f"delta_tokens={_fmt(ep['delta_tokens_pct'], '%', signed=True)}"
            )
    return "\n".join(lines)


def _render_sim_section(header: str, sim_data: dict[str, Any], stats: dict[str, Any]) -> str:
    lines = [
        f"{header} — SIM Root-Cause Ledger",
        "=" * (len(header) + 25),
        f"Problems: {stats['problem_count']} | Attempts: {stats['attempt_count']} "
        f"(accepted={stats['accepted_attempts']}, rejected={stats['rejected_attempts']}) | "
        f"Precision: {_fmt_pct(stats['prediction_precision'])} | Reuse: {_fmt_pct(stats['diagnosis_reuse_rate'])}",
    ]
    for prob in sim_data.get("problems") or []:
        lines.append(f"\n[{prob['id']}] {prob['name']} (opened r{prob.get('opened_round', '?')})")
        for att in (prob.get("attempts") or [])[-5:]:
            tag = "OK" if att.get("accepted") else "REJ"
            lines.append(
                f"  r{att.get('round')} [{tag} {att.get('provider','')}/{att.get('model','')}] "
                f"skills={','.join(att.get('skills') or []) or '-'} | {att.get('what','')[:120]} "
                f"| fixed={len(att.get('fixed') or [])}/{len(att.get('predicted') or [])}"
            )
    return "\n".join(lines)


def _render_guard_section(
    header: str, m1: dict[str, Any], m2: dict[str, Any], rejections: list[dict[str, Any]],
) -> str:
    lines = [
        f"{header} — GUARD Regret & Soft-HURT Gate",
        "=" * (len(header) + 33),
        f"Graded turns: {m1['turns_graded']} | Mean S: {_fmt(m1['overall_score'])} | Strict S: {_fmt(m1['overall_strict'])}",
        f"Gate evals: {m2['gate_evaluations']} | Promoted: {m2['promoted']} (repaired={m2['repaired']}) | "
        f"Rejected: {m2['rejected']} | Rollbacks: {m2['rollbacks']} | Hurt/Help: {_fmt(m2['hurt_to_help_ratio'])}",
    ]
    if rejections:
        lines.append("\nActive Rejection Ledger (most recent):")
        for r in rejections:
            lines.append(
                f"  - r{r.get('round','?')} skill={r.get('skill','?')} gain={_fmt(r.get('net_gain'), signed=True)} "
                f"hurt={_fmt(r.get('hurt'))} reason={r.get('reason','')} | {str(r.get('excerpt',''))[:100]}"
            )
    else:
        lines.append("\nActive Rejection Ledger: empty")
    return "\n".join(lines)


def _render_epochs_section(header: str, epochs: list[dict[str, Any]]) -> str:
    lines = [f"{header} — Rolling Pre/Post Edit Epochs", "=" * (len(header) + 31)]
    if not epochs:
        lines.append("No promoted skill edit epochs recorded yet.")
        return "\n".join(lines)
    for ep in epochs:
        skills = ", ".join(ep.get("skills") or []) or "general"
        lines.append(
            f"- r{ep.get('round','?')} {ep.get('ts','')} [{ep.get('mode','')}] {skills} "
            f"(Pre n={ep['pre_n']} -> Post n={ep['post_n']}): "
            f"S {_fmt(ep['pre_score'])}->{_fmt(ep['post_score'])} ({_fmt(ep['delta_score'], signed=True)}), "
            f"Strict {_fmt(ep['pre_strict'])}->{_fmt(ep['post_strict'])} ({_fmt(ep['delta_strict'], signed=True)}), "
            f"Tokens {_fmt(ep['delta_tokens_pct'], '%', signed=True)}, Iters {_fmt(ep['delta_iters'], signed=True)}"
        )
    return "\n".join(lines)


def render_markdown_report(bundle: dict[str, Any]) -> str:
    """Full Markdown telemetry report suitable for saving to disk or sharing."""
    attr = bundle["attribution"]
    m = attr["metrics"]
    m1 = m["1_checklist_obedience"]
    m2 = m["2_directional_regret_gate"]
    m3 = m["3_sim_calibration_transfer"]
    m4 = m["4_execution_efficiency"]
    m5 = m["5_skill_hygiene_reuse"]

    lines = [
        "# Hermes SIM + GUARD Self-Improvement Report",
        "",
        f"- **Generated:** `{attr['generated_at']}`",
        f"- **Rolling Window:** `W = {attr['window']}` turns",
        f"- **Skill Filter:** `{attr.get('skill_filter') or 'all'}`",
        f"- **Total Telemetry Events:** `{attr['event_count']}`",
        "",
        "## 5 Core Quantitative Metrics (Rolling Pre vs Post)",
        "",
        "| # | Metric | Pre | Post | Delta / Health |",
        "|---|--------|-----|------|----------------|",
        f"| 1 | Checklist Obedience (`S` / `S_strict`) | {_fmt(m1['pre_score'])} / {_fmt(m1['pre_strict'])} | {_fmt(m1['post_score'])} / {_fmt(m1['post_strict'])} | **{_fmt(m1['delta_score'], signed=True)}** / **{_fmt(m1['delta_strict'], signed=True)}** (`n={m1['turns_graded']}`) |",
        f"| 2 | Directional Regret & Soft-HURT Gate | hurt/help={_fmt(m2['hurt_to_help_ratio'])} | accept={_fmt_pct(m2['accept_rate'])} | mean Δhurt={_fmt(m2['mean_delta_hurt'])}, rollbacks={m2['rollbacks']}/{m2['gate_evaluations']} |",
        f"| 3 | SIM Calibration & Transfer | problems={m3['problem_count']} | reuse={_fmt_pct(m3['diagnosis_reuse_rate'])} | precision={_fmt_pct(m3['prediction_precision'])}, x-model Δ={_fmt(m3['cross_model_retention_delta'], signed=True)} |",
        f"| 4 | Execution Efficiency (Tokens / Iters) | {_fmt(m4['pre_tokens_per_turn'])} tok / {_fmt(m4['pre_tool_iters'])} it | {_fmt(m4['post_tokens_per_turn'])} tok / {_fmt(m4['post_tool_iters'])} it | **{_fmt(m4['delta_tokens_pct'], '%', signed=True)}** tok, **{_fmt(m4['delta_tool_iters'], signed=True)}** it |",
        f"| 5 | Skill Hygiene & Reuse | loads={m5['skill_loads']} | reuse={_fmt_pct(m5['reuse_after_patch_rate'])} | B_W ok={_fmt_pct(m5['budget_compliance_rate'])}, churn={_fmt(m5['churn_rollback_ratio'])} |",
        "",
    ]
    epochs = attr.get("epochs") or []
    if epochs:
        lines += [
            "## Promoted Edit Epochs",
            "",
            "| Round | Skills | Mode | Pre n -> Post n | ΔS | ΔStrict | ΔTokens | ΔIters |",
            "|-------|--------|------|-----------------|----|---------|---------|--------|",
        ]
        for ep in epochs:
            lines.append(
                f"| r{ep.get('round','?')} | {', '.join(ep.get('skills') or []) or 'general'} | `{ep.get('mode','')}` | "
                f"{ep['pre_n']} → {ep['post_n']} | {_fmt(ep['delta_score'], signed=True)} | "
                f"{_fmt(ep['delta_strict'], signed=True)} | {_fmt(ep['delta_tokens_pct'], '%', signed=True)} | "
                f"{_fmt(ep['delta_iters'], signed=True)} |"
            )
        lines.append("")
    if rejections := bundle.get("recent_rejections"):
        lines += ["## Recent Soft-HURT Rejections", ""]
        for r in rejections:
            lines.append(
                f"- **r{r.get('round','?')}** `{r.get('skill','?')}` — `{r.get('reason','')}` "
                f"(gain `{_fmt(r.get('net_gain'), signed=True)}`, hurt `{_fmt(r.get('hurt'))}`): {r.get('excerpt','')}"
            )
        lines.append("")
    return "\n".join(lines)


def build_arg_parser(prog: str = "hermes sim-stats") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, add_help=True)
    parser.add_argument(
        "section",
        nargs="?",
        default="summary",
        choices=("summary", "attribution", "sim", "guard", "report"),
        help="Report view to render (default: summary)",
    )
    parser.add_argument("--window", "-w", type=int, default=20, help="Rolling Pre/Post window W (default: 20)")
    parser.add_argument("--skill", "-s", type=str, default=None, help="Filter attribution to one skill name")
    parser.add_argument(
        "--format", "-f", dest="fmt", choices=("text", "md", "json"), default="text",
        help="Output format (default: text; 'report' defaults to md)",
    )
    parser.add_argument("--export", "-o", type=str, default=None, help="Write report output to file path")
    parser.add_argument("--data-dir", type=str, default=None, help="Override plugin data directory")
    return parser


def run_report_command(
    argv: list[str],
    *,
    default_data_dir: Path,
    default_window: int = 20,
    playbook_max_chars: int = 2600,
) -> str:
    parser = build_arg_parser("/hermes-sim-stats")
    try:
        ns = parser.parse_args(argv)
    except SystemExit:
        return parser.format_help().strip()

    data_dir = Path(ns.data_dir).expanduser() if ns.data_dir else default_data_dir
    window = ns.window if "--window" in argv or "-w" in argv else default_window
    bundle = build_report_data(
        data_dir=data_dir,
        window=window,
        skill=ns.skill,
        playbook_max_chars=playbook_max_chars,
    )
    fmt = "md" if (ns.section == "report" and "--format" not in argv and "-f" not in argv) else ns.fmt
    if fmt == "json":
        output = json.dumps(bundle["attribution"], indent=2, ensure_ascii=False)
    elif fmt == "md":
        output = render_markdown_report(bundle)
    else:
        output = render_text_report(bundle, section=ns.section)

    if ns.export:
        out_path = Path(ns.export).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
        return f"Report written to {out_path}\n\n{output}"
    return output


def handle_slash_command(
    raw_args: str,
    *,
    default_data_dir: Path,
    default_window: int = 20,
    playbook_max_chars: int = 2600,
) -> str:
    try:
        argv = shlex.split(raw_args or "")
    except ValueError as exc:
        return f"Invalid arguments: {exc}"
    return run_report_command(
        argv,
        default_data_dir=default_data_dir,
        default_window=default_window,
        playbook_max_chars=playbook_max_chars,
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone drop-in CLI entrypoint."""
    parser = build_arg_parser("hermes-sim-stats")
    ns = parser.parse_args(argv)
    data_dir = Path(ns.data_dir).expanduser() if ns.data_dir else Path.cwd()
    bundle = build_report_data(data_dir=data_dir, window=ns.window, skill=ns.skill)
    fmt = "md" if (ns.section == "report" and ns.fmt == "text") else ns.fmt
    if fmt == "json":
        text = json.dumps(bundle["attribution"], indent=2, ensure_ascii=False)
    elif fmt == "md":
        text = render_markdown_report(bundle)
    else:
        text = render_text_report(bundle, section=ns.section)
    if ns.export:
        out_path = Path(ns.export).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
