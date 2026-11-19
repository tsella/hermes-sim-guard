"""Hermes SIM + GUARD Plugin — Continual Self-Improvement with Regret Guidance.

Implements:
  - Self-Improvement Memory (SIM, arXiv:2611.18856)
  - GUARD: Regret-Guided Harness Evolution & Soft-HURT Gate (arXiv:2611.23476)
  - Rolling Pre/Post attribution over 5 core quantitative metrics
  - Slash command ``/hermes-sim-stats`` + CLI subcommand ``hermes sim-stats``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .config import AUX_TASK_EVOLVER, AUX_TASK_REVIEWER, SimGuardConfig
    from .reporter import build_arg_parser, handle_slash_command, run_report_command
    from .runtime import SimGuardRuntime
except ImportError:
    from config import AUX_TASK_EVOLVER, AUX_TASK_REVIEWER, SimGuardConfig
    from reporter import build_arg_parser, handle_slash_command, run_report_command
    from runtime import SimGuardRuntime


def register(ctx: Any) -> None:
    """Register auxiliary slots, lifecycle hooks, slash command, and CLI subcommand."""
    # Dedicated auxiliary LLM slots inheriting from Hermes's background_review routing.
    ctx.register_auxiliary_task(
        AUX_TASK_REVIEWER,
        display_name="SIM+GUARD Self-Reviewer",
        description="Pre-execution checklist derivation, Layer-2 semantic self-review, and four-criterion critic.",
        inherit_from="background_review",
    )
    ctx.register_auxiliary_task(
        AUX_TASK_EVOLVER,
        display_name="SIM+GUARD Evolver",
        description="GUARD candidate distillation, in-round [iff] repair, and ContrastEvolve prompt evolution.",
        inherit_from="background_review",
    )

    runtime = SimGuardRuntime(ctx)

    # Turn observation + Layer-1/2 checklist grading
    ctx.register_hook("pre_llm_call", runtime.on_pre_llm_call)
    ctx.register_hook("post_tool_call", runtime.on_post_tool_call)
    ctx.register_hook("post_api_request", runtime.on_post_api_request)
    ctx.register_hook("post_llm_call", runtime.on_post_llm_call)
    ctx.register_hook("on_session_end", runtime.on_session_end)
    ctx.register_hook("on_skill_lifecycle", runtime.on_skill_lifecycle)

    # Background skill self-improvement & curation lifecycle hooks (observe / augment / override)
    ctx.register_hook("pre_background_review", runtime.on_pre_background_review)
    ctx.register_hook("post_background_review", runtime.on_post_background_review)
    ctx.register_hook("pre_curator_run", runtime.on_pre_curator_run)
    ctx.register_hook("post_curator_run", runtime.on_post_curator_run)

    # In-session slash command: /hermes-sim-stats
    def _slash_sim_stats(raw_args: str = "", **_kw: Any) -> str:
        cfg = SimGuardConfig.from_ctx(ctx)
        return handle_slash_command(
            raw_args,
            default_data_dir=Path(ctx.state.data_dir),
            default_window=cfg.attribution_window,
            playbook_max_chars=cfg.playbook_max_chars,
        )

    ctx.register_command(
        "hermes-sim-stats",
        _slash_sim_stats,
        description="Show SIM+GUARD rolling Pre/Post attribution metrics, ledger, and reports",
        args_hint="[summary|attribution|sim|guard|report] [--window N] [--skill NAME] [--format text|md|json] [--export PATH]",
        argument_mode="text",
    )

    # Shell CLI subcommand: hermes sim-stats ...
    def _setup_cli(subparser: Any) -> None:
        subparser.add_argument(
            "section",
            nargs="?",
            default="summary",
            choices=("summary", "attribution", "sim", "guard", "report"),
            help="Report view to render (default: summary)",
        )
        subparser.add_argument("--window", "-w", type=int, default=None, help="Rolling Pre/Post window W")
        subparser.add_argument("--skill", "-s", type=str, default=None, help="Filter attribution to one skill")
        subparser.add_argument(
            "--format", "-f", dest="fmt", choices=("text", "md", "json"), default="text",
            help="Output format (default: text; 'report' defaults to md)",
        )
        subparser.add_argument("--export", "-o", type=str, default=None, help="Write report output to path")

    def _run_cli(args: Any) -> None:
        cfg = SimGuardConfig.from_ctx(ctx)
        argv = [args.section]
        if args.window is not None:
            argv += ["--window", str(args.window)]
        if args.skill:
            argv += ["--skill", args.skill]
        if args.fmt:
            argv += ["--format", args.fmt]
        if args.export:
            argv += ["--export", args.export]
        print(
            run_report_command(
                argv,
                default_data_dir=Path(ctx.state.data_dir),
                default_window=cfg.attribution_window,
                playbook_max_chars=cfg.playbook_max_chars,
            )
        )

    ctx.register_cli_command(
        "sim-stats",
        help="SIM+GUARD rolling Pre/Post telemetry & report generator",
        setup_fn=_setup_cli,
        handler_fn=_run_cli,
        description="Inspect SIM root-cause history, GUARD Soft-HURT gate stats, and 5-metric Pre/Post attribution.",
    )
