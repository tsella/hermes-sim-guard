"""End-to-end & unit tests for the ``hermes-sim-guard`` plugin.

Tests real PluginManager discovery against a temp HERMES_HOME, SIM ledger semantics,
GUARD checklist / paired regret / Soft-HURT / ContrastEvolve / rollback, both
``augment+gate`` and ``override`` modes, 5-metric rolling Pre/Post attribution,
and the ``/hermes-sim-stats`` slash command + report generator.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from config import SimGuardConfig
from guard_checklist import CriterionVerdict, compute_scores, grade_turn
from guard_propose_gate import (
    RejectionLedger,
    StrategyPoolStore,
    check_soft_hurt_gate,
    critic_score_candidates,
    effective_size_ceiling,
    evaluate_candidate_metrics,
    run_override_evolution_round,
)
from guard_regret import build_regret_bundle
from reporter import handle_slash_command
from runtime import SimGuardRuntime
from sim_memory import SimMemoryStore
from telemetry import TelemetryStore, compute_rolling_attribution


def test_sim_memory_merge_cross_model_and_backfill(tmp_path):
    store = SimMemoryStore(tmp_path / "sim_memory.json")
    r1 = store.record_attempt(
        root_cause="Over-broad skill trigger",
        what="Initial guard",
        skills=["deploy-helper"],
        predicted=["req_scope", "req_format"],
        accepted=True,
        domain="cli",
        model="claude-opus-5-5",
        provider="anthropic",
    )
    assert r1["problem_id"] == "P1"

    # Cross-model continuation citing same root cause (case/whitespace variation) merges deterministically.
    r2 = store.record_attempt(
        root_cause="  over-broad   skill trigger ",
        what="Adapted [iff] guard for Gemini",
        skills=["deploy-helper"],
        predicted=["req_scope", "req_format"],
        accepted=True,
        domain="cli",
        model="gemini-3.8-flash",
        provider="google",
    )
    assert r2["problem_id"] == "P1"

    updated = store.backfill_outcomes(
        skills_loaded=["deploy-helper"],
        passed_criteria=["req_scope"],
        failed_criteria=["req_format"],
        model="gemini-3.8-flash",
        domain="cli",
    )
    assert updated == 1

    stats = store.summary_stats()
    assert stats["problem_count"] == 1
    assert stats["attempt_count"] == 2
    assert stats["diagnosis_reuse_count"] == 1
    assert stats["cross_model_continuations"] == 1
    assert stats["prediction_precision"] == 0.5
    assert stats["carried_from"]["into"]["model"] == "gemini-3.8-flash"

    prompt = store.render_for_prompt(current_model="gemini-3.8-flash", current_provider="google", current_domain="cli")
    assert "SELF-IMPROVEMENT MEMORY" in prompt
    assert "[P1]" in prompt


def test_guard_checklist_scoring_and_paired_regret():
    v_all = [CriterionVerdict("c1", "d1", "exec", True), CriterionVerdict("c2", "d2", "semantic", True)]
    assert compute_scores(v_all, blocking_error=False) == (100.0, 100.0)
    v_half = [CriterionVerdict("c1", "d1", "exec", True), CriterionVerdict("c2", "d2", "semantic", False)]
    assert compute_scores(v_half, blocking_error=False) == (50.0, 25.0)
    assert compute_scores(v_half, blocking_error=True) == (50.0, 0.0)

    prev_turns = [
        {"score": 100.0, "strict_score": 100.0, "verdicts": [
            {"id": "c_regressed", "desc": "keep idempotent", "passed": True},
            {"id": "c_mastered", "desc": "valid json", "passed": True},
            {"id": "c_gap", "desc": "verify exit code", "passed": False},
        ], "meta": {"skills_loaded": ["deploy-helper"]}},
    ]
    curr_turns = [
        {"score": 33.3, "strict_score": 16.7, "verdicts": [
            {"id": "c_regressed", "desc": "keep idempotent", "passed": False, "why": "duplicated write"},
            {"id": "c_mastered", "desc": "valid json", "passed": True},
            {"id": "c_gap", "desc": "verify exit code", "passed": False, "why": "skipped check"},
        ], "meta": {"skills_loaded": ["deploy-helper"]}},
        {"score": 100.0, "strict_score": 100.0, "verdicts": [
            {"id": "c_regressed", "desc": "keep idempotent", "passed": False, "why": "dup"},
            {"id": "c_mastered", "desc": "valid json", "passed": True},
            {"id": "c_gap", "desc": "verify exit code", "passed": False},
        ], "meta": {"skills_loaded": ["deploy-helper"]}},
    ]
    bundle = build_regret_bundle(current_turns=curr_turns, previous_turns=prev_turns, delta_con=10.0)
    assert [r.criterion_id for r in bundle.regressed] == ["c_regressed"]
    assert bundle.regressed[0].priority == 2.0  # Delta_hurt=1.0 + Delta_gap=1.0
    assert [r.criterion_id for r in bundle.unsolved_gap] == ["c_gap"]
    assert [r.criterion_id for r in bundle.mastered] == ["c_mastered"]
    assert len(bundle.contrasts) == 1
    rendered = bundle.to_prompt_block()
    assert "REGRESSED" in rendered and "MASTERED" in rendered and "UNSOLVED GAP" in rendered


def test_soft_hurt_gate_and_override_round_with_repair_and_contrast_evolve(tmp_path):
    # Direct gate invariant checks (Eq. 12-13)
    ok = check_soft_hurt_gate(net_gain=12.0, hurt_penalty=3.0, helped_count=2, hurt_count=1, h_max=1, mu_min=2.5)
    assert ok.promoted is True
    bad_hurt = check_soft_hurt_gate(net_gain=4.0, hurt_penalty=5.0, helped_count=1, hurt_count=1, h_max=1, mu_min=2.5)
    assert bad_hurt.promoted is False
    bad_margin = check_soft_hurt_gate(net_gain=2.0, hurt_penalty=1.0, helped_count=1, hurt_count=1, h_max=1, mu_min=2.5)
    assert bad_margin.promoted is False

    # Full override round with fake structured LLM:
    # Winner initially has 1 hurt episode (net_gain=2.0 < mu_min=2.5), triggers [iff] repair,
    # repaired candidate clears Soft-HURT with mu_min=0, and ContrastEvolve updates strategy pool!
    call_counts = {"critic": 0}

    class _FakeLlm:
        def complete_structured(self, *, purpose="", **_kw):
            if purpose == "guard_distill_candidate":
                return SimpleNamespace(parsed={
                    "root_cause": "missing guard",
                    "problem_id": "",
                    "what": "add rule",
                    "predicted": ["req_1"],
                    "skill_content": "# Skill\n[iff deploying] check env.\n",
                })
            if purpose == "guard_four_criterion_critic":
                call_counts["critic"] += 1
                if call_counts["critic"] == 1:
                    return SimpleNamespace(parsed={
                        "incumbent_scores": [
                            {"episode": 0, "score": 60.0, "strict": 30.0},
                            {"episode": 1, "score": 90.0, "strict": 45.0},
                        ],
                        "candidates": [
                            {"index": 0, "scores": [
                                {"episode": 0, "score": 70.0, "strict": 35.0},
                                {"episode": 1, "score": 84.0, "strict": 42.0},
                            ]},
                            {"index": 1, "scores": [
                                {"episode": 0, "score": 55.0, "strict": 27.5},
                                {"episode": 1, "score": 70.0, "strict": 35.0},
                            ]},
                        ],
                    })
                # Second critic call: repaired candidate eliminates hurt!
                return SimpleNamespace(parsed={
                    "incumbent_scores": [
                        {"episode": 0, "score": 60.0, "strict": 30.0},
                        {"episode": 1, "score": 90.0, "strict": 45.0},
                    ],
                    "candidates": [
                        {"index": 0, "scores": [
                            {"episode": 0, "score": 85.0, "strict": 42.5},
                            {"episode": 1, "score": 90.0, "strict": 45.0},
                        ]},
                    ],
                })
            if purpose == "guard_iff_repair":
                return SimpleNamespace(parsed={
                    "what": "scoped to prod deploy only",
                    "skill_content": "# Skill\n[iff prod deploy] check env.\n",
                })
            if purpose == "guard_contrast_evolve":
                return SimpleNamespace(parsed={
                    "learnings": ["narrow trigger prevented regression"],
                    "updated_best_prompt": "Exploit narrow [iff] triggers.",
                    "updated_worst_prompt": "Stop emitting broad always-on rules.",
                })
            return SimpleNamespace(parsed={})

    pool = StrategyPoolStore(tmp_path / "pool.json")
    ledger = RejectionLedger(tmp_path / "rejections.jsonl")
    cfg = SimGuardConfig(mode="override", pool_size=2, h_max=1, h_rep=2, mu_min=2.5)
    out = run_override_evolution_round(
        llm=_FakeLlm(),
        skill_name="deploy-helper",
        incumbent_content="# Skill\nOld body.\n",
        regret_prompt_block="REGRET",
        sim_prompt_block="SIM",
        episodes=[{"score": 60.0, "strict_score": 30.0}, {"score": 90.0, "strict_score": 45.0}],
        strategy_pool=pool,
        rejection_ledger=ledger,
        cfg=cfg,
        round_num=1,
    )
    assert out["promoted"] is True
    assert out["repaired"] is True
    assert "[iff prod deploy]" in out["final_content"]
    assert pool.get_prompts("deploy-helper", 2) == [
        "Exploit narrow [iff] triggers.",
        "Stop emitting broad always-on rules.",
    ]


def test_five_core_metrics_rolling_attribution_and_slash_report(tmp_path):
    tel = TelemetryStore(tmp_path / "telemetry.jsonl")
    sim = SimMemoryStore(tmp_path / "sim_memory.json")

    # Pre-edit turns (lower score, higher tokens & tool iters)
    for i in range(4):
        tel.append("turn_graded", {
            "skills_loaded": ["deploy-helper"],
            "score": 60.0,
            "strict_score": 30.0,
            "total_tokens": 4000,
            "cache_read_ratio": 0.40,
            "tool_iterations": 5,
            "mean_delta_hurt": 0.25,
            "model": "model-a",
            "platform": "cli",
        })
    # Promoted skill edit epoch + gate + SIM attempt
    sim.record_attempt(
        root_cause="missing preflight check",
        what="add [iff deploy] preflight",
        skills=["deploy-helper"],
        predicted=["req_1"],
        fixed=["req_1"],
        accepted=True,
        model="model-a",
        domain="cli",
    )
    tel.append("gate_decision", {
        "skill": "deploy-helper", "promoted": True, "repaired": False, "rolled_back": False,
        "hurt_penalty": 0.0, "helped_count": 2, "hurt_count": 0,
    })
    tel.append("skill_edit_epoch", {
        "round": 1, "mode": "augment+gate", "skills": ["deploy-helper"], "skill_chars": 1200,
    })
    tel.append("skill_lifecycle", {"action": "loaded", "skill_name": "deploy-helper", "reuse_after_patch": True})
    # Post-edit turns (higher score, fewer tokens & tool iters)
    for i in range(4):
        tel.append("turn_graded", {
            "skills_loaded": ["deploy-helper"],
            "score": 95.0,
            "strict_score": 90.0,
            "total_tokens": 2800,
            "cache_read_ratio": 0.70,
            "tool_iterations": 3,
            "mean_delta_hurt": 0.0,
            "model": "model-b",
            "platform": "cli",
        })

    report = compute_rolling_attribution(tel.read_events(), sim_stats=sim.summary_stats(), window=10)
    m = report["metrics"]
    assert m["1_checklist_obedience"]["delta_score"] == 35.0
    assert m["1_checklist_obedience"]["delta_strict"] == 60.0
    assert m["2_directional_regret_gate"]["accept_rate"] == 1.0
    assert m["3_sim_calibration_transfer"]["prediction_precision"] == 1.0
    assert m["4_execution_efficiency"]["delta_tokens_pct"] == -30.0
    assert m["4_execution_efficiency"]["delta_tool_iters"] == -2.0
    assert m["5_skill_hygiene_reuse"]["reuse_after_patch_rate"] == 1.0
    assert m["5_skill_hygiene_reuse"]["budget_compliance_rate"] == 1.0

    # Slash command output formats (text, md, json, export)
    text_out = handle_slash_command("summary --window 10", default_data_dir=tmp_path)
    assert "5 Core Quantitative Metrics" in text_out
    assert "+35.00" in text_out

    export_file = tmp_path / "sim-report.md"
    md_out = handle_slash_command(f"report --window 10 --export {export_file}", default_data_dir=tmp_path)
    assert export_file.is_file()
    assert "# Hermes SIM + GUARD Self-Improvement Report" in export_file.read_text(encoding="utf-8")
    assert "Report written to" in md_out

    json_out = handle_slash_command("summary --format json", default_data_dir=tmp_path)
    parsed = json.loads(json_out)
    assert len(parsed["metrics"]) == 5


def test_cache_ratio_denominator_and_full_prompt_tokens(tmp_path):
    """Hermes reports input_tokens as NON-cached prompt tokens; cache_read_ratio must stay in
    [0, 1] using full prompt_tokens (uncached + cache_read + cache_write), and total_tokens
    must reflect the full prompt + output."""
    fake_ctx = SimpleNamespace(
        state=SimpleNamespace(data_dir=tmp_path),
        get_config=lambda key, default=None: False if key == "enable_llm_review" else default,
        llm=None,
    )
    rt = SimGuardRuntime(fake_ctx)
    rt.on_pre_llm_call(session_id="s-cache", user_message="run check", model="claude-sonnet-4-6", platform="cli")
    # Heavy cache-hit turn with cache_read >> uncached input + cache_write (previously produced >100x ratio).
    rt.on_post_api_request(
        session_id="s-cache",
        usage={
            "input_tokens": 100,
            "cache_read_tokens": 8000,
            "cache_write_tokens": 1900,
            "prompt_tokens": 10000,
            "output_tokens": 250,
        },
        model="claude-sonnet-4-6",
    )
    rt.on_post_llm_call(
        session_id="s-cache",
        user_message="run check",
        assistant_response="All good.",
        model="claude-sonnet-4-6",
        platform="cli",
    )

    events = TelemetryStore(tmp_path / "telemetry.jsonl").read_events(event_type="turn_graded")
    assert len(events) == 1
    ev = events[0]
    assert ev["prompt_tokens"] == 10000
    assert ev["total_tokens"] == 10250
    assert ev["cache_write_tokens"] == 1900
    assert ev["cache_read_ratio"] == 0.8


def test_large_incumbent_size_ceiling_fallback_and_llm_critic_preview():
    """Skills already larger than playbook_max_chars use incumbent * 1.15 as their ceiling in
    both the deterministic fallback and the LLM critic preview window."""
    cfg = SimGuardConfig(playbook_max_chars=2600, enable_llm_review=False)
    incumbent = "# Big Skill\n" + ("rule line\n" * 400)  # ~4012 chars > 2600
    ceiling = effective_size_ceiling(incumbent, cfg.playbook_max_chars)
    assert ceiling == int(len(incumbent.strip()) * 1.15)

    # Scoped edit within +15% promotes under deterministic fallback; bloated +30% edit fails.
    good_edit = incumbent + "\n[iff prod] verify migrations.\n"
    bloated_edit = incumbent + ("\nAlways check extra stuff." * 120)
    inc_s, cand_lists = critic_score_candidates(
        llm=None,
        skill_name="big-skill",
        incumbent_content=incumbent,
        candidate_contents=[good_edit, bloated_edit],
        episodes=[{"score": 75.0, "strict_score": 37.5}],
        cfg=cfg,
    )
    m_good = evaluate_candidate_metrics(
        incumbent_scores=inc_s, candidate_scores=cand_lists[0], delta_noise=2.0, beta_hurt=3.5, gamma_help=1.5,
    )
    m_bloat = evaluate_candidate_metrics(
        incumbent_scores=inc_s, candidate_scores=cand_lists[1], delta_noise=2.0, beta_hurt=3.5, gamma_help=1.5,
    )
    assert check_soft_hurt_gate(
        net_gain=m_good["net_gain"], hurt_penalty=m_good["hurt_penalty"],
        helped_count=len(m_good["helped"]), hurt_count=len(m_good["hurt"]), h_max=1, mu_min=2.5,
    ).promoted is True
    assert check_soft_hurt_gate(
        net_gain=m_bloat["net_gain"], hurt_penalty=m_bloat["hurt_penalty"],
        helped_count=len(m_bloat["helped"]), hurt_count=len(m_bloat["hurt"]), h_max=1, mu_min=2.5,
    ).promoted is False

    # LLM critic receives the tail of a large incumbent instead of truncating at playbook_max_chars + 400.
    seen_payloads: list[dict] = []

    class _CaptureLlm:
        def complete_structured(self, *, input, **_kw):
            seen_payloads.append(json.loads(input[0]["text"]))
            return SimpleNamespace(parsed={
                "incumbent_scores": [{"episode": 0, "score": 75.0, "strict": 37.5}],
                "candidates": [{"index": 0, "scores": [{"episode": 0, "score": 85.0, "strict": 42.5}]}],
            })

    critic_score_candidates(
        llm=_CaptureLlm(),
        skill_name="big-skill",
        incumbent_content=incumbent,
        candidate_contents=[good_edit],
        episodes=[{"score": 75.0, "strict_score": 37.5}],
        cfg=SimGuardConfig(playbook_max_chars=2600, enable_llm_review=True),
    )
    assert len(seen_payloads) == 1
    assert seen_payloads[0]["playbook_max_chars"] == ceiling
    assert "[iff prod] verify migrations." in seen_payloads[0]["candidates"][0]["content"]


def test_plugin_real_discovery_hooks_and_slash_command(tmp_path, monkeypatch):
    """Load the plugin through Hermes's real PluginManager against a temp HERMES_HOME."""
    home = tmp_path / ".hermes"
    plugins_root = home / "plugins"
    plugins_root.mkdir(parents=True)
    (home / "skills").mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - hermes-sim-guard\n",
        encoding="utf-8",
    )
    shutil.copytree(
        PLUGIN_DIR,
        plugins_root / "hermes-sim-guard",
        ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"),
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import hermes_constants
    importlib.reload(hermes_constants)
    plugins_mod = importlib.import_module("hermes_cli.plugins")

    orig_mgr = plugins_mod._plugin_manager
    mgr = plugins_mod.PluginManager()
    plugins_mod._plugin_manager = mgr
    try:
        mgr.discover_and_load(force=True)
        assert "hermes-sim-guard" in mgr._plugins
        loaded = mgr._plugins["hermes-sim-guard"]
        assert loaded.enabled is True
        assert loaded.error is None
        assert "hermes-sim-guard" not in loaded.commands_registered
        assert "hermes-sim-stats" in mgr._plugin_commands
        assert "sim-stats" in mgr._cli_commands
        assert "sim_reviewer" in mgr._aux_tasks
        assert "sim_evolver" in mgr._aux_tasks

        # Invoke pre_background_review through the live hook bus -> gets SIM/GUARD prompt augmentation.
        results = mgr.invoke_hook(
            "pre_background_review",
            session_id="s1",
            review_memory=False,
            review_skills=True,
            explicit=False,
            focus=None,
            prompt="BASE",
            model="claude-sonnet-4-6",
            provider="anthropic",
            routed=False,
            message_count=4,
            skills_loaded=[],
            extra_tools=[],
        )
        assert any(isinstance(r, dict) and "GUARD REGRET & CONTRAST BUNDLE" in r.get("append_prompt", "") for r in results)

        # Invoke /hermes-sim-stats slash handler directly
        slash_output = mgr._plugin_commands["hermes-sim-stats"]["handler"]("summary")
        assert "5 Core Quantitative Metrics" in slash_output

        # End-to-end augment+gate rejection -> automatic rollback via skill_ledger!
        from tools import skill_ledger

        skill_dir = home / "skills" / "deploy-helper"
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        original_text = "---\nname: deploy-helper\ndescription: Deploy safely.\n---\n# Deploy\n[iff prod] verify.\n"
        skill_md.write_text(original_text, encoding="utf-8")
        before_snap = skill_ledger.capture_before(skill_dir)
        # Write an over-broad, over-budget edit that deterministic Soft-HURT rejects.
        bad_text = "---\nname: deploy-helper\ndescription: Deploy safely.\n---\n# Deploy\n" + ("Always run extra steps. " * 200)
        skill_md.write_text(bad_text, encoding="utf-8")
        skill_ledger.record_mutation(
            action="patch", skill="deploy-helper", before=before_snap, after_root=skill_dir, actor="curator",
        )

        mgr.invoke_hook(
            "post_background_review",
            session_id="s1",
            review_memory=False,
            review_skills=True,
            explicit=False,
            focus=None,
            model="claude-sonnet-4-6",
            provider="anthropic",
            routed=False,
            outcome="ok",
            result_kind="skill",
            actions=["Skill 'deploy-helper' patched"],
            tool_calls=[],
            skills_touched=["deploy-helper"],
            usage={},
            duration_seconds=1.0,
            error=None,
            skip_reason=None,
        )
        # Soft-HURT rejected the bloated unconditional edit and rolled SKILL.md back to original_text!
        assert skill_md.read_text(encoding="utf-8") == original_text
        guard_output = mgr._plugin_commands["hermes-sim-stats"]["handler"]("guard")
        assert "Rollbacks: 1" in guard_output
    finally:
        plugins_mod._plugin_manager = orig_mgr
