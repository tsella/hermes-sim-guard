# hermes-sim-guard

Standalone Hermes Agent plugin implementing continual skill self-improvement from two papers without modifying core agent files:

- **SIM** ([arXiv:2611.18856](https://arxiv.org/abs/2611.18856)) — *Self-Improvement Memory Enables Continual Recursive Self-Improvement*: root-cause-grouped edit ledger (`sim_memory.json`) with deterministic merging, aliases, cross-model/cross-domain provenance (`carried_from`), and predicted-vs-observed outcome calibration.
- **GUARD** ([arXiv:2611.23476](https://arxiv.org/abs/2611.23476)) — *Stabilizing Recursive Self-Improvement with Regret Guidance*: pre-execution checklists $\mathcal{C}(q)$, two-layer self-review ($S$, $S^{\text{strict}}$), paired directional regret ranking ($\Delta_{\text{hurt}}$, $\Delta_{\text{gap}}$, $\mathrm{Priority}_t$), $M \ge 2$ group-relative candidate pooling + `ContrastEvolve`, and the **Soft-HURT** promotion gate with in-round `[iff]` repair, bounded JSONL rejection ledger $\mathcal{L}_g$, and automatic `skill_ledger` rollback.

## Modes (`plugins.entries.hermes-sim-guard.settings.mode`)

- **`augment+gate` (default):** Augments Hermes's warm-cache background review prompt via `pre_background_review` with the full SIM ledger + GUARD Regret Bundle, then evaluates landed skill edits in `post_background_review` with the Soft-HURT gate (auto-rolling back regressive edits via `skill_ledger`).
- **`override`:** Returns `{"action": "skip"}` from `pre_background_review` on skill-only reviews and runs GUARD's full $M \ge 2$ candidate pool + four-criterion critic + `ContrastEvolve` + Soft-HURT gate + in-round `[iff]` repair directly.

## Auxiliary Model Routing

Registers two auxiliary slots inheriting from `auxiliary.background_review`:
- `auxiliary.sim_reviewer` — checklist derivation, Layer-2 grading, four-criterion critic
- `auxiliary.sim_evolver` — candidate distillation, `[iff]` repair, `ContrastEvolve`

## Telemetry & Reports (`/hermes-sim-stats` & `hermes sim-stats`)

Tracks rolling Pre/Post attribution (default window `W = 20` turns) across 5 core quantitative metrics:
1. Checklist Obedience & Strict Pass Rate ($\Delta \bar{S}$, $\Delta \bar{S}^{\text{strict}}$)
2. Directional Regret & Soft-HURT Gate Health ($\Delta_{\text{hurt}}$, hurt/help ratio, accept/repair/rollback rates)
3. SIM Prediction Calibration & Transfer Retention (`predicted → fixed` precision, diagnosis reuse, cross-model delta)
4. Execution Efficiency Delta (Pre vs Post tokens/turn, cache-read ratio, tool iterations)
5. Skill Hygiene & Reuse (`reuse_after_patch`, $B_W$ character budget compliance, churn/rollback ratio)

Usage:
```bash
/hermes-sim-stats
/hermes-sim-stats attribution --window 20 --skill deploy-helper
/hermes-sim-stats sim
/hermes-sim-stats guard
/hermes-sim-stats report --export ./sim-report.md
hermes sim-stats summary --format json
```
