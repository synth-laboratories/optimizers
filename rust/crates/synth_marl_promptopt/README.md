# synth_marl_promptopt

Research-engineering harness for MARL-inspired prompt-optimizer dynamics over
the public Rust GEPA container contract. It intentionally reuses GEPA's exact
workspace proposer while changing candidate evaluation, credit assignment, and
selection only.

The shared core freezes:

- one `prompt_program.v1` (`shared_instruction`, `communication_policy`, and
  `role_prompts`);
- one policy model and proposer model from a common GEPA TOML profile;
- disjoint train, selection (`gepa.task_pools.pareto`), and heldout rows;
- exact train and paired heldout rollout budgets;
- one Rust environment-owned `/rollout` score and trace contract.

The four strategy modules are isolated under `src/variants/`: COMA
counterfactual credit, IC3Net speak gating, IMAC communication bottleneck, and
RODE role hierarchy. Public GEPA itself is the baseline and runs from the same
common profile.

Run that baseline directly against the same Rust workspace revision:

```bash
cargo run -p synth_marl_promptopt --bin gepa_baseline -- --config path/to/gepa.toml
```

Run a variant:

```bash
cargo run -p synth_marl_promptopt --bin marl_promptopt -- --config path/to/variant.toml
```

Heldout rows are loaded only after the proposer/search loop and never enter a
proposer workspace. Diagnostic arms are matched interventions on the same Rust
checkpoint; they consume the same fixed rollout budget rather than receiving
free extra evaluations.
