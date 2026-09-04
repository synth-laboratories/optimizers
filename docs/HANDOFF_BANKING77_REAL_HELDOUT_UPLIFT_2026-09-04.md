# Banking77 real-training heldout-uplift completion

Date: 2026-09-04

## Status

The RL implementation is now proven to update Tinker parameters, save and
restore real training state, and change deterministic behavior. What is still
missing is the experiment's actual acceptance result: **paired uplift on a
fresh, untouched 77-intent heldout panel after effective training**.

Do not report the run-13 heldout tie as evidence. Run 13 made 60 zero-gradient
optimizer calls because its advantages were dropped at the provider boundary.
Do not substitute the run-15 training-panel result for heldout uplift. Run 15's
`+13.33` percentage-point result proves learning on its 15 training rows, not
generalization.

## Assets already available

Run `b77_variance8_gate_15` is a valid real-training run:

- 8 effective Tinker optimizer updates;
- 9 sampled groups, one zero-variance skip, no stale groups;
- 128 trained examples and 11,956 selected training tokens;
- every train call has nonzero `loss:sum`;
- deterministic training probe: 10/15 baseline versus 12/15 trained;
- baseline checkpoint: `ckpt_b229ee0836324a7d96b0d4b0`;
- trained checkpoint: `ckpt_acfd561eda65c74c9fa351ab`;
- durable catalog:
  `/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_variance8_gate_15/checkpoints.sqlite3`;
- durable run receipts:
  `/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_variance8_gate_15/receipts`.

The repaired implementation and proof artifacts are committed as `e78fca6`.
The full debugging evidence is in
`docs/receipts/banking77-cispo-debug-proof-20260904.md`.

## What remains

### 1. Freeze a genuinely unused heldout panel

Build one new panel containing exactly one heldout example for each of the 77
Banking77 intents. Exclude every heldout task id appearing in:

- all `docs/e2e/configs/run_b77*.toml` files;
- all existing Banking77 evaluation receipts under the durable Synth Prod
  artifact directories;
- the partially observed run-12 third panel;
- run-12's fresh fourth panel;
- run-13's fresh fifth panel.

Write the selected ids, labels, selection algorithm, exclusion-set digest, and
panel digest to a durable JSON file before any sampling. Once frozen, do not
replace individual rows based on their outcomes.

### 2. Evaluate the valid run-15 checkpoint first

This is the cheapest direct answer because run 15 already contains effective
training. Restart the Banking77 server at temperature zero, then run a paired
evaluation with:

- trained selector: `ckpt_acfd561eda65c74c9fa351ab`;
- baseline: `ckpt_b229ee0836324a7d96b0d4b0`;
- scope run: `b77_variance8_gate_15`;
- parameter group: `pg-0`;
- policy type: `policy-0`;
- roster: `instance-0=pg-0:policy-0`;
- split: `heldout`;
- no explicit reward-channel override; use negotiated `score::team-0`;
- all 77 frozen seeds in identical order for both arms;
- a new evaluation id that has never been submitted to the container.

Use the run-15 catalog plus a two-entry immutable artifact-digest map and a pin
matching plan `a9b30c52d6c788cc21f3a08081953485`. Restart the container before a
retry if the handshake probe has already become terminal.

### 3. Validate the receipt before interpreting the score

The evaluation is valid only if all of these hold:

- 77 unique task ids and 77 paired rows;
- exactly 154 completed attempts and no failed/cancelled attempts;
- no train/heldout overlap and no overlap with any earlier heldout panel;
- baseline and trained arms resolve to the checkpoint ids above;
- catalogued sampler refs equal loaded sampler refs for each arm;
- baseline and trained refs, rollout ids, proxy request ids, and trace digests
  are disjoint across arms;
- every reward binds to `score::team-0`;
- seed order is identical across arms;
- the receipt file's SHA-256 is recorded after completion.

Report baseline accuracy, trained accuracy, percentage-point delta,
wins/losses/ties, paired standard deviation, and a paired uncertainty interval.
Report the honest result even if uplift is zero or negative.

### 4. Only if eight updates do not provide useful heldout uplift

Do not reuse the invalid run-13 conclusion. Start a fresh effective-training
run from the immutable baseline and retain the original up-front screening
rule: sample every candidate eight times and train only rows scoring 1 through
7 correct.

Because rows can become deterministic during training, use bounded curriculum
refreshes between training stages rather than spending indefinitely on
zero-variance groups:

1. Train an initial 8 effective updates.
2. Re-screen the candidate pool 8x against that stage's immutable checkpoint.
3. Freeze the new 1–7/8 subset and train the next bounded stage.
4. Repeat until the predeclared effective-update target or cost ceiling is
   reached.

Every stage must preserve its screening receipt, exact selected ids, sampler
checkpoint, resumable state, nonzero coefficient evidence, provider metrics,
and parent lineage. Never count zero-variance skips as optimizer updates.

Evaluate the final checkpoint once on another panel frozen before evaluation.
Do not tune stage count, learning rate, or task selection based on that panel.

## Completion criteria

This work is complete only when a committed Markdown and JSON receipt contain:

- a valid effective-training checkpoint and immutable baseline;
- a fresh 77-intent heldout panel with its exclusion proof;
- a fully valid paired evaluation receipt;
- baseline/trained means and honest heldout uplift;
- wins, losses, ties, uncertainty, tokens, throughput, and cost attribution;
- provider refs and identity digests for sampler and resumable state;
- exact artifact paths and SHA-256 values;
- confirmation that all server, evaluator, and trainer processes were stopped.

Provider-reported `0.0` cost must continue to be labeled unknown while
`cost_missing=true`.
