# Prompting Best Practices

Use this guidance when proposing or reflecting on GEPA prompt changes.

## Instruction Typology

Diagnose which category is missing or wrong before proposing:

- `premise`: a fact about the task the model needs to know
- `context`: broader background
- `task_priority`: what outcome to optimize for
- `core_task_description`: what the task is
- `heuristics`: strategy or approach for a specific case
- `constraints`: soft limits to minimize violations of
- `rules`: hard constraints, must or must not
- `input_description`: what the input looks like
- `output_description`: what the output should look like

## Evidence-First Loop

1. Read the exact current prompt text, parent payload, and applied candidate diffs.
2. Inspect the reusable/context files before adding new edits.
3. Identify the current best, baseline, parent, and Pareto-front candidates.
4. Read explicit improvement, regression, and verifier/rationale summaries.
5. Inspect concrete rollout rows for at least one winning pattern and one losing pattern.
6. Pair summary metadata with actual prompt, input, prediction, expected output, reward, and trace/rationale evidence.
7. Use sampled train rows and recent trial rows to understand task breadth before proposing.

## Diagnosis

- Format errors usually want `rules`, `constraints`, or `output_description`.
- Knowledge errors usually want `premise` or `context`.
- Task-family-specific mistakes usually want `heuristics`.
- Priority inversions usually want `task_priority`.
- Input misunderstanding usually wants `input_description`.
- Output-shape drift usually wants `output_description` plus a hard `rule`.

## Proposal Shape

- Submit targeted changes grounded in evidence.
- Prefer one coherent rule, heuristic, premise, or output contract over a broad rewrite.
- Think in terms of reusable guidance that can generalize across heldout examples.
- For finite closed-output tasks, compact boundary examples or label-disambiguation tables can be valid when they teach the output contract.
- For open-output tasks, convert trace evidence into general procedures; do not copy observed training targets into candidate prompts.
- GEPA needs high-variance candidates. At most one proposal should be conservative. The rest should be very ambitious, task-specific prompt updates that could plausibly produce substantially better task performance than the parent.
- The proposer is expected to shoot for large wins, not incremental polish. A proposal that only modestly clarifies the parent is usually a wasted candidate unless it is the single conservative control.
- Every ambitious proposal should attack named top failure clusters. If it cannot name the failure cluster it fixes, it is not ready.
- Output-format polish alone is not useful unless the dominant failures are format failures. Pair output contracts with task premises, decision heuristics, conflict precedence, or error-specific rules.
- For classification tasks, strong proposals often add label-family taxonomies, conflict precedence, boundary examples, and negative rules for near-neighbor labels.
- For extraction or QA tasks, strong proposals often add answer-type routing, distractor rejection, evidence-selection checks, span verification, and final-form constraints.

## Good Prompt Changes

- Enforce the exact output format.
- Forbid a recurring unwanted output pattern.
- Add one missing domain premise.
- Add one task-group-specific heuristic.
- Clarify the input fields the model must attend to.
- State the shortest valid answer contract for extraction tasks.

## Bad Prompt Changes

- Rephrasing the baseline without new information.
- Generic advice like "be careful" or "answer accurately."
- Small wording polish that cannot plausibly fix the top failure clusters.
- Low-risk variants that are unlikely to substantially outperform the parent.
- Multiple conservative proposals in the same round.
- Task-specific rules that obviously break other task groups.
- Literal training-target memorization on open-output tasks.
- Large lookup tables that teach spurious correlations instead of the task contract.
