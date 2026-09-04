# Banking77 variance-screened 60-update run (2026-09-04)

Run 13 screened the 56 run-12 candidate training rows against the immutable
baseline checkpoint eight times each at temperature 1.3 and eight-way
concurrency. It retained exactly the rows with one through seven correct
answers out of eight.

The screen completed all 448 attempts. Fifteen rows were retained; 17 scored
0/8 and 24 scored 8/8. The single-arm screening runner never called a training
or checkpoint-save API.

Training then reached 60 durable optimizer updates after 66 sampled groups:
60 trained groups, six zero-variance skips, 1,056 rollouts, and no stale groups.
The provider receipt records 960 training examples and 690,698 training tokens.
Provider cost attribution was missing, so the recorded zero must not be treated
as an actual zero-dollar cost.

The final checkpoint is `ckpt_7e62dff529b5f6f417500065` (`pg-0@60`), with
sampler digest
`sha256:c2788719a4876f2bc880829144fd080da2eb2701346ed604077fcafcf9655a13`.

On a fifth untouched 77-intent panel at temperature zero, baseline and trained
both scored 64/77 (83.12%). Every pair tied: zero wins, zero losses, and 77
ties, for an honest heldout delta of 0.00 percentage points.

Durable raw artifacts are under
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_run13_screen_8x` and
`/Users/joshuapurtell/Documents/ChatGPT/Synth Prod/b77_variance60_uplift_13`.
The evaluation receipt SHA-256 is
`83861b849224e1fa60c5df259dba375bf3d463777bc9a5b60b2d9fc2350bed94`.
