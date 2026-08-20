# `eval doctor` receipts for `eval.mlx.local-policy.smoke.v1` (2026-08-20)

Three fresh eval homes, all on this branch's catalog (which pins the published
digest `sha256:1954fb48382590744643a6716a897234eed8a65899297d72a1313e54c7c7ab5d`).

| File | Home | What it shows |
| --- | --- | --- |
| `doctor-catalog-digest-not-pulled.json` | fresh, no pins | `available: false`, reason `target image … is not present locally; pull or build it first` — the shipped digest is honest about needing a pull; nothing is pulled for the operator. |
| `doctor-catalog-digest-local-build-mismatch.json` | fresh, no pins, a **locally built** image tagged as the catalog image | still `available: false` — the local build's id is not the catalog's published digest, and doctor names the mismatch rather than running whatever the tag points at. |
| `pins.toml` + `doctor-operator-pin-local-build.{json,txt}` | fresh, operator pin via `synth-optimizers eval pin --home … --recipe eval.mlx.local-policy.smoke.v1 --digest <local image id>` | `available: true`, `resolvedReference` = the pinned image id — the no-new-cut operator path (`home.py:write_pin`) makes the recipe `ready` once the pinned image is present locally. |

The published GHCR digest itself could not be pulled on the build machine:
anonymous and `gh`-token pulls both get `unauthorized`/`403` (the package is
not publicly visible yet; the workflow's make-public step returns success
without effect — same defect as `workshop-craftax-eval-target`). The local
build (`docker/gsm8k-eval-target/build.sh` at containers `9916cd74`, image id
`sha256:a454cd9e80c775703f1cbcd4ead294ffe53e9420c69cb244f9304e648ad4935e`)
stands in for it in the operator-pin receipt; once the package is public,
`docker pull ghcr.io/synth-laboratories/workshop-gsm8k-eval-target@sha256:1954fb48…`
makes the first home `ready` with no pin at all.
