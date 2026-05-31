# Banking77 Better SDK Example

Banking77 is the golden paired SDK path for `synth-containers` and
`synth-optimizers`.

Primary run command:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
bash dev_examples/banking77/run_fresh_gepa.sh
```

The shell wrapper only loads local API-key env and calls the Python runner. The
Python path owns container serving with `Container.serve()`, passes
`handle.connection()` into `GepaConfig`, and executes through
`OptimizerRun(config).execute()`.

Expected local branches and versions:

- `containers`: `better-sdk`, `synth-containers==0.2.0.dev20260531`
- `optimizers`: `better-sdk`, `synth-optimizers==0.2.0.dev20260531`
