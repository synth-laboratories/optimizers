"""Real HealthBench rubric judge or real Rust Craftax, never fixture worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dual_benchmark_budget import ROOT, load_credentials, reserve, settle
from async_benchmark_runtime import enable_async, enable_world_cleanup

GITHUB = Path('/Users/joshuapurtell/GitHub')
sys.path.insert(0, str(GITHUB / 'wt-containers-cispo-conformance/src'))
os.environ['SYNTH_CISPO_RENDERER_CANARY_DIGEST'] = '43e18d1c29ee9cc6a849f8fc77c9efee'


def healthbench(args):
    sys.path.insert(0, str(GITHUB / 'evals/containers/images/healthbench2'))
    from healthbench_chat import cispo
    from healthbench_chat.targets import HEALTHBENCH_CHAT
    from synth_containers.platform.app import create_compat_app
    from serve_healthbench2 import HttpSampler

    enable_async(cispo.HealthBenchAttemptRuntime)

    load_credentials('OPENROUTER_API_KEY')
    os.environ['HEALTHBENCH_GRADER_PROVIDER'] = 'openrouter'
    os.environ['HEALTHBENCH_GRADER_MODEL'] = 'gpt-4.1-2025-04-14'
    os.environ['SYNTH_HEALTHBENCH_DATASET_PATH'] = str(ROOT / 'healthbench_dataset.jsonl')

    class BoundedJudge(cispo.ProviderRubricJudge):
        # One shared pool caps paid rubric concurrency across every episode.
        _pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix='healthbench-rubric')

        def grade_many(self, *, conversation, rubrics):
            futures = [self._pool.submit(self.grade, conversation=conversation, rubric=rubric, index=index)
                       for index, rubric in enumerate(rubrics)]
            return [future.result() for future in futures]

        def grade(self, *, conversation, rubric, index):
            # UTF-8 bytes bound text BPE tokens, with ample framing allowance.
            upper_input = len(conversation.encode()) + len(json.dumps(rubric).encode()) + 1024
            key = reserve('healthbench:gpt-4.1-rubric', (upper_input*2 + 512*8)/1e6)
            verdict = super().grade(conversation=conversation, rubric=rubric, index=index)
            usage = verdict.usage
            if usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
                settle(key, (usage['prompt_tokens']*2+usage['completion_tokens']*8)/1e6, usage)
            return verdict

    manifest = json.loads((ROOT / 'panels.json').read_text())['healthbench']
    ids = {r['task_id'] for split in ('train', 'validation', 'final') for r in manifest[split]}
    tasks = tuple(t for t in cispo.declared_tasks(count=5000) if t.task_id in ids)
    assert len(tasks) == len(ids)
    judge = BoundedJudge()
    target = cispo.HealthBenchCispoTarget.install(tasks=tasks, judge=judge, transport=HttpSampler(), handshake_ttl_seconds=14400, max_answer_tokens=1024, temperature=args.temperature)
    cispo.set_installed_target(target)
    app = create_compat_app(HEALTHBENCH_CHAT)
    cispo.mount_cispo_routes(app)
    return app, None


def craftax(args):
    sys.path.insert(0, str(GITHUB / 'evals/containers/images/craftax-gamebench-rust'))
    from craftax_gold import cispo
    from craftax_gold.stack import extend_app, resolve_binary
    from craftax_gold.targets import CRAFTAX_REACT
    from synth_containers.platform.app import create_compat_app
    from serve_healthbench2 import HttpSampler

    enable_world_cleanup(cispo.CraftaxAttemptRuntime)
    enable_async(cispo.CraftaxAttemptRuntime)

    panel = json.loads((ROOT / 'panels.json').read_text())['craftax']
    cispo.SPLIT_SEEDS = {'train': tuple(r['seed'] for r in panel['train']), 'heldout': tuple(r['seed'] for split in ('validation', 'final') for r in panel[split])}
    os.environ['SYNTH_CRAFTAX_MAX_STEPS'] = '64'
    binary = resolve_binary()
    binary_digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    engine_port = args.port + 100
    log = (ROOT / f'craftax_gold_{args.port}.log').open('a')
    engine = subprocess.Popen([str(binary), '--host', '127.0.0.1', '--port', str(engine_port)], stdout=log, stderr=subprocess.STDOUT)
    log.close()
    url = f'http://127.0.0.1:{engine_port}'
    try:
        for _ in range(120):
            if engine.poll() is not None:
                raise RuntimeError('Rust Craftax exited before readiness')
            try:
                with urllib.request.urlopen(url+'/health', timeout=1) as reply:
                    if reply.status == 200:
                        break
            except OSError:
                time.sleep(.5)
        else:
            raise RuntimeError('Rust Craftax failed readiness')
        class Sampler(HttpSampler):
            def _probe_answer(self, body):
                # Use a legal action explicitly present in the probe observation.
                messages = body.get('messages', [])
                prompt = '\n'.join(str(m.get('content','')) for m in messages)
                legal = json.loads(prompt[prompt.rfind('valid_actions=')+14:].strip().splitlines()[0])
                response = dict(super()._probe_answer(body))
                from serve_healthbench2 import render_tokens
                text = json.dumps([legal[0]])
                tokens = list(render_tokens(text))
                response['choices'][0]['message']['content'] = text
                response['token_ids']['completion'] = tokens
                response['logprobs']['completion'] = [-.1]*len(tokens)
                return response

        app = create_compat_app(CRAFTAX_REACT)
        extend_app(app, declaration=cispo.craftax_cispo_declaration(
            profile=cispo.renderer_profile(tokenizer_id='openai/gpt-oss-20b', tokenizer_digest='sha256:gpt-oss-20b-tokenizer-unpinned', stop_token_ids=(200002,199999), canary_digest=os.environ['SYNTH_CISPO_RENDERER_CANARY_DIGEST']),
            image_digest='sha256:'+binary_digest, policy_calls=8, advertised_concurrency=24),
            transport=Sampler(), world_factory=lambda:cispo.gold_world(base_url=url, steps=64),
            temperature=args.temperature, max_completion_tokens=384, handshake_ttl_seconds=14400)
        return app, engine
    except BaseException:
        engine.terminate()
        engine.wait(timeout=10)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', choices=['healthbench','craftax'])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--temperature', type=float, default=1)
    args = parser.parse_args()
    app, engine = globals()[args.benchmark](args)
    import uvicorn
    try:
        uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
    finally:
        if engine is not None and engine.poll() is None:
            engine.terminate()
            engine.wait(timeout=10)
