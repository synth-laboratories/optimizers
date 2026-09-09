"""Installed benchmark adapters; no checkout discovery or method reassignment.

The Rust Craftax engine is an independently managed endpoint. This process owns
its episode sessions and grader/episode pools, not the engine's process lifetime.
"""
import argparse
from dataclasses import replace
from contextlib import asynccontextmanager
import hashlib
import json
import socket
from pathlib import Path
import urllib.request
from urllib.parse import urlsplit

from .budget import ExperimentBudget
from .config import from_mapping
from .experiment import ExperimentSpec
from .grading import BudgetedRubricJudge
from .runtime_adapters import BoundedEpisodeRuntime


class BenchmarkSampler:
    def __init__(self, benchmark):
        self.benchmark = benchmark

    def reachable(self, origin):
        # Connection reachability only; authentication is checked on the bound
        # callback. No invented /models endpoint or paid sampling preflight.
        parts = urlsplit(origin.base_url)
        if parts.scheme not in {'http', 'https'} or not parts.hostname:
            return False
        try:
            with socket.create_connection((parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80)), timeout=5):
                return True
        except OSError:
            return False

    def post(self, url, *, headers, body):
        if url.startswith('probe://'):
            prompt = '\n'.join(str(m.get('content', '')) for m in body.get('messages', []))
            if self.benchmark == 'craftax':
                legal = json.loads(prompt[prompt.rfind('valid_actions=')+14:].strip().splitlines()[0])
                answer = json.dumps([legal[0]])
            elif self.benchmark == 'tblite':
                answer = '```bash\necho MINI_SWE_DONE\n```'
            else:
                answer = 'Seek appropriate in-person medical care.'
            def tokens(text):
                return [100000 + int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 50000 for word in text.split() or ['']]
            completion = tokens(answer)
            return {'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}],
                    'prompt_token_ids': tokens(prompt), 'token_ids': {'completion': completion},
                    'logprobs': {'completion': [-0.1]*len(completion)},
                    'usage': {'completion_tokens': len(completion)}}
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
            headers={**headers, 'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)


def create_benchmark_app(spec, *, temperature, dataset_path=None, engine_url=None):
    if temperature not in {0, 1}:
        raise ValueError('training uses temperature 1; paired evaluation uses 0')
    if not spec.renderer_profile:
        raise ValueError('a pinned renderer_profile is required by the benchmark server')
    from synth_containers.cispo_contract import RendererProfileDeclaration
    from synth_containers.platform.app import create_compat_app
    from synth_containers.http_adapter import register_cispo_routes
    profile = RendererProfileDeclaration(**spec.renderer_profile)
    config = from_mapping(spec.run)
    policy = config.budget
    budget = ExperimentBudget(policy.ledger, spec.experiment_id, policy.cap_usd)
    sampler = BenchmarkSampler(spec.benchmark)
    workers = max(spec.screening.concurrency, spec.evaluation_concurrency, config.pipeline.max_execution_slots)
    def runtime_factory(runtime):
        return BoundedEpisodeRuntime(runtime, workers=workers)
    owners = []
    if spec.benchmark == 'healthbench':
        if dataset_path is None:
            raise ValueError('HealthBench requires a local frozen dataset path')
        from healthbench_chat import cispo
        from healthbench_chat.targets import HEALTHBENCH_CHAT
        if cispo.CISPO_IMPORT_PATH != 'installed':
            raise ValueError('install the pinned synth-containers wheel; checkout fallback is not supported')
        ids = {task.task_id for panel in (spec.train, spec.validation, spec.final) for task in panel}
        corpus = tuple(json.loads(line) for line in Path(dataset_path).read_text().splitlines() if line.strip())
        tasks = tuple(task for task in cispo.declared_tasks(source=lambda: corpus, count=len(corpus)) if task.task_id in ids)
        if {task.task_id for task in tasks} != ids:
            raise ValueError('frozen HealthBench tasks are missing from the installed dataset')
        judge = BudgetedRubricJudge(cispo.ProviderRubricJudge(), budget,
            input_rate=spec.judge_input_usd_per_million, output_rate=spec.judge_output_usd_per_million)
        owners.append(judge)
        actual_protocol = {'identity': judge.identity(), 'temperature': 0, 'max_tokens': 512,
                           'normalization': 'none', 'adapter': 'healthbench.rubric.v1'}
        if actual_protocol != spec.judge_protocol:
            judge.close()
            raise ValueError('configured HealthBench judge differs from the frozen protocol')
        declaration = cispo.healthbench_cispo_declaration(tasks, profile=profile,
            evaluation_plan_id=judge.identity()['evaluation_plan_ref'], advertised_concurrency=workers)
        target = cispo.HealthBenchCispoTarget.install(tasks=tasks, judge=judge, transport=sampler,
            declaration=declaration, runtime_factory=runtime_factory, probe_judge=cispo.DeterministicRubricJudge(),
            temperature=temperature, max_answer_tokens=spec.max_tokens, handshake_ttl_seconds=14400)
        # The image's legacy hook installs its default task window. This server
        # installs the frozen target itself; duplicate routes would shadow it.
        app = create_compat_app(replace(HEALTHBENCH_CHAT, mount_routes=None))
    elif spec.benchmark == 'craftax':
        if not engine_url:
            raise ValueError('Craftax requires an independently managed Rust engine URL')
        from craftax_gold import cispo
        from craftax_gold.targets import CRAFTAX_REACT
        if cispo.CISPO_IMPORT_PATH != 'installed':
            raise ValueError('install the pinned synth-containers wheel; checkout fallback is not supported')
        pools = {'train': tuple(t.seed for t in spec.train),
                 'heldout': tuple(t.seed for panel in (spec.validation, spec.final) for t in panel)}
        if spec.judge_protocol != {'adapter': 'craftax.environment_return.v1', 'env_steps': spec.craftax_env_steps,
                                   'normalization': 'none'}:
            raise ValueError('Craftax environment reward differs from the frozen protocol')
        def cleanup(world, log):
            if world.rollout_id:
                world._request('DELETE', f'/rollouts/{world.rollout_id}', None)
                log.append('env.session.released', {'engine_rollout_id': world.rollout_id})
        target = cispo.CraftaxCispoTarget(
            declaration=cispo.craftax_cispo_declaration(profile=profile, split_seeds=pools, advertised_concurrency=workers,
                policy_calls=spec.craftax_policy_calls),
            split_seeds=pools, transport=sampler, runtime_factory=runtime_factory, world_cleanup=cleanup,
            world_factory=lambda: cispo.gold_world(base_url=engine_url), temperature=temperature,
            max_completion_tokens=spec.max_tokens, handshake_ttl_seconds=14400,
            env_step_limit=spec.craftax_env_steps)
        def metadata_extra(payload):
            # This app mounts its routes directly, without the image's global
            # install hook. Advertise the route table for this app instance.
            for key in ('metadata', 'capabilities'):
                section = payload.setdefault(key, {})
                section.setdefault('optimizer_contracts', {})['cispo'] = cispo.cispo_optimizer_block()
            return payload
        app = create_compat_app(replace(CRAFTAX_REACT, metadata_extra=metadata_extra))
    else:
        raise ValueError('benchmark server supports healthbench or craftax')
    owners.insert(0, target.attempts)
    register_cispo_routes(app, target)
    from fastapi.responses import JSONResponse
    from .experiment_runner import failure_code
    async def runtime_failure(_request, error):
        code = failure_code(error)
        status = {'provider_credit_exhausted': 402, 'authentication_failed': 401,
                  'provider_overloaded': 429, 'experiment_budget_exhausted': 409,
                  'storage_failure': 507}.get(code, 502)
        return JSONResponse(status_code=status, content={'schema_version': 'rl_runtime_error.v1',
            'code': code, 'retryable': False, 'reconciliation_required': True})
    app.add_exception_handler(Exception, runtime_failure)
    manifest = {'schema_version': 'rl_benchmark_runtime.v1', 'experiment_id': spec.experiment_id,
                'spec_digest': hashlib.sha256(spec.model_dump_json().encode()).hexdigest(),
                'temperature': temperature, 'max_tokens': spec.max_tokens,
                'budgeted_grading': spec.benchmark == 'healthbench', 'normalization': 'none'}
    @app.get('/rl/experiment')
    def experiment_manifest():
        return manifest
    previous_lifespan = app.router.lifespan_context
    @asynccontextmanager
    async def lifespan(application):
        async with previous_lifespan(application):
            try:
                yield
            finally:
                for owner in owners:
                    owner.close()
    app.router.lifespan_context = lifespan
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--temperature', required=True, type=float, choices=(0, 1))
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--dataset')
    parser.add_argument('--engine-url')
    args = parser.parse_args()
    spec = ExperimentSpec.model_validate_json(Path(args.spec).read_text())
    app = create_benchmark_app(spec, temperature=args.temperature, dataset_path=args.dataset, engine_url=args.engine_url)
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
