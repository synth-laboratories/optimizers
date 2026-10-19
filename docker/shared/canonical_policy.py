"""Thin native-step adapter to the EXACT committed frontend ReActHarness.

No second agent loop: the canonical harness owns history, tools, compaction,
plan commitment and parsing. This module records transport receipts and stops
on failure; it never substitutes noops or a heuristic after an error.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

from core.models import Rates
from suites.nonproduct.craftax.budget import Budget, BudgetExhausted
from suites.nonproduct.craftax.harness import ReActHarness

_AGENT = None
_LAST_MESSAGES = None
_LAST_COMPACTIONS = 0


def record(row: dict) -> None:
    with Path(os.environ['EVAL_LLM_USAGE_PATH']).open('a') as handle:
        handle.write(json.dumps(row) + '\n')


class InstrumentedReAct(ReActHarness):
    def _post_chat(self, body: dict) -> dict:
        global _LAST_MESSAGES, _LAST_COMPACTIONS
        messages = copy.deepcopy(body['messages'])
        boundary = self._compactions != _LAST_COMPACTIONS
        prefix_ok = _LAST_MESSAGES is None or boundary or messages[:len(_LAST_MESSAGES)] == _LAST_MESSAGES
        if not prefix_ok:
            raise RuntimeError('canonical request prefix changed outside declared compaction')
        started = time.time()
        payload = super()._post_chat(body)
        raw = payload.get('usage') or {}
        details = raw.get('prompt_tokens_details') or {}
        cached = details.get('cached_tokens')
        cached = cached if type(cached) is int and cached >= 0 else None
        prompt = raw.get('prompt_tokens')
        completion = raw.get('completion_tokens')
        if type(prompt) is not int or type(completion) is not int:
            raise RuntimeError('provider did not report token accounting')
        usd = self.budget.rates.cost(prompt, completion, cached or 0)
        record({'event':'policy.call', 'ply':self._actions_taken, 'seed':self.seed,
                'model':self.model, 'effort':self.reasoning_effort,
                'requested_max_completion_tokens':body.get('max_completion_tokens'),
                'elapsed_s':round(time.time()-started,6), 'prompt_tokens':prompt,
                'completion_tokens':completion, 'cached_tokens':cached,
                'cache_reported':cached is not None, 'usd':usd,
                'request_messages':messages, 'assistant':(payload.get('choices') or [{}])[0].get('message'),
                'prefix_verified':prefix_ok, 'compaction_boundary':boundary,
                'compactions':self._compactions,
                'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
                'harness':'frontend.ReActHarness', 'cache_mode':'append_only'})
        _LAST_MESSAGES, _LAST_COMPACTIONS = messages, self._compactions
        return payload


def choose_actions(*, observation_text, session, valid_actions, readout, seed, ply, engine=None):
    global _AGENT
    if _AGENT is None:
        rates=Rates(model=os.environ['EVAL_LLM_MODEL'],
                    usd_per_1m_input=float(os.environ['EVAL_LLM_USD_PER_1M_INPUT']),
                    usd_per_1m_output=float(os.environ['EVAL_LLM_USD_PER_1M_OUTPUT']),
                    usd_per_1m_cached_input=float(os.environ['EVAL_LLM_USD_PER_1M_CACHED_INPUT']),
                    source='trusted recipe',as_of='2026-10-19')
        _AGENT=InstrumentedReAct(model=os.environ['EVAL_LLM_MODEL'],seed=seed,
                                temperature=0,reasoning_effort=os.environ['EVAL_LLM_EFFORT'],
                                plan_min=5,plan_max=20,compact_every=16,max_attempts=1,
                                budget=Budget(max_calls=int(os.environ['EVAL_LLM_MAX_CALLS']),
                                              max_usd=float(os.environ['EVAL_LLM_MAX_USD']),rates=rates,
                                              max_input_tokens=int(os.environ['EVAL_LLM_MAX_INPUT_TOKENS'])
                                              if os.environ.get('EVAL_LLM_MAX_INPUT_TOKENS') else None))
        manifest=json.loads(Path('/opt/evals/canonical-source.json').read_text())
        record({'event':'policy.protocol','harness':'frontend.ReActHarness',
                'cache_mode':'append_only','rules':'symbolic_survival',
                'source':manifest,'metadata':_AGENT.metadata()})
    try:
        action=_AGENT.choose(readout)
    except BudgetExhausted as error:
        record({'event':'budget_exhausted','reason':error.reason,'detail':error.detail,'ply':ply})
        return {'actions':[], 'stop_episode':True,'stop_reason':error.reason}
    except Exception as error:
        record({'event':'policy.error','error':type(error).__name__+': '+str(error),'ply':ply})
        return {'actions':[], 'stop_episode':True,'stop_reason':type(error).__name__+': '+str(error)}
    return {'actions':[action], 'rationale':'frontend append-only ReAct'}
