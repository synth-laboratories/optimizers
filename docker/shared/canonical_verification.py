"""Fail-closed protocol checks over canonical harness and native receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify_canonical(rows: list[dict], report: dict, root: Path, *, max_steps: int) -> tuple[dict, list[dict]]:
    source=json.loads((root/'canonical-source.json').read_text())
    source_ok=all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in source['files'].items())
    protocol=[r for r in rows if r.get('event')=='policy.protocol']
    calls=[r for r in rows if r.get('event')=='policy.call']
    prefixes=bool(calls)
    previous=None;compactions=0
    for row in calls:
        messages=row.get('request_messages')
        boundary=row.get('compaction_boundary') is True
        current=row.get('compactions')
        if not isinstance(messages,list) or not messages or type(current) is not int:
            prefixes=False;break
        if previous is not None:
            if boundary:
                prefixes=prefixes and current==compactions+1 and messages[0]==previous[0]
            else:
                prefixes=prefixes and current==compactions and len(messages)>len(previous) and messages[:len(previous)]==previous
        prefixes=prefixes and row.get('prefix_verified') is True and row.get('cache_mode')=='append_only'
        previous=messages;compactions=current
    private=[(e.get('state') or {}).get('private') or {} for e in report.get('episodes') or []]
    # Running out of spend (dollars, or input tokens when the route is unpriced) is a valid way to end an episode.
    spend_reasons=('usd_budget_exhausted','input_token_budget_exhausted')
    usd_exhausted=any(r.get('event')=='budget_exhausted' and r.get('reason') in spend_reasons for r in rows)
    complete=len(private)==1 and (private[0].get('terminated') is True or private[0].get('step_index')==max_steps or usd_exhausted)
    no_errors=not any(r.get('event')=='policy.error' or (r.get('event')=='budget_exhausted' and r.get('reason') not in spend_reasons) or r.get('error') for r in rows)
    passed={
        'canonical_harness':source_ok and len(protocol)==1 and protocol[0].get('harness')=='frontend.ReActHarness' and protocol[0].get('source')==source,
        'append_only_context':prefixes,
        'survival_enabled':len(protocol)==1 and protocol[0].get('rules')=='symbolic_survival',
        'model_coverage':bool(calls) and no_errors,
        'episode_complete':complete,
    }
    receipt={'schema_version':'eval.craftax-canonical-protocol.v2','protocol':'frontend-react-survival.v2',
             'source':source,'gates':passed,'model_calls':len(calls),'compactions':compactions,
             'benchmark_eligible':all(passed.values()),'max_steps':max_steps,'rules':'symbolic_survival'}
    return receipt,[{'id':name,'passed':value} for name,value in passed.items()]
