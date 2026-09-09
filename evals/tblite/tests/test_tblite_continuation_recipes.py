"""Regression checks for the explicitly versioned research grading recipes."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

RECIPES=Path(__file__).parents[3]/'docs/e2e'

@pytest.fixture
def recipes(monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPES))
    import screen_tblite_preparation_repairs as module
    return module

def test_seed_copy_retains_owner_write_but_not_agent_seed_write(recipes):
    commands=' '.join(recipes.setup()['reproducibility-and-envsetup'])
    assert 'chmod -R u+w /seed && chmod -R go-w /seed' in commands
    assert 'chmod -R a-w /seed' not in commands
    assert 'pytest==8.4.1' in commands
    assert 'numpy==2.1.3' in commands

@pytest.mark.parametrize('corruption',[None,'expiration','key'])
def test_certificate_metadata_duplicate_subjects_and_corruption(recipes,tmp_path,corruption):
    certdir=tmp_path/'certs';certdir.mkdir()
    expiry=(datetime.now(timezone.utc)+timedelta(days=10)).replace(microsecond=0)
    subject='same.prod.example.com'
    records=[]
    for i,kind in enumerate(('server','client')):
        (certdir/f'service_{i}.crt').write_text(f'# CERT_TYPE: {kind}\n# SPIFFE_ID: \n')
        (certdir/f'service_{i}.key').write_text('test-only-key')
        records.append({'subject_name':subject,'expiration_date':expiry.strftime('%Y-%m-%d'),
            'days_to_expiry':10.0,'cert_type':kind,'spiffe_id':None})
    if corruption=='expiration':records[0]['expiration_date']='2000-01-01'
    (tmp_path/'cert_analysis.json').write_text(json.dumps({'certificates':records}))
    def run(argv,**kwargs):
        if '-subject' in argv:
            return SimpleNamespace(stdout='subject=CN = '+subject+'\nnotAfter='+expiry.strftime('%b %d %H:%M:%S %Y GMT')+'\n')
        return SimpleNamespace(stdout=b'wrong-key' if corruption=='key' and argv[1]=='pkey' else b'public-key')
    namespace={'json':json,'datetime':datetime,'subprocess':SimpleNamespace(run=run),
        'Path':lambda p:tmp_path/p.removeprefix('/app/')}
    exec(compile(recipes.certificate_check(),'<certificate-check>','exec'),namespace)
    if corruption:
        with pytest.raises(AssertionError):namespace['test_factual_certificate_metadata_v2']()
    else:namespace['test_factual_certificate_metadata_v2']()

def test_timeline_thirds_can_reach_full_credit_and_missing_event_cannot():
    # The repair changes arithmetic only, not timestamp acceptance criteria.
    native=sum([.33]*3)*.3+.4+.3
    repaired=sum([1/3]*3)*.3+.4+.3
    missing=sum([1/3]*2)*.3+.4+.3
    assert native==pytest.approx(.997)
    assert repaired==pytest.approx(1)
    assert missing<1-1e-12
