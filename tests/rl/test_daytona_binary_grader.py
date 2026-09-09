import pytest
from synth_optimizers.rl.daytona_binary_grader import validate_score, full_credit_score


@pytest.mark.parametrize('value',[0,0.0,1,1.0])
def test_binary_score(value):
    assert validate_score(value)==float(value)


@pytest.mark.parametrize('value',[True,False,'1',None,.5,float('nan'),float('inf'),2,-1])
def test_nonbinary_or_malformed_score_refused(value):
    with pytest.raises(ValueError):
        validate_score(value)


@pytest.mark.parametrize('value,expected',[(0,0),(.5,0),(.999999,0),(1,1),(1-1e-15,1)])
def test_full_credit_requires_all_native_credit(value,expected):
    assert full_credit_score(value)==expected


@pytest.mark.parametrize('value',[True,False,'1',None,float('nan'),float('inf'),2,-1])
def test_full_credit_invalid_scores_fail_closed(value):
    with pytest.raises(ValueError): full_credit_score(value)
