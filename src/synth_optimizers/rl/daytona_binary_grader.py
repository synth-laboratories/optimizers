"""Isolated entry point for reviewed read-only corpus graders, never agent code.

Installed in /root; invoke with Python -I. The corpus wrapper's pytest exit
status is not a reward: it passes whenever grade() returns, even for score 0.
"""
import importlib.util
import json
import math
from pathlib import Path


def validate_score(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
        raise ValueError('reviewed binary grader returned a non-binary score')
    return float(value)


def full_credit_score(value):
    """Opt-in success predicate for reviewed composite native scores.

    Preserve the raw score separately; fractional progress is not a success.
    Tolerance covers floating-point sums of component weights, not partial credit.
    """
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1+1e-12:
        raise ValueError('invalid native composite score')
    return float(abs(value-1.0) <= 1e-12)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-credit',action='store_true')
    args=parser.parse_args()
    spec = importlib.util.spec_from_file_location('trusted_corpus_grader', '/tests/grader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.grade()
    evidence = {'reward': (full_credit_score if args.full_credit else validate_score)(result.score),
                'native_score': result.score, 'success_rule': 'full_credit_1e-12' if args.full_credit else 'native_binary', 'feedback': result.feedback,
                'subscores': result.subscores, 'weights': result.weights}
    Path('/root/grade-result.json').write_text(json.dumps(evidence))


if __name__ == '__main__':
    main()
