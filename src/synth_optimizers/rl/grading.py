"""Budgeted bounded rubric grading, sharing the experiment's durable ledger."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
import threading
import uuid
import time

from .budget import micros


class BudgetedRubricJudge:
    def __init__(self, judge, budget, *, input_rate, output_rate, workers=32,
                 max_output_tokens=512):
        if type(workers) is not int or not 1 <= workers <= 128:
            raise ValueError('grader workers must be between 1 and 128')
        if max_output_tokens != 512:
            raise ValueError('HealthBench rubric adapter requires the declared 512 token cap')
        self.judge, self.budget = judge, budget
        self.input_rate, self.output_rate = Decimal(str(input_rate)), Decimal(str(output_rate))
        micros(self.input_rate)
        micros(self.output_rate)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='rl-grader')
        self._slots = threading.BoundedSemaphore(workers * 2)

    def __getattr__(self, name):
        return getattr(self.judge, name)

    def grade(self, *, conversation, rubric, index):
        # UTF-8 byte ceiling plus fixed prompt/framing allowance; no hidden
        # retry of an ambiguous request. Missing usage retains this full bound.
        upper_input = len(conversation.encode()) + len(json.dumps(rubric).encode()) + 1024
        reservation = (upper_input*self.input_rate + 512*self.output_rate) / 1_000_000
        operation = 'rubric:' + uuid.uuid4().hex
        self.budget.reserve(operation, 'rubric_judge', reservation)
        started = time.monotonic()
        result = self.judge.grade(conversation=conversation, rubric=rubric, index=index)
        usage = result.usage
        counted = None
        if usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
            counted = (Decimal(str(usage['prompt_tokens']))*self.input_rate +
                       Decimal(str(usage['completion_tokens']))*self.output_rate) / 1_000_000
        self.budget.settle(operation, counted, duration_seconds=time.monotonic()-started)
        return result

    def grade_many(self, *, conversation, rubrics):
        futures = []
        for index, rubric in enumerate(rubrics):
            self._slots.acquire()
            try:
                future = self._pool.submit(self.grade, conversation=conversation, rubric=rubric, index=index)
                future.add_done_callback(lambda _: self._slots.release())
                futures.append(future)
            except BaseException:
                self._slots.release()
                raise
        # Drain all admitted work before propagating a failure. It cannot keep
        # spending behind a supposedly quiescent failed evaluation.
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        return results

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=False)
