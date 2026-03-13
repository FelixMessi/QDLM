import random

import evaluate as hf_evaluate

from lm_eval.tasks.humaneval.sanitize_utils import sanitize


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            sanitize(
                doc["prompt"] + "\n" + r.split("```python\n", 1)[-1].split("```")[0],
                doc["entry_point"],
            )
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


def sample_30_docs_seed_20260209(dataset):
    """Deterministic random 30-sample subset for quick, fair A/B runs."""
    total = len(dataset)
    if total <= 30:
        return dataset
    rng = random.Random(20260209)
    indices = sorted(rng.sample(range(total), 30))
    return dataset.select(indices)
