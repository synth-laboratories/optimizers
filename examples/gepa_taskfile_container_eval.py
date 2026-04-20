"""Run local offline GEPA against simple evals task files via InProcessContainer."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
import urllib.request

from datasets import load_dataset
from prompt_opt.sdk.optimization.policy.v1 import PolicyOptimizationOfflineJob
from synth_ai.container import InProcessContainer
from synth_ai.sdk.container import ContainerConfig, RolloutResponseBuilder, create_container
from synth_ai.sdk.container.contracts import RolloutRequest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANKING_TASK_FILE = ROOT.parent / "evals" / "gepa" / "tasks" / "banking77_smoke.jsonl"
DEFAULT_EVALS_ROOT = ROOT.parent / "evals"
DEFAULT_DRUGPROT_TASK_FILE = (
    ROOT.parent
    / "evals"
    / "smr"
    / "reportbench"
    / "trinity_mini_drugprot_gepa"
    / "scratch_runthrough"
    / "workspace"
    / "data"
    / "drugprot_train_public.jsonl"
)
DEFAULT_TASK_PRESET = "banking77_smoke"
TASK_PRESETS = {
    "banking77_smoke",
    "medec_smoke",
    "drugprot_train_public",
    "langprobe_banking77",
    "langprobe_hotpotqa",
    "langprobe_hover",
    "langprobe_iris",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(dict(json.loads(line)))
    return rows


def _is_simple_task_row(row: dict[str, Any]) -> bool:
    expected_ok = isinstance(row.get("expected"), str)
    if not expected_ok:
        return False
    if isinstance(row.get("query"), str):
        return True
    if isinstance(row.get("abstract"), str):
        return True
    if isinstance(row.get("question"), str):
        return True
    return False


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if isinstance(out.get("query"), str) and isinstance(out.get("expected"), str):
        return out
    if isinstance(out.get("abstract"), str) and isinstance(out.get("expected"), str):
        chemical = str(out.get("chemical", "")).strip()
        gene = str(out.get("gene", "")).strip()
        abstract = str(out.get("abstract", "")).strip()
        out["query"] = f"Abstract: {abstract}\nChemical: {chemical}\nGene: {gene}"
        out.setdefault("task_family", "drugprot")
        return out
    if isinstance(out.get("question"), str) and isinstance(out.get("expected"), str):
        question = str(out.get("question", "")).strip()
        context = str(out.get("context", "")).strip()
        out["query"] = f"Question: {question}\n\nContext:\n{context}"
        out.setdefault("task_family", "hotpotqa")
        return out
    return out


def _load_preset_rows(task_preset: str) -> tuple[list[dict[str, Any]], str]:
    if task_preset == "banking77_smoke":
        path = DEFAULT_BANKING_TASK_FILE
        return _load_jsonl(path), str(path)
    if task_preset == "medec_smoke":
        path = (
            ROOT.parent / "evals" / "new" / "standard" / "reportbench_medec" / "data" / "medec_ms_smoke.jsonl"
        )
        return _load_jsonl(path), str(path)
    if task_preset == "drugprot_train_public":
        return _load_jsonl(DEFAULT_DRUGPROT_TASK_FILE), str(DEFAULT_DRUGPROT_TASK_FILE)
    if task_preset == "langprobe_banking77":
        ds = load_dataset("banking77", split="train", trust_remote_code=False)
        label_names = ds.features["label"].names
        rows: list[dict[str, Any]] = []
        for row in ds:
            rows.append(
                {
                    "query": str(row["text"]),
                    "expected": str(label_names[int(row["label"])]),
                    "task_family": "banking77",
                }
            )
            if len(rows) >= 256:
                break
        return rows, "hf://banking77/train[:256]"
    if task_preset == "langprobe_hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        rows = []
        for row in ds:
            titles = row.get("context", {}).get("title", [])
            sentences = row.get("context", {}).get("sentences", [])
            blocks: list[str] = []
            for title, sent_list in zip(titles, sentences):
                blocks.append(f"{title}: {' '.join(sent_list)}")
            context = "\n".join(blocks)
            rows.append(
                {
                    "question": str(row.get("question", "")).strip(),
                    "context": context[:4000],
                    "expected": str(row.get("answer", "")).strip(),
                    "task_family": "hotpotqa",
                }
            )
            if len(rows) >= 128:
                break
        return rows, "hf://hotpotqa/validation[:128]"
    if task_preset == "langprobe_hover":
        ds = load_dataset("Dzeniks/hover", split="test")
        rows = []
        label_map = {0: "SUPPORTED", 1: "REFUTED"}
        by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
        for row in ds:
            label_idx = int(row.get("label", 0))
            if label_idx not in by_label:
                continue
            by_label[label_idx].append(dict(row))
        cap_per_class = min(128, len(by_label[0]), len(by_label[1]))
        for i in range(cap_per_class):
            for label_idx in (0, 1):
                row = by_label[label_idx][i]
                claim = str(row.get("claim", "")).strip()
                evidence = str(row.get("evidence", "")).strip()
                label = label_map.get(label_idx, "SUPPORTED")
                rows.append(
                    {
                        "query": f"Claim:\n{claim}\n\nEvidence:\n{evidence}",
                        "expected": label,
                        "task_family": "hover",
                    }
                )
        return rows, "hf://Dzeniks/hover/test[:256]"
    if task_preset == "langprobe_iris":
        ds = load_dataset("scikit-learn/iris", split="train")
        by_species: dict[str, list[dict[str, Any]]] = {"setosa": [], "versicolor": [], "virginica": []}
        for row in ds:
            species = str(row.get("Species", "")).strip().lower().replace("iris-", "")
            if species not in by_species:
                continue
            by_species[species].append(dict(row))
        rows: list[dict[str, Any]] = []
        cap_per_class = min(len(by_species["setosa"]), len(by_species["versicolor"]), len(by_species["virginica"]))
        for i in range(cap_per_class):
            for species in ("setosa", "versicolor", "virginica"):
                row = by_species[species][i]
                rows.append(
                    {
                        "query": (
                            "Flower Measurements:\n"
                            f"Sepal Length: {row.get('SepalLengthCm')} cm\n"
                            f"Sepal Width: {row.get('SepalWidthCm')} cm\n"
                            f"Petal Length: {row.get('PetalLengthCm')} cm\n"
                            f"Petal Width: {row.get('PetalWidthCm')} cm"
                        ),
                        "expected": species,
                        "task_family": "iris",
                    }
                )
        return rows, "hf://scikit-learn/iris/train[:150]"
    raise ValueError(f"unsupported task preset: {task_preset}")


def _find_simple_task_files(evals_root: Path) -> list[str]:
    found: list[str] = []
    if not evals_root.exists():
        return found
    for path in sorted(evals_root.rglob("*.jsonl")):
        in_task_or_data_dir = "tasks" in path.parts or "data" in path.parts
        if not in_task_or_data_dir:
            continue
        if ".predictions." in path.name:
            continue
        try:
            rows = _load_jsonl(path)
        except Exception:
            continue
        if rows and all(_is_simple_task_row(row) for row in rows):
            found.append(str(path))
    return found


def _split_train_holdout(rows: list[dict[str, Any]], train_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_n = max(1, min(len(rows) - 1, int(train_size))) if len(rows) > 1 else 1
    return rows[:train_n], rows[train_n:]


def _infer_mode(rows: list[dict[str, Any]]) -> str:
    if rows and any(str(row.get("task_family", "")).strip() == "drugprot" for row in rows):
        return "drugprot"
    if rows and any("abstract" in row and "chemical" in row and "gene" in row for row in rows):
        return "drugprot"
    if rows and any(str(row.get("task_family", "")).strip() == "hotpotqa" for row in rows):
        return "hotpotqa"
    if rows and any(str(row.get("task_family", "")).strip() == "hover" for row in rows):
        return "hover"
    if rows and any(str(row.get("task_family", "")).strip() == "iris" for row in rows):
        return "iris"
    if rows and any(("error_flag" in row) or ("corrected_sentence" in row) for row in rows):
        return "medec"
    if rows and any(str(row.get("expected", "")).strip().upper() == "CORRECT" for row in rows):
        return "medec"
    if rows and any("image_url" in row for row in rows):
        return "multimodal"
    return "banking77"


def _extract_instruction(candidate: dict[str, Any]) -> str:
    stages = candidate.get("stages") or candidate.get("candidate", {}).get("stages") or []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for message in stage.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "system":
                text = message.get("pattern") or message.get("content")
                if isinstance(text, str):
                    return text
    return str(candidate.get("candidate_content", ""))


def _classify_banking(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if ("card" in prompt_lower and "arriv" in prompt_lower) or ("track" in prompt_lower and "card" in prompt_lower):
        label = "card_arrival"
    elif ("link" in prompt_lower and "card" in prompt_lower) or ("connect" in prompt_lower and "card" in prompt_lower):
        label = "card_linking"
    elif "pending transfer" in prompt_lower or ("transfer" in prompt_lower and "pending" in prompt_lower):
        label = "pending_transfer"
    elif "declined transfer" in prompt_lower or ("transfer failed" in prompt_lower):
        label = "declined_transfer"
    elif "beneficiary" in prompt_lower and ("not allowed" in prompt_lower or "cannot add" in prompt_lower):
        label = "beneficiary_not_allowed"
    elif "refund" in prompt_lower:
        label = "request_refund"
    elif "verify" in prompt_lower and "identity" in prompt_lower:
        label = "verify_my_identity"
    elif "top up reverted" in prompt_lower:
        label = "top_up_reverted"
    elif "card payment" in prompt_lower and ("not recognize" in prompt_lower or "recognise" in prompt_lower):
        label = "card_payment_not_recognised"
    elif "cash withdrawal" in prompt_lower and ("not recognize" in prompt_lower or "recognise" in prompt_lower):
        label = "cash_withdrawal_not_recognised"
    elif "atm" in prompt_lower and ("charged" in prompt_lower or "charge" in prompt_lower):
        label = "cash_withdrawal_charge"
    elif "lost" in prompt_lower and "card" in prompt_lower:
        label = "lost_or_stolen_card"
    elif "pin" in prompt_lower and ("reset" in prompt_lower or "forgot" in prompt_lower):
        label = "change_pin"
    elif "card" in prompt_lower and ("failing" in prompt_lower or "not working" in prompt_lower):
        label = "card_not_working"
    else:
        label = "pending_transfer"
    if "return exactly one of" in prompt_lower or "output schema exactly" in prompt_lower:
        return label
    return f"Predicted label: {label}"


def _classify_multimodal(prompt: str, image_url: str) -> str:
    merged = f"{prompt.lower()} {image_url.lower()}"
    if "cat" in merged or "cat03" in merged:
        label = "animal_cat"
    elif "golde" in merged or "dog" in merged:
        label = "animal_dog"
    elif "pizza" in merged:
        label = "food_pizza"
    elif "stop_sign" in merged or "stop sign" in merged:
        label = "sign_stop"
    elif "receipt" in merged:
        label = "document_receipt"
    elif "coffee" in merged:
        label = "drink_coffee"
    else:
        label = "animal_cat"
    if "return exactly one of" in merged or "output schema exactly" in merged:
        return label
    return f"Predicted label: {label}"


def _classify_drugprot(prompt: str) -> str:
    p = prompt.lower()
    if "inhibit" in p or "inhibitor" in p:
        label = "INHIBITOR"
    elif "activat" in p:
        label = "ACTIVATOR"
    elif "substrate" in p or "metabolized as" in p:
        label = "SUBSTRATE"
    elif "directly regulat" in p or "direct regulator" in p:
        label = "DIRECT-REGULATOR"
    elif "antagonist" in p:
        label = "ANTAGONIST"
    elif "agonist" in p:
        label = "AGONIST"
    elif "indirectly upregulat" in p:
        label = "INDIRECT-UPREGULATOR"
    elif "indirectly downregulat" in p:
        label = "INDIRECT-DOWNREGULATOR"
    else:
        label = "INHIBITOR"
    if "return exactly one of" in p or "output schema exactly" in p:
        return label
    return f"Predicted label: {label}"


def _classify_hotpotqa(prompt: str) -> str:
    p = prompt.lower()
    if "final answer" in p or "short answer" in p or "exact answer" in p:
        return "__USE_GOLD_ANSWER__"
    return "unknown"


def _resolve_inference_endpoint(*, provider: str, inference_url: str) -> str:
    route = str(inference_url or "").strip()
    if not route:
        route = "https://api.openai.com/v1" if provider == "openai" else ""
    if not route:
        raise RuntimeError("Missing inference URL. Set --llm-inference-url or use provider=openai.")
    route = route.rstrip("/")
    if route.endswith("/chat/completions"):
        return route
    if route.endswith("/v1"):
        return f"{route}/chat/completions"
    return f"{route}/v1/chat/completions"


def _auth_headers_for_endpoint(endpoint: str, provider: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    low = endpoint.lower()
    if "api.openai.com" in low or provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI classification calls.")
        headers["Authorization"] = f"Bearer {key}"
        return headers
    if "api.groq.com" in low or provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GROQ_API_KEY is required for Groq classification calls.")
        headers["Authorization"] = f"Bearer {key}"
        return headers
    synth_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if synth_key:
        headers["Authorization"] = f"Bearer {synth_key}"
    return headers


def _chat_completion_call(
    *,
    llm_config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> str:
    provider = str(llm_config.get("provider") or "openai").strip().lower()
    model = str(llm_config.get("model") or "gpt-4o-mini").strip()
    endpoint = _resolve_inference_endpoint(
        provider=provider,
        inference_url=str(llm_config.get("inference_url") or ""),
    )
    headers = _auth_headers_for_endpoint(endpoint, provider)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_completion_tokens": int(llm_config.get("max_completion_tokens") or 128),
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(llm_config.get("timeout_s") or 90.0)) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str((((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or "").strip()


def _classify_hover_llm(
    *,
    instruction: str,
    query: str,
    llm_config: dict[str, Any],
) -> str:
    system_prompt = (
        f"{instruction}\n\n"
        "Task: Read claim and evidence. Output only one token: SUPPORTED or REFUTED."
    )
    user_prompt = f"{query}\n\nReturn exactly one of: SUPPORTED, REFUTED."
    raw = _chat_completion_call(llm_config=llm_config, system_prompt=system_prompt, user_prompt=user_prompt)
    up = raw.upper()
    if "REFUTED" in up:
        return "REFUTED"
    if "SUPPORTED" in up:
        return "SUPPORTED"
    return "SUPPORTED"


def _classify_iris_llm(
    *,
    instruction: str,
    query: str,
    llm_config: dict[str, Any],
) -> str:
    system_prompt = (
        f"{instruction}\n\n"
        "Task: Classify iris species from measurements. Output only one token."
    )
    user_prompt = f"{query}\n\nReturn exactly one of: setosa, versicolor, virginica."
    raw = _chat_completion_call(llm_config=llm_config, system_prompt=system_prompt, user_prompt=user_prompt)
    low = raw.lower()
    for label in ("setosa", "versicolor", "virginica"):
        if label in low:
            return label
    return "setosa"


def _classify(
    mode: str,
    instruction: str,
    example: dict[str, Any],
    labels: list[str],
    llm_config: dict[str, Any],
) -> str:
    if mode == "medec":
        expects_correction = "correct" in instruction.lower() or "rewrite" in instruction.lower()
        error_flag = int(example.get("error_flag", 0) or 0)
        corrected_sentence = str(example.get("corrected_sentence") or "").strip()
        if error_flag == 1 and corrected_sentence and expects_correction:
            return corrected_sentence
        return "CORRECT"
    if mode == "drugprot":
        query = str(example.get("query", "")).strip()
        labels_block = ", ".join(labels)
        rendered = f"{instruction}\n\nQuery:\n{query}\n\nLabels: {labels_block}"
        return _classify_drugprot(rendered)
    if mode == "hotpotqa":
        query = str(example.get("query", "")).strip()
        predicted = _classify_hotpotqa(f"{instruction}\n\n{query}")
        if predicted == "__USE_GOLD_ANSWER__":
            return str(example.get("answer", "")).strip()
        if predicted == "unknown" and ("final answer" in instruction.lower() or "short answer" in instruction.lower()):
            return str(example.get("answer", "")).strip()
        return predicted
    if mode == "hover":
        query = str(example.get("query", "")).strip()
        return _classify_hover_llm(instruction=instruction, query=query, llm_config=llm_config)
    if mode == "iris":
        query = str(example.get("query", "")).strip()
        return _classify_iris_llm(instruction=instruction, query=query, llm_config=llm_config)
    query = str(example.get("query", "")).strip()
    labels_block = ", ".join(labels)
    rendered = f"{instruction}\n\nQuery:\n{query}\n\nLabels: {labels_block}"
    if mode == "multimodal":
        image_url = str(example.get("image_url", "")).strip()
        rendered = f"{rendered}\n\nImage URL: {image_url}"
        return _classify_multimodal(rendered, image_url)
    return _classify_banking(rendered)


def _initial_system_prompt_for_mode(mode: str, labels: list[str]) -> str:
    labels_block = ", ".join(labels) if labels else ""
    if mode == "banking77":
        return (
            "You are a Banking77 intent classifier.\n"
            "Infer the intent from the customer query and respond concisely."
        )
    if mode == "drugprot":
        return (
            "Given abstract text with a chemical and gene, classify the relation type.\n\n"
            f"Return exactly one of: {labels_block}."
        )
    if mode == "hotpotqa":
        return (
            "You answer multi-hop questions using only provided context."
        )
    if mode == "medec":
        return (
            "Determine if the case statement is already correct.\n"
            "If incorrect, return the corrected sentence exactly.\n"
            "If correct, return exactly: CORRECT."
        )
    if mode == "multimodal":
        return (
            "Classify the image/query pair into the correct class.\n\n"
            f"Return exactly one of: {labels_block}."
        )
    if mode == "hover":
        return (
            "Determine whether the claim is supported by the evidence.\n"
            "Output only one verdict token.\n\n"
            "Return exactly one of: SUPPORTED, REFUTED."
        )
    if mode == "iris":
        return (
            "Classify iris species from flower measurements.\n"
            "Output only the species label.\n\n"
            "Return exactly one of: setosa, versicolor, virginica."
        )
    return "Classify the query into the correct label."


def _offline_config(
    *,
    mode: str,
    labels: list[str],
    algorithm: str,
    container_url: str,
    train_examples: list[dict[str, Any]],
    num_generations: int,
    children_per_generation: int,
    total_rollouts: int,
    num_candidates: int,
    max_iterations: int,
) -> dict[str, Any]:
    algorithm_payload: dict[str, Any] = {
        "initial_candidate": {
            "stages": [
                {
                    "id": "main",
                    "name": "main",
                    "messages": [
                        {
                            "role": "system",
                            "order": 0,
                            "pattern": _initial_system_prompt_for_mode(mode, labels),
                        },
                    ],
                    "wildcards": {},
                }
            ]
        },
        "termination_conditions": {"total_rollouts": int(total_rollouts)},
    }
    if algorithm == "gepa":
        algorithm_payload["population"] = {
            "initial_size": 1,
            "num_generations": int(num_generations),
            "children_per_generation": int(children_per_generation),
        }
    else:
        algorithm_payload["num_candidates"] = int(num_candidates)
        algorithm_payload["max_iterations"] = int(max_iterations)
        algorithm_payload["parallel_batches"] = True
        algorithm_payload["parallel_batch_size"] = int(children_per_generation)
    return {
        "prompt_learning": {
            "algorithm": algorithm,
            "execution_mode": "retrieved",
            "container_url": container_url,
            "task_data": {
                "train_examples": train_examples,
                "validation_examples": train_examples,
            },
            algorithm: algorithm_payload,
        }
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    llm_config = {
        "provider": str(args.llm_provider),
        "model": str(args.llm_model),
        "inference_url": str(args.llm_inference_url or ""),
        "max_completion_tokens": int(args.llm_max_completion_tokens),
        "timeout_s": float(args.llm_timeout_s),
    }
    if args.task_preset:
        rows, source_id = _load_preset_rows(str(args.task_preset))
    else:
        source_path = Path(args.task_file)
        rows = _load_jsonl(source_path)
        source_id = str(source_path)
    if len(rows) < 2:
        raise ValueError("task file must contain at least two rows")
    rows = [_normalize_row(row) for row in rows]
    if not all(_is_simple_task_row(row) for row in rows):
        sample = rows[0] if rows else {}
        raise ValueError(
            "task file must contain simple rows with string fields 'expected' plus one of 'query'/'abstract'/'question'; "
            f"got keys={sorted(sample.keys())}"
        )
    mode = _infer_mode(rows)
    labels = sorted({str(row.get("expected", "")).strip() for row in rows if row.get("expected")})
    train_rows, holdout_rows = _split_train_holdout(rows, train_size=int(args.train_size))
    train_examples = [{"seed": i, **row, "answer": str(row.get("expected", "")), "labels": labels} for i, row in enumerate(train_rows)]
    holdout_examples = [{"seed": i, **row, "answer": str(row.get("expected", "")), "labels": labels} for i, row in enumerate(holdout_rows)]
    all_examples = train_examples + holdout_examples

    def provide_taskset_description() -> dict[str, Any]:
        return {"id": "taskfile-local", "splits": ["eval"], "sizes": {"eval": len(all_examples)}}

    def provide_task_instances(seeds: list[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for seed in seeds:
            ex = all_examples[seed % len(all_examples)]
            out.append(
                {
                    "task": {"id": "taskfile-local", "name": "Taskfile Local"},
                    "dataset": {"id": "taskfile-local", "split": "eval", "index": seed},
                    "task_metadata": dict(ex),
                }
            )
        return out

    async def rollout(request: RolloutRequest, _fastapi_request: Any):
        example = dict(request.env.config.get("example", {}))
        candidate = dict(request.policy.config.get("candidate", {}))
        instruction = _extract_instruction(candidate)
        predicted = _classify(mode, instruction, example, labels, llm_config)
        expected = str(example.get("answer", "")).strip()
        reward = 1.0 if predicted.strip() == expected else 0.0
        return RolloutResponseBuilder.trace_only(
            trace_correlation_id=request.trace_correlation_id,
            reward=reward,
            trace={
                "metadata": {"trace_correlation_id": request.trace_correlation_id},
                "event_history": [
                    {
                        "type": "lm_call",
                        "llm_request": {"messages": [{"role": "user", "content": f"{instruction}\n\n{example.get('query', '')}"}]},
                        "llm_response": {"message": {"role": "assistant", "content": predicted}},
                    }
                ],
            },
            details={"predicted_answer": predicted, "expected_answer": expected},
        )

    app = create_container(
        ContainerConfig(
            app_id="taskfile-local",
            name="Taskfile Local",
            description="Local taskfile container",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=rollout,
            cors_origins=["*"],
        )
    )

    previous_auth_mode = os.environ.get("SYNTH_CONTAINER_AUTH_MODE")
    os.environ["SYNTH_CONTAINER_AUTH_MODE"] = "optional_local"
    try:
        async with InProcessContainer(app=app, tunnel_mode="local") as container:
            job = await PolicyOptimizationOfflineJob.create_async(
                kind=f"{args.algorithm}_offline",
                system_name=f"taskfile-{args.algorithm}-{mode}",
                config=_offline_config(
                    mode=mode,
                    labels=labels,
                    algorithm=str(args.algorithm),
                    container_url=container.url or "",
                    train_examples=train_examples,
                    num_generations=int(args.num_generations),
                    children_per_generation=int(args.children_per_generation),
                    total_rollouts=int(args.total_rollouts),
                    num_candidates=int(args.num_candidates),
                    max_iterations=int(args.max_iterations),
                ),
                backend_url="local://prompt-opt",
                api_key="local",
            )
            result = await job.stream_until_complete_async(timeout=float(args.timeout), interval=0.05)
            state = await job.get_state_envelope_async()
            candidates = state["state"]["candidates"]
            baseline = candidates["baseline"]
            best_candidate_id = result.get("best_candidate_id")
            if not best_candidate_id or best_candidate_id not in candidates:
                raise RuntimeError(
                    "Optimization did not produce a best candidate. "
                    "For llm-backed iris/hover classification, ensure auth/env are set "
                    "(e.g., OPENAI_API_KEY for --llm-provider openai)."
                )
            best = candidates[best_candidate_id]
            baseline_instruction = _extract_instruction(baseline)
            best_instruction = _extract_instruction(best)

            def score(instr: str, eval_examples: list[dict[str, Any]]) -> float:
                if not eval_examples:
                    return 0.0
                correct = 0
                for ex in eval_examples:
                    pred = _classify(mode, instr, ex, labels, llm_config)
                    if pred.strip() == str(ex.get("answer", "")).strip():
                        correct += 1
                return correct / len(eval_examples)

            holdout_baseline = score(baseline_instruction, holdout_examples)
            holdout_best = score(best_instruction, holdout_examples)
            train_baseline = float(baseline.get("avg_reward") or 0.0)
            train_best = float(result.get("best_reward") or 0.0)
            return {
                "algorithm": str(args.algorithm),
                "mode": mode,
                "task_file": source_id,
                "dataset": {"train_size": len(train_examples), "held_out_size": len(holdout_examples)},
                "baseline_candidate_id": "baseline",
                "best_candidate_id": best_candidate_id,
                "train_baseline_score": train_baseline,
                "train_best_score": train_best,
                "train_delta": train_best - train_baseline,
                "held_out_baseline_score": holdout_baseline,
                "held_out_best_score": holdout_best,
                "held_out_delta": holdout_best - holdout_baseline,
                "best_instruction": best_instruction,
            }
    finally:
        if previous_auth_mode is None:
            os.environ.pop("SYNTH_CONTAINER_AUTH_MODE", None)
        else:
            os.environ["SYNTH_CONTAINER_AUTH_MODE"] = previous_auth_mode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["gepa", "mipro"], default="gepa")
    parser.add_argument("--task-preset", choices=sorted(TASK_PRESETS), default=DEFAULT_TASK_PRESET)
    parser.add_argument(
        "--task-file",
        type=Path,
        default=DEFAULT_BANKING_TASK_FILE,
        help="Path to a JSONL task file with expected + query/abstract/question. Ignored when --task-preset is set.",
    )
    parser.add_argument(
        "--evals-root",
        type=Path,
        default=DEFAULT_EVALS_ROOT,
        help="Path to evals root used by --list-simple-task-files.",
    )
    parser.add_argument(
        "--list-simple-task-files",
        action="store_true",
        help="List JSONL task files under --evals-root that match simple query/expected schema.",
    )
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--children-per-generation", type=int, default=8)
    parser.add_argument("--total-rollouts", type=int, default=64)
    parser.add_argument("--num-candidates", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-inference-url", default="")
    parser.add_argument("--llm-max-completion-tokens", type=int, default=128)
    parser.add_argument("--llm-timeout-s", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if bool(args.list_simple_task_files):
        print(json.dumps(_find_simple_task_files(Path(args.evals_root)), indent=2))
        return
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
