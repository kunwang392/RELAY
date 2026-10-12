"""Fresh non-thinking rollouts and the paper's signed GPT-5.1 scoring protocol."""
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from roll.utils.evaluation_protocol import (
    LLMEVAL_MED_PROMPTS, PROTOCOL_VERSION, format_llmeval_prompt,
    format_rubric_prompt, parse_judgments, rubric_issues, safe_error, score_judgments,
)

RESPONSE_LIMIT = 5120
JUDGE_MODEL = "gpt-5.1-2025-11-13"
REPEATS = 3
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "max_tokens": RESPONSE_LIMIT, "n": 1}
DOMAIN_BENCHMARKS = {
    "medicine": ("rar_medicine", "healthbench", "llmeval_med"),
    "science": ("rar_science", "researchqa"),
}
EXPECTED_ROWS = {"rar_medicine": 2242, "healthbench": 500, "llmeval_med": 571,
                 "rar_science": 2292, "researchqa": 776}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def as_list(value):
    if isinstance(value, str):
        value = json.loads(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("Expected a list or a JSON-encoded list.")
    return value


def validate_rows(benchmark, rows, check_size=True):
    """Validate external prepared inputs without modifying or writing the source data."""
    if not rows or (check_size and len(rows) != EXPECTED_ROWS[benchmark]):
        raise ValueError(f"{benchmark}: expected {EXPECTED_ROWS[benchmark]} rows, got {len(rows)}.")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{benchmark}: duplicate question IDs.")
    normalized = []
    turn_count = 0
    for row, row_id in zip(rows, ids):
        item = dict(row, id=row_id)
        if benchmark == "llmeval_med":
            turns = as_list(row["llmeval_turns"])
            if row["llmeval_category"] not in LLMEVAL_MED_PROMPTS or not turns:
                raise ValueError("Invalid LLMEval-Med category or empty conversation.")
            item["llmeval_turns"] = []
            for turn in turns:
                question = turn.get("question")
                reference = turn.get("reference", turn.get("sanswer"))
                checklist = turn.get("checklist") or ""
                if not isinstance(question, str) or not question or not isinstance(reference, str):
                    raise ValueError("LLMEval-Med turns require question and reference strings.")
                if not isinstance(checklist, str):
                    raise ValueError("LLMEval-Med checklists must be strings.")
                item["llmeval_turns"].append(dict(question=question, reference=reference, checklist=checklist))
            turn_count += len(turns)
        else:
            item["messages"] = as_list(row["messages"])
            if not item["messages"] or any(
                m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str)
                for m in item["messages"]
            ):
                raise ValueError("Evaluation requires text chat messages.")
            if not isinstance(row["prompt_str"], str) or not isinstance(row["rubrics_str"], str):
                raise ValueError("Rubric prompts and criteria must be strings.")
            weights = as_list(row["rubric_weights"])
            if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w == 0
                   for w in weights) or not any(w > 0 for w in weights):
                raise ValueError("Rubric weights must be finite and include a positive requirement.")
            if len(weights) != int(row["rubric_count"]):
                raise ValueError("Rubric count differs from the number of weights.")
            rubric_issues(row["rubrics_str"], weights)
            if benchmark == "researchqa" and any(w != 1 for w in weights):
                raise ValueError("ResearchQA requires binary criteria with unit weights.")
            item["rubric_weights"] = weights
        normalized.append(item)
    if benchmark == "llmeval_med" and check_size and turn_count != 667:
        raise ValueError(f"LLMEval-Med requires 667 turns, got {turn_count}.")
    return normalized


def conversation_messages(turns, answers, index):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn, answer in zip(turns[:index], answers):
        messages.extend([{"role": "user", "content": turn["question"]},
                         {"role": "assistant", "content": answer["text"]}])
    messages.append({"role": "user", "content": turns[index]["question"]})
    return messages


def sample_seed(base_seed, repeat, benchmark, question_id, turn):
    value = json.dumps([base_seed, repeat, benchmark, question_id, turn]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


class VLLMGenerator:
    def __init__(self, model, tensor_parallel_size=4, max_model_len=32768):
        from vllm import LLM
        self.engine = LLM(model=model, tensor_parallel_size=tensor_parallel_size,
                          dtype="bfloat16", max_model_len=max_model_len,
                          gpu_memory_utilization=0.85, seed=42)
        self.tokenizer = self.engine.get_tokenizer()
        self.max_model_len = max_model_len

    def __call__(self, messages, seeds):
        from vllm import SamplingParams
        prompts = []
        for chat in messages:
            tokens = self.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(tokens) + RESPONSE_LIMIT > self.max_model_len:
                raise ValueError("Evaluation history exceeds context capacity; increase --max-model-len. "
                                 "Inputs and the 5120-token response budget are never silently truncated.")
            prompts.append({"prompt_token_ids": tokens})
        params = [SamplingParams(**SAMPLING, seed=seed) for seed in seeds]
        outputs = self.engine.generate(prompts, sampling_params=params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError("Generator returned an incomplete batch.")
        results = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise RuntimeError("Expected one fresh answer per question.")
            answer = output.outputs[0]
            results.append({"text": answer.text, "tokens": len(answer.token_ids),
                            "finish_reason": answer.finish_reason})
        return results


def generate_records(benchmark, rows, generator, repeat, base_seed=42, batch_size=32):
    """Generate full conversations; later turns see only this repeat's prior candidate answers."""
    for offset in range(0, len(rows), batch_size):
        group = rows[offset:offset + batch_size]
        answers = [[] for _ in group]
        num_turns = [len(r["llmeval_turns"]) if benchmark == "llmeval_med" else 1 for r in group]
        for turn in range(max(num_turns)):
            active = [i for i, count in enumerate(num_turns) if turn < count]
            messages = [conversation_messages(group[i]["llmeval_turns"], answers[i], turn)
                        if benchmark == "llmeval_med" else group[i]["messages"] for i in active]
            seeds = [sample_seed(base_seed, repeat, benchmark, group[i]["id"], turn) for i in active]
            generated = generator(messages, seeds)
            if len(generated) != len(active):
                raise RuntimeError("Generator returned an incomplete batch.")
            for i, answer, seed in zip(active, generated, seeds):
                if not isinstance(answer.get("text"), str):
                    raise ValueError("Generator did not return text.")
                answers[i].append(dict(answer, seed=seed))
        for row, generated in zip(group, answers):
            record = {"benchmark": benchmark, "id": row["id"], "repeat": repeat, "answers": generated}
            if benchmark == "llmeval_med":
                record.update(category=row["llmeval_category"], turns=row["llmeval_turns"])
            else:
                record.update(prompt=row["prompt_str"], rubrics=row["rubrics_str"], weights=row["rubric_weights"])
            yield record


def strict_schema(name, properties):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}


def judge_requests(record):
    if record["benchmark"] != "llmeval_med":
        properties = {str(i + 1): {"type": "string", "enum": ["PRESENT", "NOT_PRESENT"]}
                      for i in range(len(record["weights"]))}
        prompt = format_rubric_prompt(record["prompt"], record["answers"][0]["text"], record["rubrics"])
        return [(prompt, strict_schema("rubric_judgment", properties))]
    requests, history = [], []
    for turn, answer in zip(record["turns"], record["answers"]):
        prompt = format_llmeval_prompt(record["category"], turn["question"], answer["text"],
                                      turn["reference"], turn["checklist"], tuple(history))
        prompt += ("\nTransport format only: return a JSON object with score and explanation fields. "
                   "Put the original [1] through [5] prefix in score, and its explanation in explanation. "
                   "Keep all judging instructions and scoring rules unchanged.")
        schema = strict_schema("llmeval_judgment", {
            "score": {"type": "string", "enum": [f"[{i}]" for i in range(1, 6)]},
            "explanation": {"type": "string"}})
        requests.append((prompt, schema))
        history.append((turn["question"], answer["text"]))
    return requests


def parse_grade(content, record):
    if record["benchmark"] == "llmeval_med":
        value = json.loads(content)
        if (not isinstance(value, dict) or set(value) != {"score", "explanation"}
                or value["score"] not in [f"[{i}]" for i in range(1, 6)]
                or not isinstance(value["explanation"], str)):
            raise ValueError("Invalid LLMEval-Med structured grade.")
        return {"score": int(value["score"][1]), "explanation": value["explanation"]}
    judgments = parse_judgments(content, len(record["weights"]))
    metrics = score_judgments(judgments, record["weights"], clip_score=record["benchmark"] != "healthbench")
    return {"judgments": judgments, **metrics}


class GPTJudge:
    def __init__(self, client, concurrency=16, attempts=3, max_tokens=2048):
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.attempts = attempts
        self.max_tokens = max_tokens

    async def grade(self, record, previous=None):
        async def one(prompt, schema):
            for attempt in range(self.attempts):
                try:
                    async with self.semaphore:
                        result = await self.client.chat.completions.create(
                            model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                            reasoning_effort="none", max_completion_tokens=self.max_tokens,
                            response_format=schema, stream=False)
                    choice = result.choices[0]
                    if choice.finish_reason != "stop" or not choice.message.content:
                        raise ValueError("Judge did not return a complete structured answer.")
                    grade = parse_grade(choice.message.content, record)
                    return {"success": True, "grade": grade, "raw": choice.message.content,
                            "usage": result.usage.model_dump() if result.usage else None,
                            "attempts": attempt + 1}
                except Exception as error:
                    detail = safe_error(error)
                    if attempt + 1 < self.attempts:
                        await asyncio.sleep(2 ** attempt)
            return {"success": False, "error": detail, "attempts": self.attempts}
        requests = judge_requests(record)
        previous_grades = previous["grades"] if previous else [None] * len(requests)
        if len(previous_grades) != len(requests):
            raise ValueError("Saved judgment count differs from the number of turns.")
        async def reuse_or_grade(request, old):
            return old if old and old["success"] else await one(*request)
        grades = await asyncio.gather(*(reuse_or_grade(request, old)
                                        for request, old in zip(requests, previous_grades)))
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return {"id": record["id"], "benchmark": record["benchmark"], "repeat": record["repeat"],
                "record_sha256": fingerprint, "success": all(g["success"] for g in grades), "grades": grades}


def summarize_grades(benchmark, rows):
    grades = [grade for row in rows for grade in row["grades"]]
    valid = [g["grade"]["score"] for g in grades if g["success"]]
    complete = bool(grades) and len(valid) == len(grades)
    score = None
    if complete:
        mean = statistics.mean(valid)
        score = 20 * mean if benchmark == "llmeval_med" else 100 * min(max(mean, 0), 1)
    return {"score": score, "complete": complete, "expected_judgments": len(grades),
            "successful_judgments": len(valid), "failed_judgments": len(grades) - len(valid)}


async def judge_directory(directory, concurrency=16, attempts=3, client=None):
    """Re-use saved answers on retry; never replace successful judgments or resample answers."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["protocol"] != protocol_settings():
        raise ValueError("Evaluation manifest uses a different protocol.")
    owns_client = client is None
    if owns_client:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ["GPT_EVAL_API_KEY"],
                             base_url=os.environ.get("GPT_EVAL_API_BASE") or None,
                             timeout=120, max_retries=0)
    judge = GPTJudge(client, concurrency, attempts)
    results = {name: [] for name in DOMAIN_BENCHMARKS[manifest["domain"]]}
    try:
        for repeat in range(REPEATS):
            for benchmark in results:
                stem = directory / f"repeat-{repeat + 1}-{benchmark}"
                records = read_jsonl(stem.with_suffix(".responses.jsonl"))
                if len(records) != manifest["inputs"][benchmark]["rows"]:
                    raise ValueError("Response file is incomplete; regenerate in a new output directory.")
                expected_ids = {r["id"] for r in records}
                if len(expected_ids) != len(records):
                    raise ValueError("Response file contains duplicate IDs.")
                if any(r["benchmark"] != benchmark or r["repeat"] != repeat for r in records):
                    raise ValueError("Response file benchmark/repeat mismatch.")
                if any(len(r["answers"]) != (len(r["turns"]) if benchmark == "llmeval_med" else 1)
                       or not r["answers"] for r in records):
                    raise ValueError("Response file is missing conversation turns.")
                path = stem.with_suffix(".grades.jsonl")
                cached = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
                pending = []
                for record in records:
                    digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    old = cached.get(record["id"])
                    if old and old.get("record_sha256") != digest:
                        raise ValueError("Saved answers changed after judging; use a new evaluation directory.")
                    if not old or not old["success"]:
                        pending.append(record)
                # Bound pending tasks as well as concurrent HTTP requests.
                with path.open("a") as target:
                    for offset in range(0, len(pending), concurrency):
                        graded = await asyncio.gather(*(judge.grade(r, cached.get(r["id"]))
                                                        for r in pending[offset:offset + concurrency]))
                        for row in graded:
                            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            cached[row["id"]] = row
                        target.flush()
                results[benchmark].append(summarize_grades(benchmark, [cached[r["id"]] for r in records]))
    finally:
        if owns_client:
            await client.close()
    benchmarks = {}
    for name, repeats in results.items():
        scores = [r["score"] for r in repeats]
        complete = all(score is not None for score in scores)
        benchmarks[name] = {"mean": statistics.mean(scores) if complete else None,
                            "std": statistics.pstdev(scores) if complete else None,
                            "repeats": repeats}
    summary = {"domain": manifest["domain"], "model_size": manifest["model_size"],
               "protocol": manifest["protocol"], "benchmarks": benchmarks,
               "complete": all(r["mean"] is not None for r in benchmarks.values())}
    write_json(directory / "summary.json", summary)
    return summary


def protocol_settings():
    return {"version": PROTOCOL_VERSION, "judge": JUDGE_MODEL, "reasoning_effort": "none",
            "response_format": "strict_json_schema", "sampling": SAMPLING,
            "thinking": False, "repeats": REPEATS, "std_ddof": 0}


def combine_domains(summaries):
    if len(summaries) != 2 or {s["domain"] for s in summaries} != set(DOMAIN_BENCHMARKS):
        raise ValueError("Provide one medicine and one science summary.")
    if len({s["model_size"] for s in summaries}) != 1:
        raise ValueError("Aggregate must use the same model size across domains.")
    if any(not s["complete"] or s["protocol"] != protocol_settings() for s in summaries):
        raise ValueError("Aggregate requires three complete evaluations under the same protocol.")
    benchmarks = {}
    for summary in summaries:
        if set(summary["benchmarks"]) != set(DOMAIN_BENCHMARKS[summary["domain"]]):
            raise ValueError("Domain summary contains the wrong benchmarks.")
        benchmarks.update(summary["benchmarks"])
    return {"model_size": summaries[0]["model_size"], "benchmarks": benchmarks,
            "aggregate": statistics.mean(b["mean"] for b in benchmarks.values()),
            "protocol": protocol_settings()}
