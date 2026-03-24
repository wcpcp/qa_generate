from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, Iterable, List, Tuple

from .providers import OpenAIResponsesProvider, ProviderExecutionError


def execute_postprocess_jobs(
    jobs: Iterable[Dict[str, Any]],
    canonical_samples: Iterable[Dict[str, Any]],
    passthrough_sample_ids: Iterable[str],
    filtered_sample_ids: Iterable[str],
    *,
    provider: OpenAIResponsesProvider,
) -> Dict[str, Any]:
    # 统一执行后处理 jobs。
    # counting 是“视觉纠错 + 重包装”，caption 是“视觉增强式重包装”，
    # 其他任务则是“保持真值不变的文本重包装”。
    jobs = list(jobs)
    samples_by_id = {sample["sample_id"]: sample for sample in canonical_samples}
    passthrough_ids = set(passthrough_sample_ids)
    filtered_ids = set(filtered_sample_ids)

    final_samples: List[Dict[str, Any]] = []
    processed_ids = set()
    unresolved_jobs: List[Dict[str, Any]] = []

    for sample_id in passthrough_ids:
        sample = samples_by_id.get(sample_id)
        if sample is not None:
            final_samples.append(_canonical_as_final(sample, source="canonical_passthrough"))
            processed_ids.add(sample_id)

    for sample_id in filtered_ids:
        processed_ids.add(sample_id)

    for job in jobs:
        sample = samples_by_id.get(job["sample_id"])
        if sample is None:
            unresolved_jobs.append(
                {
                    "job_id": job["job_id"],
                    "sample_id": job["sample_id"],
                    "reason": "missing_canonical_sample",
                }
            )
            continue

        schema = _json_schema_for_expected_output(job["expected_output"])
        try:
            provider_result = provider.run_structured_messages(
                messages=job["messages"],
                schema_name=_schema_name(job["job_id"]),
                schema=schema,
                metadata={"job_id": job["job_id"], "sample_id": job["sample_id"], "mode": job["mode"]},
            )
        except ProviderExecutionError as exc:
            _handle_unresolved(job, sample, unresolved_jobs, final_samples, processed_ids, f"provider_error: {exc}")
            continue

        merged, validation = _merge_postprocess_output(sample, job, provider_result["output_json"])
        if validation["status"] == "accepted":
            merged["postprocess"] = {
                "job_id": job["job_id"],
                "mode": job["mode"],
                "provider": provider_result["provider"],
                "model": provider_result["model"],
                "response_id": provider_result.get("response_id"),
                "usage": provider_result.get("usage"),
                "cache_key": provider_result.get("cache_key"),
                "validation": validation,
                "output_json": provider_result["output_json"],
            }
            final_samples.append(merged)
            processed_ids.add(sample["sample_id"])
            continue

        _handle_unresolved(
            job,
            sample,
            unresolved_jobs,
            final_samples,
            processed_ids,
            validation.get("reason", "validation_failed"),
            output_json=provider_result["output_json"],
            validation=validation,
        )

    for sample in canonical_samples:
        if sample["sample_id"] not in processed_ids:
            final_samples.append(_canonical_as_final(sample, source="canonical_fallback"))

    final_samples.sort(key=lambda item: item["sample_id"])
    return {
        "summary": {
            "job_count": len(jobs),
            "final_sample_count": len(final_samples),
            "processed_count": len([sample for sample in final_samples if "postprocess" in sample]),
            "passthrough_count": len([sample for sample in final_samples if sample.get("finalization_source", "").startswith("canonical")]),
            "filtered_count": len(filtered_ids),
            "unresolved_count": len(unresolved_jobs),
        },
        "final_samples": final_samples,
        "unresolved_jobs": unresolved_jobs,
    }


def derive_execution_context(
    canonical_samples: Iterable[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    # 从 prepare 阶段已经保存好的 canonical_samples.jsonl 恢复执行上下文。
    # 这样 execute 阶段只需读取 canonical + jobs，无需重新访问 metadata。
    passthrough_ids: List[str] = []
    filtered_ids: List[str] = []
    for sample in canonical_samples:
        disposition = sample.get("postprocess_disposition")
        if disposition == "filtered":
            filtered_ids.append(sample["sample_id"])
        elif disposition != "job":
            passthrough_ids.append(sample["sample_id"])
    return passthrough_ids, filtered_ids


def _handle_unresolved(
    job: Dict[str, Any],
    sample: Dict[str, Any],
    unresolved_jobs: List[Dict[str, Any]],
    final_samples: List[Dict[str, Any]],
    processed_ids: set[str],
    reason: str,
    *,
    output_json: Dict[str, Any] | None = None,
    validation: Dict[str, Any] | None = None,
) -> None:
    # 统一处理失败或无法接受的 job。
    # 对 counting 这类强制验证任务直接过滤，其他任务则回退到 canonical。
    fallback_policy = job.get("fallback_policy", "use_canonical")
    unresolved_jobs.append(
        {
            "job_id": job["job_id"],
            "sample_id": sample["sample_id"],
            "task_family": sample["task_family"],
            "fallback_policy": fallback_policy,
            "reason": reason,
            "output_json": output_json,
            "validation": validation,
        }
    )
    if fallback_policy == "use_canonical":
        final_samples.append(_canonical_as_final(sample, source="canonical_fallback"))
        processed_ids.add(sample["sample_id"])


def _merge_postprocess_output(
    sample: Dict[str, Any],
    job: Dict[str, Any],
    output_json: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # 将模型输出合并为最终样本。
    mode = job["mode"]
    if mode == "counting_visual_correct":
        return _merge_counting(sample, output_json)
    if mode == "caption_visual_refine":
        return _merge_caption(sample, output_json)
    return _merge_text_repackage(sample, output_json)


def _merge_counting(sample: Dict[str, Any], output_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    decision = str(output_json.get("decision", "")).strip().lower()
    if decision == "filter":
        return copy.deepcopy(sample), {"status": "rejected", "reason": "counting_filtered_by_model"}
    verified_count = _parse_int(output_json.get("verified_count"))
    if verified_count is None:
        return copy.deepcopy(sample), {"status": "rejected", "reason": "invalid_verified_count"}
    question = str(output_json.get("question", "")).strip() or sample["canonical_question"]
    full_answer = str(output_json.get("full_answer", "")).strip()
    if not full_answer:
        return copy.deepcopy(sample), {"status": "rejected", "reason": "empty_full_answer"}
    if not _count_answer_mentions_value(full_answer, verified_count):
        return copy.deepcopy(sample), {"status": "rejected", "reason": "verified_count_not_reflected_in_full_answer"}

    final_sample = _canonical_as_final(sample, source="llm_postprocess")
    final_sample["canonical_answer"] = verified_count
    final_sample["answer_text"] = full_answer
    final_sample["final_question"] = question
    final_sample["final_answer_text"] = full_answer
    final_sample["messages"] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_answer},
    ]
    return final_sample, {"status": "accepted", "decision": decision or "keep", "corrected": verified_count != sample.get("canonical_answer")}


def _merge_caption(sample: Dict[str, Any], output_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    decision = str(output_json.get("decision", "")).strip().lower() or "keep"
    if decision == "filter":
        return copy.deepcopy(sample), {"status": "rejected", "reason": "caption_filtered_by_model"}
    question = str(output_json.get("question", "")).strip()
    full_answer = str(output_json.get("full_answer", "")).strip()
    if not question:
        return copy.deepcopy(sample), {"status": "rejected", "reason": "empty_question"}
    if not full_answer:
        return copy.deepcopy(sample), {"status": "rejected", "reason": "empty_caption_answer"}

    final_sample = _canonical_as_final(sample, source="llm_postprocess")
    final_sample["answer_text"] = full_answer
    final_sample["final_question"] = question
    final_sample["final_answer_text"] = full_answer
    final_sample["messages"] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_answer},
    ]
    return final_sample, {"status": "accepted", "decision": decision}


def _merge_text_repackage(sample: Dict[str, Any], output_json: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    question = str(output_json.get("question", "")).strip()
    full_answer = str(output_json.get("full_answer", "")).strip()
    if not question:
        return copy.deepcopy(sample), {"status": "rejected", "reason": "empty_question"}
    if not full_answer:
        return copy.deepcopy(sample), {"status": "rejected", "reason": "empty_full_answer"}

    final_sample = _canonical_as_final(sample, source="llm_postprocess")
    final_sample["final_question"] = question
    final_sample["final_answer_text"] = full_answer
    final_sample["messages"] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": final_sample["final_answer_text"]},
    ]
    return final_sample, {"status": "accepted"}


def _canonical_as_final(sample: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    # 把 canonical sample 包装成最终输出格式。
    final_sample = copy.deepcopy(sample)
    for key in [
        "postprocess_disposition",
        "postprocess_job_id",
        "postprocess_mode",
        "postprocess_requires_visual",
        "postprocess_fallback_policy",
        "postprocess_reason",
    ]:
        final_sample.pop(key, None)
    final_sample["finalization_source"] = source
    final_sample["final_question"] = sample["canonical_question"]
    final_sample["final_answer_text"] = sample["answer_text"]
    return final_sample


def _schema_name(job_id: str) -> str:
    return job_id.replace(":", "_").replace("-", "_")


def _json_schema_for_expected_output(expected_output: Dict[str, Any]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for key, type_name in expected_output.items():
        required.append(key)
        properties[key] = {"type": "string"} if type_name == "string" else {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _normalize_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(str(value).strip().lower().split())


def _semantic_answer_match(sample: Dict[str, Any], answer_core: str) -> bool:
    expected = _normalize_text(sample.get("canonical_answer"))
    received = _normalize_text(answer_core)
    if received == expected:
        return True
    if sample["task_family"] == "caption":
        return bool(received)
    if sample["task_family"] == "grounding":
        return received == _normalize_text(sample.get("canonical_answer"))
    if sample["answer_type"] in {"boolean"}:
        normalized_received = "yes" if received == "true" else "no" if received == "false" else received
        return normalized_received == expected
    return False


def _parse_int(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _count_answer_mentions_value(full_answer: str, verified_count: int) -> bool:
    numbers = re.findall(r"-?\d+", full_answer)
    return str(verified_count) in numbers
