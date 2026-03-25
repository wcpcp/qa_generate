from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_CACHE_DIR = Path(os.environ.get("ERP_DATA_GENERATION_CACHE_DIR", "/private/tmp/erp_data_generation_cache"))


class ProviderExecutionError(RuntimeError):
    # 对外部 provider 执行失败做统一封装，便于上层捕获和汇总。
    pass


class OpenAIResponsesProvider:
    # 一个统一的 OpenAI-compatible provider 封装。
    # 当前同时支持 responses 和 chat/completions 两类接口。
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 120,
        max_retries: int = 2,
        store: bool = False,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        # 这里同时兼容 OPENAI_API_KEY 和 SL_KEY，方便本地用不同中转直接切换。
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SL_KEY")
            or "sk-nojnwinyjonehaigutukryqdgcorfqkrthirupcopvcxujhv"
        )
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.store = store
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.api_key:
            raise ProviderExecutionError("No API key is set for provider execution.")

    def run_structured_prompt(
        self,
        *,
        prompt_text: str,
        schema_name: str,
        schema: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 执行一个“要求结构化 JSON 输出”的纯文本 prompt。
        return self._run_structured_request(
            payload=_build_payload(
                endpoint_mode=_endpoint_mode(self.base_url),
                model=self.model,
                prompt_text=prompt_text,
                schema_name=schema_name,
                schema=schema,
                store=self.store,
            ),
            metadata=metadata,
        )

    def run_structured_messages(
        self,
        *,
        messages: List[Dict[str, Any]],
        schema_name: str,
        schema: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 执行一个“要求结构化 JSON 输出”的 messages 请求。
        # 这条接口主要给后续 visual verification / multimodal job 使用。
        endpoint_mode = _endpoint_mode(self.base_url)
        if endpoint_mode == "responses":
            raise ProviderExecutionError("Structured multimodal messages currently require a chat/completions endpoint.")
        payload = {
            "model": self.model,
            "messages": _coerce_chat_messages(messages, schema_name, schema),
            "temperature": 0.2,
            "max_tokens": 512,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        }
        return self._run_structured_request(payload=payload, metadata=metadata)

    def _run_structured_request(self, *, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # 统一处理缓存、重试、输出解析，供 prompt 和 messages 两种调用路径复用。
        endpoint_mode = _endpoint_mode(self.base_url)
        url = _request_url(self.base_url, endpoint_mode)
        cache_path = self._cache_path(url, payload)
        if self.use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        attempts = 0
        last_error: Optional[str] = None
        while attempts <= self.max_retries:
            attempts += 1
            try:
                response_json = self._post_json(url, payload)
                output_text = _extract_output_text(response_json, endpoint_mode)
                parsed = _loads_json_output(output_text)
                result = {
                    "provider": "openai_compatible",
                    "endpoint_mode": endpoint_mode,
                    "model": _response_model(response_json, self.model),
                    "response_id": response_json.get("id"),
                    "output_text": output_text,
                    "output_json": parsed,
                    "usage": response_json.get("usage"),
                    "metadata": metadata or {},
                    "raw_response": response_json,
                    "cache_key": cache_path.stem,
                    "request_url": url,
                }
                if self.use_cache:
                    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                return result
            except (ProviderExecutionError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempts > self.max_retries:
                    break
                time.sleep(min(2 ** (attempts - 1), 4))

        raise ProviderExecutionError(last_error or "Structured prompt execution failed.")

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 对 chat/completions 优先使用 curl，以绕开某些 Python urllib 的超时问题。
        if url.rstrip("/").endswith("/chat/completions"):
            try:
                return self._post_json_with_curl(url, payload)
            except ProviderExecutionError as exc:
                if "curl is not available" not in str(exc):
                    raise

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderExecutionError(f"HTTP {exc.code}: {body}")
        except urllib.error.URLError as exc:
            raise ProviderExecutionError(f"Network error: {exc}")

    def _post_json_with_curl(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # 用 curl 执行实际请求，适合作为一些 OpenAI-compatible 中转的稳妥路径。
        payload_text = json.dumps(payload, ensure_ascii=False)
        command = [
            "curl",
            "-sS",
            "-m",
            str(self.timeout_seconds),
            "-H",
            f"Authorization: Bearer {self.api_key}",
            "-H",
            "Content-Type: application/json",
            "-X",
            "POST",
            url,
            "--data-binary",
            "@-",
            "-w",
            "\n%{http_code}",
        ]
        try:
            completed = subprocess.run(command, input=payload_text, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise ProviderExecutionError(f"curl is not available: {exc}")
        except OSError as exc:
            raise ProviderExecutionError(f"curl request failed before execution: {exc}")

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise ProviderExecutionError(f"curl request failed: {stderr}")

        stdout = completed.stdout.rstrip()
        if not stdout:
            raise ProviderExecutionError("curl request returned an empty response.")

        body, _, status_code = stdout.rpartition("\n")
        if not body:
            body = status_code
            status_code = ""
        if status_code and status_code.isdigit() and int(status_code) >= 400:
            raise ProviderExecutionError(f"HTTP {status_code}: {body}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderExecutionError(f"Invalid JSON response: {exc}: {body[:500]}")

    def _cache_path(self, url: str, payload: Dict[str, Any]) -> Path:
        # 用请求 URL + payload 生成稳定 cache key。
        cache_key = hashlib.sha256(
            json.dumps({"url": url, "payload": payload}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.json"


def _endpoint_mode(base_url: str) -> str:
    # 根据 base_url 判断要走 responses 还是 chat/completions。
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return "responses"
    return "chat_completions"


def _request_url(base_url: str, endpoint_mode: str) -> str:
    # 在给定 endpoint mode 下，组装最终请求 URL。
    normalized = base_url.rstrip("/")
    if endpoint_mode == "responses":
        return normalized
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _build_payload(
    *,
    endpoint_mode: str,
    model: str,
    prompt_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    store: bool,
) -> Dict[str, Any]:
    # 针对不同 endpoint 模式生成对应的请求 payload。
    if endpoint_mode == "responses":
        return {
            "model": model,
            "input": prompt_text,
            "store": store,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }

    return {
        "model": model,
        "messages": _chat_messages(prompt_text, schema_name, schema),
        "temperature": 0.2,
        "max_tokens": 256,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }


def _chat_messages(prompt_text: str, schema_name: str, schema: Dict[str, Any]) -> list[Dict[str, str]]:
    # 对 chat/completions 模式，显式在 system prompt 中要求“只返回 JSON”。
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one valid JSON object and nothing else. "
                f"The object must conform to schema '{schema_name}'.\n"
                f"JSON schema:\n{schema_json}"
            ),
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]


def _coerce_chat_messages(messages: List[Dict[str, Any]], schema_name: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 在已有 messages 前注入一个 JSON-only 的 system message，并保留用户给出的多模态内容。
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    system_message = {
        "role": "system",
        "content": (
            "Return exactly one valid JSON object and nothing else. "
            f"The object must conform to schema '{schema_name}'.\n"
            f"JSON schema:\n{schema_json}"
        ),
    }
    return [system_message] + list(messages)


def _extract_output_text(response_json: Dict[str, Any], endpoint_mode: str) -> str:
    # 从 provider 原始返回中抽取真正的文本输出。
    if endpoint_mode == "responses":
        output = response_json.get("output", [])
        texts = []
        for item in output:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
        if texts:
            return "\n".join(texts).strip()
        if "output_text" in response_json and isinstance(response_json["output_text"], str):
            return response_json["output_text"].strip()
        raise ProviderExecutionError("No output_text found in provider response.")

    choices = response_json.get("choices", [])
    if not choices:
        raise ProviderExecutionError("No choices found in chat completion response.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
        content = "\n".join(text_chunks)
    if not isinstance(content, str) or not content.strip():
        raise ProviderExecutionError("No text content found in chat completion response.")
    return content.strip()


def _response_model(response_json: Dict[str, Any], fallback: str) -> str:
    # provider 不返回 model 时，回退到请求时的 model。
    return response_json.get("model", fallback)


def _loads_json_output(output_text: str) -> Dict[str, Any]:
    # 将模型返回的文本尽量稳健地解析为 JSON object。
    # 兼容代码块包裹、前后多余文本等情况。
    text = output_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ProviderExecutionError("Structured output is not a JSON object.")
    return parsed
