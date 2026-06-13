from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


PROMPT_ATOMS = [
    "Analyze GPU inference performance with queueing, prefill, decode, and KV cache effects.",
    "Separate TTFT from TPOT to explain user-perceived latency and hardware efficiency.",
    "Use p95 latency because mean latency hides scheduler and memory pressure artifacts.",
    "Relate benchmark traces to CUDA kernels, NCCL communication, and memory bandwidth.",
]


@dataclass(frozen=True)
class ServerCase:
    base_url: str
    model: str
    prompt_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    concurrency: int
    requests: int


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def make_prompt(target_words: int, salt: int) -> str:
    random.seed(salt)
    words = []
    while len(words) < target_words:
        words.extend(random.choice(PROMPT_ATOMS).split())
    return " ".join(words[:target_words])


async def one_streaming_request(
    client: httpx.AsyncClient,
    case: ServerCase,
    request_id: int,
) -> dict[str, Any]:
    # Word count is a backend-independent approximation. The response usage field gives exact
    # token counts when the OpenAI-compatible server returns it.
    prompt = make_prompt(max(16, case.prompt_tokens), request_id)
    payload = {
        "model": case.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": case.temperature,
        "top_p": case.top_p,
        "max_tokens": case.max_new_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    url = f"{case.base_url.rstrip('/')}/chat/completions"
    start = time.perf_counter()
    first_token_at: float | None = None
    chunk_times: list[float] = []
    output_chars = 0
    usage: dict[str, Any] = {}

    async with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                chunk_times.append(now)
                output_chars += len(content)

    end = time.perf_counter()
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = max(1, round(output_chars / 4))

    ttft_ms = ((first_token_at or end) - start) * 1000
    e2e_ms = (end - start) * 1000
    if len(chunk_times) > 1:
        inter_token_ms = [
            (right - left) * 1000 for left, right in zip(chunk_times, chunk_times[1:])
        ]
        tpot_ms = statistics.mean(inter_token_ms)
    else:
        tpot_ms = max(0.0, e2e_ms - ttft_ms) / max(1, completion_tokens)

    return {
        "request_id": request_id,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "output_chars": output_chars,
    }


async def run_case(case: ServerCase) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=case.concurrency, max_keepalive_connections=case.concurrency)
    timeout = httpx.Timeout(300.0)
    semaphore = asyncio.Semaphore(case.concurrency)
    start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def guarded(request_id: int) -> dict[str, Any]:
            async with semaphore:
                return await one_streaming_request(client, case, request_id)

        results = await asyncio.gather(*(guarded(i) for i in range(case.requests)))

    elapsed = time.perf_counter() - start
    ttft = [item["ttft_ms"] for item in results]
    tpot = [item["tpot_ms"] for item in results]
    e2e = [item["e2e_ms"] for item in results]
    output_tokens = sum(int(item["completion_tokens"]) for item in results)

    return {
        "backend": "vllm_server",
        "model": case.model,
        "prompt_tokens_target": case.prompt_tokens,
        "max_new_tokens": case.max_new_tokens,
        "temperature": case.temperature,
        "top_p": case.top_p,
        "concurrency": case.concurrency,
        "requests": case.requests,
        "elapsed_s": elapsed,
        "requests_per_s": case.requests / elapsed,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed,
        "ttft_ms_avg": statistics.mean(ttft),
        "ttft_ms_p50": percentile(ttft, 50),
        "ttft_ms_p95": percentile(ttft, 95),
        "tpot_ms_avg": statistics.mean(tpot),
        "tpot_ms_p50": percentile(tpot, 50),
        "tpot_ms_p95": percentile(tpot, 95),
        "e2e_ms_avg": statistics.mean(e2e),
        "e2e_ms_p50": percentile(e2e, 50),
        "e2e_ms_p95": percentile(e2e, 95),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for concurrency in args.concurrency:
        case = ServerCase(
            base_url=args.base_url,
            model=args.model,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            concurrency=concurrency,
            requests=args.requests,
        )
        record = await run_case(case)
        append_jsonl(args.out, record)
        print(
            f"concurrency={concurrency} output_tok/s={record['output_tokens_per_s']:.2f} "
            f"ttft_p95={record['ttft_ms_p95']:.1f}ms"
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
