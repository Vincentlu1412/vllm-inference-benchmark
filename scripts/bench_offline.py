from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_ATOMS = [
    "GPU memory bandwidth and KV cache locality dominate decode performance.",
    "PagedAttention reduces memory fragmentation for serving workloads.",
    "Prefill is compute-heavy while autoregressive decode is often memory-bound.",
    "Benchmarking should separate throughput, average latency, and tail latency.",
    "Architecture feedback requires reproducible traces and parameter sweeps.",
]


@dataclass(frozen=True)
class BenchCase:
    name: str
    batch_size: int
    prompt_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def build_prompt(tokenizer: Any, target_tokens: int, salt: int) -> str:
    random.seed(salt)
    text = " ".join(random.choice(PROMPT_ATOMS) for _ in range(max(8, target_tokens // 8)))
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += " " + random.choice(PROMPT_ATOMS)
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def normalize_cases(config: dict[str, Any]) -> list[BenchCase]:
    cases = []
    for item in config["cases"]:
        cases.append(
            BenchCase(
                name=item["name"],
                batch_size=int(item["batch_size"]),
                prompt_tokens=int(item["prompt_tokens"]),
                max_new_tokens=int(item["max_new_tokens"]),
                temperature=float(item.get("temperature", 0.0)),
                top_p=float(item.get("top_p", 1.0)),
            )
        )
    return cases


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def count_output_tokens(tokenizer: Any, outputs: list[str]) -> int:
    return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in outputs)


def run_vllm_case(model: Any, tokenizer: Any, case: BenchCase, prompts: list[str]) -> dict[str, Any]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=case.temperature,
        top_p=case.top_p,
        max_tokens=case.max_new_tokens,
    )
    synchronize_cuda()
    start = time.perf_counter()
    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    synchronize_cuda()
    elapsed = time.perf_counter() - start
    texts = [item.outputs[0].text for item in outputs]
    output_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
    if output_tokens == 0:
        output_tokens = count_output_tokens(tokenizer, texts)
    return make_metrics(case, elapsed, output_tokens)


def run_hf_case(model: Any, tokenizer: Any, case: BenchCase, prompts: list[str]) -> dict[str, Any]:
    import torch

    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    do_sample = case.temperature > 0

    synchronize_cuda()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=do_sample,
            temperature=case.temperature if do_sample else None,
            top_p=case.top_p,
            max_new_tokens=case.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    synchronize_cuda()
    elapsed = time.perf_counter() - start

    prompt_width = encoded["input_ids"].shape[1]
    output_token_count = int(generated[:, prompt_width:].numel())
    return make_metrics(case, elapsed, output_token_count)


def make_metrics(case: BenchCase, elapsed: float, output_tokens: int) -> dict[str, Any]:
    prompt_tokens = case.batch_size * case.prompt_tokens
    total_tokens = prompt_tokens + output_tokens
    per_request_ms = (elapsed * 1000.0) / case.batch_size
    request_latencies = [per_request_ms] * case.batch_size
    return {
        "case": case.name,
        "batch_size": case.batch_size,
        "prompt_tokens_per_request": case.prompt_tokens,
        "max_new_tokens": case.max_new_tokens,
        "temperature": case.temperature,
        "top_p": case.top_p,
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests_per_s": case.batch_size / elapsed,
        "output_tokens_per_s": output_tokens / elapsed,
        "total_tokens_per_s": total_tokens / elapsed,
        "latency_ms_avg": statistics.mean(request_latencies),
        "latency_ms_p50": percentile(request_latencies, 50),
        "latency_ms_p95": percentile(request_latencies, 95),
    }


def load_vllm(model_name: str, backend_config: dict[str, Any]) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=backend_config.get("trust_remote_code", True),
    )
    llm_kwargs = {
        "model": model_name,
        "dtype": backend_config.get("dtype", "auto"),
        "trust_remote_code": backend_config.get("trust_remote_code", True),
        "tensor_parallel_size": int(backend_config.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(backend_config.get("gpu_memory_utilization", 0.9)),
    }
    return LLM(**llm_kwargs), tokenizer


def load_hf(model_name: str, backend_config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = backend_config.get("dtype", "auto")
    torch_dtype = "auto"
    if dtype_name == "float16":
        torch_dtype = torch.float16
    elif dtype_name == "bfloat16":
        torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=backend_config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=backend_config.get("device_map", "auto"),
        trust_remote_code=backend_config.get("trust_remote_code", True),
    )
    model.eval()
    return model, tokenizer


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "hf"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    warmup = int(config.get("warmup", 1))
    repeat = int(config.get("repeat", 3))
    seed = int(config.get("seed", 7))
    backend_config = dict(config.get("backend_config", {}))

    if args.backend == "vllm":
        model, tokenizer = load_vllm(args.model, backend_config)
        runner = run_vllm_case
    else:
        model, tokenizer = load_hf(args.model, backend_config)
        runner = run_hf_case

    for case in normalize_cases(config):
        prompts = [
            build_prompt(tokenizer, case.prompt_tokens, salt=seed + i)
            for i in range(case.batch_size)
        ]

        for _ in range(warmup):
            runner(model, tokenizer, case, prompts)
        gc.collect()
        synchronize_cuda()

        repeats = []
        for repeat_id in range(repeat):
            metrics = runner(model, tokenizer, case, prompts)
            metrics.update(
                {
                    "backend": args.backend,
                    "model": args.model,
                    "repeat_id": repeat_id,
                    "backend_config": backend_config,
                }
            )
            append_jsonl(args.out, metrics)
            repeats.append(metrics["output_tokens_per_s"])
            print(
                f"{args.backend} {case.name} repeat={repeat_id} "
                f"output_tok/s={metrics['output_tokens_per_s']:.2f}"
            )

        print(f"{case.name}: median output_tok/s={statistics.median(repeats):.2f}")


if __name__ == "__main__":
    main()
