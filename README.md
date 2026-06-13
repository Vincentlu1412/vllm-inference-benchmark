# vLLM Inference Benchmark

This repository benchmarks LLM inference latency and throughput with **vLLM** and **Hugging Face Transformers**. It is designed as a practical performance-engineering project for NVIDIA system software, GPU performance tooling, and deep-learning hardware architecture interviews.

This repository does **not** include pre-filled benchmark results. Every result should be generated on the target machine and paired with `results/env.json`, which records the actual GPU, driver, CUDA, PyTorch, vLLM, and transformers versions.

The project focuses on measurable questions:

- How do batch size, prompt length, output length, and sampling parameters affect throughput and latency?
- How does vLLM compare with `transformers.generate` on the same GPU?
- How do TTFT, TPOT, p50/p95 latency, and token throughput expose prefill/decode bottlenecks?
- What optimization attempts are meaningful on an 8 GB consumer GPU?

## Hardware Assumptions

The benchmark is hardware-aware but not hardware-specific. Run `scripts/collect_env.py` first, then interpret results based on the captured GPU and software stack.

Suggested starting points:

| GPU class | Suggested model range | Notes |
| --- | --- |
| 8 GB VRAM | 0.5B to 3B | Use modest prompt length, batch size, and `float16` |
| 16 GB VRAM | 3B to 7B | Larger context and concurrency sweeps become practical |
| 24 GB+ VRAM | 7B+ | More room for serving benchmarks, longer context, and KV cache experiments |
| Multi-GPU | 7B+ / tensor parallel | Add tensor parallel and NCCL communication analysis |

Recommended smoke-test models:

| Model | Why |
| --- | --- |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Lightweight first run |
| `Qwen/Qwen2.5-1.5B-Instruct` | Good balance for local GPU benchmarking |
| `microsoft/Phi-3-mini-4k-instruct` | More demanding small model |

## Repository Layout

```text
.
├── configs
│   ├── rtx3070ti.sweep.json
│   └── sweep.example.json
├── results
│   └── .gitkeep
├── scripts
│   ├── bench_offline.py
│   ├── bench_server.py
│   ├── collect_env.py
│   └── summarize.py
├── .gitignore
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## Setup with Anaconda

Create the environment:

```bash
conda env create -f environment.yml
conda activate vllm-bench
```

Install PyTorch for your CUDA version from the official PyTorch command selector if needed. For many CUDA 12.1 setups:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For vLLM on Windows, use WSL2 Ubuntu if native install fails. Hugging Face baseline can often run in native Windows Conda, while vLLM support is usually smoother on Linux/WSL2:

```bash
wsl --install -d Ubuntu
```

Then repeat the Conda setup inside WSL2.

## Collect Environment

Run this first and keep the output with the benchmark results:

```bash
python scripts/collect_env.py --out results/env.json
```

The generated `results/env.json` records:

- GPU model, driver, memory, clocks, power limit
- Python, PyTorch, CUDA, vLLM, transformers versions
- CPU core count and system memory

For interview use, summarize the environment like this:

| Field | Source |
| --- | --- |
| GPU | `results/env.json -> nvidia_smi.gpus[].product_name` |
| VRAM | `results/env.json -> nvidia_smi.gpus[].memory_total_mib` |
| Driver / CUDA | `results/env.json -> nvidia_smi.driver_version`, `torch.cuda_version` |
| Precision | benchmark config `backend_config.dtype` |
| Model | benchmark command `--model` |
| Main metrics | output tok/s, requests/s, p50/p95 latency, TTFT, TPOT |

## Offline Benchmark

vLLM:

```bash
python scripts/bench_offline.py \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --config configs/sweep.example.json \
  --out results/vllm_offline.jsonl
```

Hugging Face baseline:

```bash
python scripts/bench_offline.py \
  --backend hf \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --config configs/sweep.example.json \
  --out results/hf_offline.jsonl
```

The script reports:

- `requests_per_s`
- `output_tokens_per_s`
- `total_tokens_per_s`
- `latency_ms_avg`
- `latency_ms_p50`
- `latency_ms_p95`
- prompt/output token counts

## vLLM Server Benchmark

Start the vLLM OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enable-prefix-caching
```

Run the streaming benchmark:

```bash
python scripts/bench_server.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-tokens 256 \
  --max-new-tokens 64 \
  --concurrency 1 2 4 8 \
  --requests 32 \
  --out results/server_stream.jsonl
```

Server metrics:

- `ttft_ms_p50/p95`: time to first token, usually dominated by queueing and prefill
- `tpot_ms_p50/p95`: time per output token, useful for decode efficiency
- `e2e_ms_p50/p95`: end-to-end request latency
- `output_tokens_per_s`: aggregate serving throughput

## Summarize Results

```bash
python scripts/summarize.py results/*.jsonl --out results/summary.md
```

Commit `results/summary.md` and selected plots/tables only after running the benchmark on a real target machine. Keep raw `.jsonl` files if they are small enough; otherwise include a short sampled result table in the README.

## Optimization Experiments

Good experiments:

| Direction | Parameters | Expected Observation |
| --- | --- | --- |
| Batch size | `batch_size=1/2/4/8` | Throughput improves until VRAM pressure or scheduling overhead dominates |
| Prompt length | `128/256/512/1024` tokens | Longer prompts increase prefill time and TTFT |
| Output length | `32/64/128` tokens | Longer outputs make decode throughput more representative |
| Server concurrency | `1/2/4/8` | Higher concurrency improves utilization but increases p95 latency |
| Prefix caching | on/off with repeated prompts | TTFT improves when prompts share a prefix |
| GPU memory utilization | `0.75/0.85/0.90` | Higher values allow larger KV cache but may risk OOM |
| Tensor parallel | `1/2/4`, if multi-GPU is available | Larger models become possible, but communication can hurt decode latency |

`configs/rtx3070ti.sweep.json` is included only as an example of an 8 GB VRAM profile. It is not a claimed test environment or result.

## Interview Talking Points

Application level:

- The benchmark sweeps prompt length, output length, batch size, sampling parameters, and serving concurrency.
- It reports p50/p95 instead of only average latency.
- It compares vLLM with a Hugging Face baseline under the same workload.

System level:

- Prefill is compute-heavy and writes KV cache.
- Decode is often memory-bandwidth and KV-cache-read heavy.
- TTFT and TPOT separate user-facing latency from steady-state decode cost.
- vLLM improves serving efficiency through batching, KV cache management, and scheduling.

Architecture level:

- Important constraints include VRAM capacity, memory bandwidth, kernel launch overhead, SM occupancy, and cache locality.
- For larger deployments, NCCL communication, tensor parallelism, scheduling, and storage/network input pipelines become visible.
- These measurements can become traces for simulator validation or architecture feedback.

Role mapping:

- `JR2015967`: workload modeling, accelerator performance trade-offs, and optimization feedback to architecture.
- `JR1996042`: benchmarking, profiling, debugging, and AI inference efficiency tooling.
- `JR2016503`: connecting application traces, runtime behavior, GPU kernels, and performance models.

## GitHub Upload

Suggested repository name:

```text
vllm-inference-benchmark
```

Suggested short description:

```text
vLLM and Hugging Face inference benchmarking with latency, throughput, TTFT, and TPOT analysis.
```

After uploading, pin the repository on the GitHub profile and add a short result table after running the benchmark on the local GPU.

## References

- vLLM Quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
- vLLM Benchmarking: https://docs.vllm.ai/en/latest/benchmarking/
- Hugging Face text generation: https://huggingface.co/docs/transformers/en/main_classes/text_generation
