from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PACKAGES = ["torch", "transformers", "vllm", "openai", "numpy", "psutil"]


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def run_json_command(args: list[str]) -> dict:
    try:
        completed = subprocess.run(args, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"available": True, "raw": completed.stdout.strip()}


def collect_nvidia_smi() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"available": False}

    query = (
        "index,name,driver_version,memory.total,memory.used,temperature.gpu,"
        "power.draw,power.limit,clocks.sm,clocks.mem,compute_cap"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        return {"available": False, "error": exc.stderr.strip() or str(exc)}

    gpus = []
    for line in completed.stdout.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 11:
            continue
        gpus.append(
            {
                "index": values[0],
                "product_name": values[1],
                "driver_version": values[2],
                "memory_total_mib": values[3],
                "memory_used_mib": values[4],
                "temperature_c": values[5],
                "power_draw_w": values[6],
                "power_limit_w": values[7],
                "sm_clock_mhz": values[8],
                "mem_clock_mhz": values[9],
                "compute_capability": values[10],
            }
        )
    return {"available": True, "gpu_count": len(gpus), "gpus": gpus}


def collect_torch() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": str(exc)}

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_mib": props.total_memory // (1024 * 1024),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )

    return {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
    }


def collect_psutil() -> dict:
    try:
        import psutil
    except ImportError:
        return {"available": False}

    memory = psutil.virtual_memory()
    return {
        "available": True,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_total_gib": round(memory.total / (1024**3), 2),
        "memory_available_gib": round(memory.available / (1024**3), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/env.json"))
    args = parser.parse_args()

    payload = {
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(),
        "torch": collect_torch(),
        "nvidia_smi": collect_nvidia_smi(),
        "psutil": collect_psutil(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
