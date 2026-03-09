#!/usr/bin/env python3
"""Start a local vLLM OpenAI-compatible API server for dataset generation.

This avoids reliance on unstable remote LLM endpoints.  Once the server
is running, point 01_generate_dataset.py at it:

    python3.12 01_generate_dataset.py \
        --model "hosted_vllm/<MODEL>" \
        --url http://localhost:8000/v1 \
        --token dummy \
        ...
"""

import argparse
import os
import shutil
import subprocess
import sys


PRESETS = {
    "7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": None,
        "description": "7B FP16 — ~16 GB VRAM, good quality",
    },
    "14b": {
        "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "quantization": "awq",
        "description": "14B 4-bit AWQ — ~10 GB VRAM, better quality",
    },
}


def _check_gpu():
    """Quick sanity check that an NVIDIA GPU is visible."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True,
        )
        for line in out.strip().splitlines():
            print(f"  GPU detected: {line.strip()}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Start a local vLLM OpenAI-compatible API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Presets (use --preset instead of --model):",
            *(f"  {k:6s}  {v['description']}" for k, v in PRESETS.items()),
        ]),
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", choices=PRESETS.keys(),
                       help="Use a predefined model configuration")
    group.add_argument("--model", type=str,
                       help="HuggingFace model ID (overrides --preset)")

    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq"],
                        help="Quantization method (required for 4-bit models)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the API server (default: 8000)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum context length (default: 4096)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use (default: 0.90)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    args = parser.parse_args()

    # Resolve model from preset or explicit flag
    if args.model:
        model = args.model
        quantization = args.quantization
    elif args.preset:
        preset = PRESETS[args.preset]
        model = preset["model"]
        quantization = args.quantization or preset["quantization"]
        print(f"Using preset '{args.preset}': {preset['description']}")
    else:
        preset = PRESETS["7b"]
        model = preset["model"]
        quantization = args.quantization or preset["quantization"]
        print(f"No model specified — defaulting to preset '7b': {preset['description']}")

    # GPU check
    print("\nChecking GPU availability...")
    if not _check_gpu():
        print("ERROR: No NVIDIA GPU detected. vLLM requires a CUDA-capable GPU.",
              file=sys.stderr)
        sys.exit(1)

    # Verify vllm is installed
    if not shutil.which("vllm"):
        print("ERROR: 'vllm' command not found. Install with: pip install vllm",
              file=sys.stderr)
        sys.exit(1)

    # Build the vllm serve command
    cmd = [
        "vllm", "serve", model,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    if quantization:
        cmd += ["--quantization", quantization]

    print(f"\nStarting vLLM server on port {args.port}...")
    print(f"Model: {model}")
    if quantization:
        print(f"Quantization: {quantization}")
    print(f"Command: {' '.join(cmd)}")

    litellm_prefix = "hosted_vllm" if quantization else "hosted_vllm"
    print(f"\n--- Use with 01_generate_dataset.py ---")
    print(f"  --model \"{litellm_prefix}/{model}\"")
    print(f"  --url http://localhost:{args.port}/v1")
    print(f"  --token dummy")
    print()

    # Replace current process with vllm serve
    os.execvp("vllm", cmd)


if __name__ == "__main__":
    main()
