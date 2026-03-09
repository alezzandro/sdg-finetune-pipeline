# sdg-finetune-pipeline

A three-step pipeline that converts technical documentation (AsciiDoc / PDF)
to Markdown, generates a synthetic QA dataset from it using a large LLM
(remote or locally served), then fine-tunes a smaller local model on that
dataset with LoRA + SFT.

| Step | Script | Tool | Purpose |
|------|--------|------|---------|
| 0 | `00_convert_docs.py` | [pypandoc](https://github.com/JessicaTegworthy/pypandoc) / [docling](https://github.com/DS4SD/docling) | Convert `.adoc` and `.pdf` files to Markdown |
| — | `00_serve_model.py` | [vLLM](https://github.com/vllm-project/vllm) | (Optional) Serve a local LLM for dataset generation |
| 1 | `01_generate_dataset.py` | [SDG Hub](https://github.com/instructlab/sdg) | Extract QA pairs from the Markdown corpus via an LLM |
| 2 | `02_train_model.py` | [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) | Fine-tune a small model on the generated dataset (LoRA + SFT) |
| 3 | `03_test_model.py` | [PEFT](https://github.com/huggingface/peft) / [transformers](https://github.com/huggingface/transformers) | Compare base vs fine-tuned model answers |
| 4 | `04_merge_model.py` | [PEFT](https://github.com/huggingface/peft) | Merge LoRA adapter into the base model for standalone deployment |

## Target Environment

This pipeline is designed for **RHEL AI 1.5** on an **AWS g6.xlarge** instance
(1 x NVIDIA L4 24 GB), though it works on any CUDA-capable machine.

| Component | Requirement |
|-----------|-------------|
| OS | RHEL AI 1.5 (or any Linux with CUDA) |
| GPU | NVIDIA L4 24 GB (or equivalent) |
| Python | 3.12 |
| System | `pandoc` auto-downloaded by the script if missing or too old |
| LLM for step 1 | Any OpenAI-compatible endpoint — remote or local (see `00_serve_model.py`) |

## Quick Start

```bash
# Create a virtual environment
python3.12 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install sdg-hub
pip install training-hub[lora]
pip install pypandoc docling
pip install vllm            # optional — only needed for local LLM serving

# Step 0 — convert AsciiDoc docs to a single Markdown file
# (pandoc is auto-downloaded on first run if missing or too old)
python 00_convert_docs.py docs/ -o corpus.md

# Step 1 — generate dataset
# Option A: use a remote LLM endpoint
python 01_generate_dataset.py \
  --model "openai/qwen3-14b" \
  --url "$LLM_ENDPOINT" \
  --token "$API_KEY" \
  --input corpus.md \
  --output dataset.csv \
  --domain "Infrastructure" \
  --outline "OpenShift Virtualization Networking"

# Option B: serve a local LLM (starts in background, logs to vllm_server.log)
#   python 00_serve_model.py --preset 7b
#   tail -f vllm_server.log  # wait for "Application startup complete."
# then generate using the local server:
#   python 01_generate_dataset.py \
#     --model "hosted_vllm/Qwen/Qwen2.5-7B-Instruct" \
#     --url http://localhost:8000/v1 --token dummy \
#     --input corpus.md --output dataset.csv \
#     --domain "Infrastructure" --outline "OpenShift Virtualization Networking"
# stop when done: python 00_serve_model.py --stop

# Step 2 — fine-tune a small model on the GPU
python 02_train_model.py \
  --dataset dataset.csv \
  --model "ibm-granite/granite-3.3-2b-instruct" \
  --output ./checkpoints \
  --system-prompt "You are an expert in OpenShift Virtualization networking." \
  --epochs 1 \
  --learning-rate 5e-5 \
  --max-seq-len 512
```

## Running in a Container (Podman + UBI9)

RHEL AI ships with Python 3.9 on the host, which is too old for this pipeline.
The recommended approach is to run everything inside a **UBI9 Python 3.12**
container, using podman to pass through the NVIDIA GPU.

### Host Prerequisites

| Requirement | Notes |
|-------------|-------|
| `podman` | Pre-installed on RHEL AI |
| NVIDIA GPU drivers | Pre-installed on RHEL AI |
| `nvidia-container-toolkit` | Pre-installed on RHEL AI; provides CDI support for rootless GPU containers |

> On non-RHEL-AI hosts, install the NVIDIA Container Toolkit following the
> [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Option A: Interactive Container

Launch an interactive UBI9 container with GPU access and the project directory
mounted as a volume:

```bash
podman run -it --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v ./:/workspace:Z \
  -w /workspace \
  registry.access.redhat.com/ubi9/ubi \
  /bin/bash
```

Inside the container, install Python 3.12 and system dependencies, then create
a virtual environment:

```bash
dnf install -y git gcc python3.12 python3.12-pip python3.12-devel libxcb mesa-libGL
git clone https://github.com/alezzandro/sdg-finetune-pipeline.git && cd sdg-finetune-pipeline
python3.12 -m venv venv && source venv/bin/activate
pip3.12 install --upgrade pip
pip3.12 install sdg-hub
pip3.12 install training-hub[lora]
pip3.12 install pypandoc docling
pip3.12 install vllm            # optional — only needed for local LLM serving

# Now run the pipeline steps as shown in Quick Start
python3.12 00_convert_docs.py docs/ -o corpus.md
# ...
```

### Option B: Build a Custom Image (Containerfile)

A `Containerfile` is included in the repository with all dependencies
pre-installed (including vLLM). The build requires GPU access because
vLLM needs CUDA at install time:

```bash
podman build --device nvidia.com/gpu=all --security-opt=label=disable \
  -t sdg-finetune-pipeline .

# Steps 0–1 (no GPU needed)
podman run --rm \
  -v ./docs:/workspace/docs:Z \
  -v ./output:/workspace/output:Z \
  sdg-finetune-pipeline \
  python3.12 00_convert_docs.py docs/ -o output/corpus.md

# Step 2 (GPU required)
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v ./output:/workspace/output:Z \
  sdg-finetune-pipeline \
  python3.12 02_train_model.py \
    --dataset output/dataset.csv \
    --model ibm-granite/granite-3.3-2b-instruct \
    --output output/checkpoints
```

### GPU Passthrough Reference

| Method | Flag | When to Use |
|--------|------|-------------|
| **CDI** (recommended) | `--device nvidia.com/gpu=all` | RHEL AI / systems with `nvidia-container-toolkit` and CDI configured |
| **CDI (specific GPU)** | `--device nvidia.com/gpu=0` | Multi-GPU hosts when you want to target a single GPU |
| **OCI hooks** (legacy) | `--hooks-dir=/usr/share/containers/oci/hooks.d/` | Older `nvidia-container-toolkit` versions without CDI |
| **Direct devices** | `--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools` | Fallback when no container toolkit is installed (limited CUDA support) |

> **Tip:** On RHEL AI, `--security-opt=label=disable` prevents SELinux from
> blocking access to the GPU device nodes. It can be omitted if you configure
> an appropriate SELinux policy instead.

### Persisting Hugging Face Cache

Step 2 downloads model weights from Hugging Face. To avoid re-downloading on
every container run, mount a cache directory:

```bash
podman run -it --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v ./:/workspace:Z \
  -v hf-cache:/root/.cache/huggingface \
  -w /workspace \
  registry.access.redhat.com/ubi9/ubi \
  /bin/bash
```

This creates a named podman volume (`hf-cache`) that persists across
container restarts.

## Step 0: Convert Documentation to Markdown

`00_convert_docs.py` converts `.adoc` (AsciiDoc) and `.pdf` files to Markdown.
AsciiDoc files are converted using **pypandoc** and PDF files are converted
using **docling**. A compatible version of `pandoc` (>= 2.15) is automatically
downloaded on first run if the system version is missing or too old.

By default all input files are merged into a single Markdown file, which is
what step 1 expects.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `inputs` (positional) | Yes | | One or more files or directories to convert (directories are scanned recursively) |
| `-o` / `--output` | No | `corpus.md` | Output file path (or directory when using `--no-merge`) |
| `--no-merge` | No | | Write individual `.md` files to the output directory instead of merging |
| `--separator` | No | `\n\n---\n\n` | Text separator between documents when merging |

### Examples

```bash
# Convert a directory of .adoc files into a single corpus
python 00_convert_docs.py /path/to/openshift-docs/networking/ -o corpus.md

# Convert specific files
python 00_convert_docs.py guide.adoc appendix.pdf -o corpus.md

# Keep individual .md files
python 00_convert_docs.py docs/ --no-merge -o converted/

# Mix files and directories
python 00_convert_docs.py overview.adoc modules/ architecture.pdf -o corpus.md
```

## Serving a Local LLM (Optional)

If you don't have a reliable remote LLM endpoint, you can serve a model
locally on the GPU using `00_serve_model.py` (a thin wrapper around
[vLLM](https://github.com/vllm-project/vllm)).  The server starts in the
**background**, writes logs to `vllm_server.log`, and exposes an
OpenAI-compatible API that `01_generate_dataset.py` can use directly.

> **Note:** The local LLM and fine-tuning (step 2) both need the GPU.
> Stop the vLLM server with `--stop` before running step 2.

### Recommended Models for the L4 24 GB

| Preset | Model | Quantization | VRAM | Quality |
|--------|-------|-------------|------|---------|
| `7b` | `Qwen/Qwen2.5-7B-Instruct` | FP16 | ~16 GB | Good |
| `14b` | `Qwen/Qwen2.5-14B-Instruct-AWQ` | AWQ 4-bit | ~10 GB | Better |

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--preset` | No | `7b` | Predefined model configuration (`7b` or `14b`) |
| `--model` | No | | HuggingFace model ID (overrides `--preset`) |
| `--quantization` | No | | Quantization method: `awq` or `gptq` |
| `--port` | No | `8000` | Port for the API server |
| `--max-model-len` | No | `16384` | Maximum context length (input + output tokens) |
| `--gpu-memory-utilization` | No | `0.90` | Fraction of GPU memory to use |
| `--tensor-parallel-size` | No | `1` | Number of GPUs for tensor parallelism |
| `--stop` | No | | Stop a running vLLM server |
| `--status` | No | | Show whether the server is running and ready |

### Examples

```bash
# Start the local LLM server in background (7B preset, safe default)
python3.12 00_serve_model.py --preset 7b

# Or use the higher-quality 14B 4-bit model
python3.12 00_serve_model.py --preset 14b

# Or specify a custom model
python3.12 00_serve_model.py --model "meta-llama/Llama-3.1-8B-Instruct"

# Watch the log for the readiness message (~1–3 minutes)
tail -f vllm_server.log
# Look for: "INFO:     Application startup complete."

# Check server status
python3.12 00_serve_model.py --status

# Generate dataset once the server is ready
python3.12 01_generate_dataset.py \
  --model "hosted_vllm/Qwen/Qwen2.5-7B-Instruct" \
  --url http://localhost:8000/v1 \
  --token dummy \
  --input corpus.md \
  --output dataset.csv \
  --domain "Infrastructure" \
  --outline "OpenShift Virtualization Networking"

# Stop the server when done (frees the GPU for training)
python3.12 00_serve_model.py --stop
```

> **Tip:** The `--model` value for `01_generate_dataset.py` must be prefixed
> with `hosted_vllm/` followed by the exact model name served by vLLM.
> The `--token` can be any non-empty string (vLLM doesn't authenticate by default).

## Step 1: Generate a Synthetic Dataset

`01_generate_dataset.py` reads a Markdown document, chunks it, and sends each
chunk to an LLM (remote or locally served) through an SDG Hub flow to produce
question-answer pairs.

The default **Key Facts** flow decomposes each chunk into atomic facts and
generates multiple QA pairs per fact, producing a rich training dataset without
requiring hand-crafted in-context learning examples.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | | LLM identifier (e.g. `openai/qwen3-14b`) |
| `--url` | Yes | | OpenAI-compatible API base URL |
| `--token` | Yes | | API key / bearer token |
| `--input` | Yes | | Path to the source Markdown document |
| `--output` | No | `dataset.csv` | Output CSV path |
| `--domain` | No | `General` | Knowledge domain label |
| `--outline` | No | | Short topic description |
| `--flow` | No | `Key Facts Knowledge Tuning Dataset Generation Flow` | SDG Hub flow name |
| `--max-chunk-chars` | No | `2500` | Max characters per document chunk |
| `--max-concurrency` | No | `10` | Max concurrent LLM requests |
| `--timeout` | No | `600` | Per-request timeout in seconds |
| `--keep-cot` | No | | Keep reasoning / chain-of-thought tags in output |
| `--batch-size` | No | `25` | Chunks per batch; results are checkpointed after each batch |
| `--resume` | No | | Resume from the last checkpoint instead of starting over |

> **Resilient processing:** With large documents the script processes chunks in
> batches (default 25). After each batch, results are saved to a checkpoint
> file. If the remote LLM connection drops mid-run, re-run the same command
> with `--resume` to continue from the last completed batch.

## Step 2: Fine-Tune a Model

`02_train_model.py` converts the CSV from step 1 to JSONL messages format and
fine-tunes a small model using Training Hub's LoRA + SFT algorithm (backed by
Unsloth for speed).

On the L4 (24 GB), QLoRA 4-bit quantization is enabled by default, which lets
you fine-tune models up to ~7 B parameters comfortably.

> **Avoiding catastrophic forgetting:** When fine-tuning instruct models, use
> a low learning rate (`5e-5`) and fewer epochs (`1`–`2`). Aggressive settings
> (e.g. `2e-4` / `3` epochs on small datasets) can cause the model to overfit
> and lose its general instruction-following ability, producing shorter and
> less coherent answers than the original model.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dataset` | Yes | | Path to the CSV from step 1 |
| `--model` | No | `ibm-granite/granite-4.0-1b` | HuggingFace model ID or local path |
| `--output` | No | `./checkpoints` | Directory for the fine-tuned model |
| `--system-prompt` | No | | System prompt prepended to each training example |
| `--lora-r` | No | `16` | LoRA rank |
| `--lora-alpha` | No | `32` | LoRA alpha |
| `--epochs` | No | `3` | Number of training epochs |
| `--learning-rate` | No | `2e-4` | Learning rate |
| `--max-seq-len` | No | `512` | Max sequence length |
| `--micro-batch-size` | No | `2` | Batch size per device |
| `--gradient-accumulation-steps` | No | `4` | Gradient accumulation steps |
| `--no-quantize` | No | | Disable QLoRA 4-bit quantization |

### Recommended Models for the L4 24 GB

> **Tip:** Always prefer an **instruct** model over a base model. Instruct
> models already know how to follow instructions, so fine-tuning only needs
> to add domain knowledge — producing much better answers.

| Model | Params | Notes |
|-------|--------|-------|
| `ibm-granite/granite-3.3-2b-instruct` | 2 B | Recommended — instruction-tuned, strong domain adaptation |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | Strong multilingual instruct model |
| `ibm-granite/granite-4.0-1b` | 2 B | Base model only — fast training but requires more data |
| `ibm-granite/granite-4.0-350m` | 0.4 B | Ultra-lightweight base model |

With QLoRA 4-bit, larger models (7 B+) also fit in 24 GB.

## Step 3: Test the Fine-Tuned Model

`03_test_model.py` loads a reference model and the LoRA-adapted model, sends
the same question to each, and prints the answers side by side so you can see
the effect of fine-tuning.

By default the reference model is auto-detected from the checkpoint's
`adapter_config.json`. Use `--base-model` to override it — for example, to
compare against the original instruct model before fine-tuning.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | No | `./checkpoints` | Path to the LoRA checkpoint directory |
| `--base-model` | No | auto-detected from adapter config | HuggingFace model ID for the reference model |
| `--question` | Yes | | Question to ask both models |
| `--system-prompt` | No | | System prompt prepended to the conversation |
| `--max-new-tokens` | No | `256` | Max tokens to generate |
| `--no-quantize` | No | | Disable 4-bit quantization |

### Examples

```bash
# Compare the instruct model (before) vs fine-tuned model (after)
python 03_test_model.py \
  --checkpoint ./checkpoints \
  --base-model "ibm-granite/granite-3.3-2b-instruct" \
  --question "How do I expose a VM with a Kubernetes service?" \
  --system-prompt "You are an expert in OpenShift Virtualization networking."

# With a different question
python 03_test_model.py \
  --checkpoint ./checkpoints \
  --base-model "ibm-granite/granite-3.3-2b-instruct" \
  --question "What is the difference between masquerade and bridge networking?" \
  --system-prompt "You are an expert in OpenShift Virtualization networking."

# Longer answers, auto-detect base model
python 03_test_model.py \
  --checkpoint ./checkpoints \
  --question "Explain how to configure SR-IOV for a virtual machine." \
  --max-new-tokens 512
```

## Step 4: Merge and Export the Model

`04_merge_model.py` merges the LoRA adapter into the base model weights and
saves a standalone model directory. The merged model can be loaded directly
with `transformers` or served via an inference engine like vLLM without
requiring PEFT at inference time.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | No | `./checkpoints` | Path to the LoRA checkpoint directory |
| `--output` | No | `./merged-model` | Output directory for the merged model |
| `--no-quantize` | No | | Disable 4-bit quantization when loading |

### Examples

```bash
# Merge with default paths
python 04_merge_model.py

# Custom checkpoint and output
python 04_merge_model.py \
  --checkpoint ./checkpoints \
  --output ./my-merged-model
```

## Project Structure

```
00_convert_docs.py      # Step 0 — document conversion (.adoc / .pdf -> .md)
00_serve_model.py       # Optional — serve a local LLM with vLLM
01_generate_dataset.py  # Step 1 — synthetic data generation
02_train_model.py       # Step 2 — LoRA + SFT fine-tuning
03_test_model.py        # Step 3 — compare base vs fine-tuned model answers
04_merge_model.py       # Step 4 — merge LoRA adapter and export standalone model
Containerfile           # UBI9 container image with all dependencies
README.md
```

## License

Apache-2.0
