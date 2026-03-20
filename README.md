# sdg-finetune-pipeline

A pipeline that converts technical documentation (AsciiDoc / PDF) to
Markdown, generates a synthetic QA dataset using a locally served LLM,
then fine-tunes a smaller model on that dataset with LoRA + SFT.

| Step | Script | Tool | Purpose |
|------|--------|------|---------|
| 0 | `00_convert_docs.py` | [pypandoc](https://github.com/JessicaTegworthy/pypandoc) / [docling](https://github.com/DS4SD/docling) | Convert `.adoc` and `.pdf` files to Markdown |
| — | `00_serve_model.py` | [vLLM](https://github.com/vllm-project/vllm) | Serve a local LLM for dataset generation |
| 1 | `01_generate_dataset.py` | [SDG Hub](https://github.com/instructlab/sdg) | Extract QA pairs from the Markdown corpus via the local LLM |
| 2 | `02_train_model.py` | [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) | Fine-tune a small model on the generated dataset (LoRA + SFT) |
| 3 | `03_test_model.py` | [PEFT](https://github.com/huggingface/peft) / [transformers](https://github.com/huggingface/transformers) | Compare base vs fine-tuned model answers |
| 4 | `04_merge_model.py` | [PEFT](https://github.com/huggingface/peft) | Merge LoRA adapter into the base model for standalone deployment |

## Target Environment

This pipeline is tested on **RHEL AI 1.5** running on an **AWS
g6.xlarge** instance (1 x NVIDIA L4 24 GB). It should work on any RHEL
or RHEL AI system with an NVIDIA GPU and `podman`.

| Component | Requirement |
|-----------|-------------|
| OS | RHEL AI 1.5 / RHEL 9 |
| GPU | NVIDIA L4 24 GB (tested) — any CUDA-capable GPU with 16+ GB VRAM |
| Container runtime | `podman` (pre-installed on RHEL AI) |
| NVIDIA GPU drivers | Pre-installed on RHEL AI |
| `nvidia-container-toolkit` | Pre-installed on RHEL AI; provides CDI for GPU passthrough |

> On non-RHEL-AI hosts, install the NVIDIA Container Toolkit following
> the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Host Setup

RHEL AI ships with Python 3.9 on the host, which is too old for this
pipeline. Everything runs inside a **UBI9** container with Python 3.12,
using podman to pass through the NVIDIA GPU.

### 1. Enable user lingering (required for remote sessions)

When you SSH into the machine and later disconnect, systemd kills all
processes belonging to your user — including podman containers. Enable
**lingering** so containers survive SSH disconnects:

```bash
sudo loginctl enable-linger $(whoami)
```

This only needs to be done once. Without it, any running container will
be stopped when you log out.

> If podman shows errors like `"invalid internal status"` after a
> disconnect, run `podman system migrate` to reset.

### 2. Build the container image

```bash
git clone https://github.com/alezzandro/sdg-finetune-pipeline.git
cd sdg-finetune-pipeline
podman build -t sdg-finetune-pipeline .
```

The `Containerfile` installs Python 3.12, system libraries, and all pip
dependencies except vLLM (which requires CUDA at install time and must
be installed inside the running container).

### 3. Create and start the container

Create a persistent, detached container with GPU access and a named
volume for the Hugging Face model cache:

```bash
podman run -d --name sdg-container \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v hf-cache:/root/.cache/huggingface \
  sdg-finetune-pipeline \
  sleep infinity
```

The container runs `sleep infinity` as PID 1, so it stays alive even
when you disconnect your exec session. The `hf-cache` volume persists
downloaded model weights across container restarts.

| Flag | Purpose |
|------|---------|
| `-d` | Run detached (in background) |
| `--name sdg-container` | Name for easy reference |
| `--device nvidia.com/gpu=all` | CDI GPU passthrough |
| `--security-opt=label=disable` | Prevent SELinux from blocking GPU access |
| `-v hf-cache:/root/.cache/huggingface` | Persist model downloads |

### 4. Connect to the container

```bash
podman exec -it sdg-container /bin/bash
```

You can disconnect and reconnect at any time — the container and all
background processes inside it keep running.

### 5. First-time setup inside the container

Install vLLM (requires GPU access, so it cannot be done at build time):

```bash
pip3.12 install vllm
```

Clone the repository:

```bash
cd /workspace
git clone https://github.com/alezzandro/sdg-finetune-pipeline.git
cd sdg-finetune-pipeline
```

Copy your source documentation (`.adoc` or `.pdf` files) into a `docs/`
directory inside the workspace.

## Quick Start

All commands below are run inside the container.

```bash
# Step 0 — convert documentation to Markdown
python3.12 00_convert_docs.py docs/ -o corpus.md

# Start the local LLM server (runs in background)
python3.12 00_serve_model.py --preset 14b
tail -f vllm_server.log
# Wait for: "INFO:     Application startup complete." (~2 minutes)
# Press Ctrl+C to stop tailing (the server keeps running)

# Step 1 — generate synthetic QA dataset (runs in background)
python3.12 01_generate_dataset.py \
  --model "hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ" \
  --url http://localhost:8000/v1 \
  --token dummy \
  --input corpus.md \
  --output dataset.csv \
  --domain "Infrastructure" \
  --outline "OpenShift Virtualization Networking" \
  --max-concurrency 4 --timeout 1800 \
  --resume --background

# Monitor progress
tail -f generate_dataset.log

# Stop the LLM server once dataset generation is done (frees GPU)
python3.12 00_serve_model.py --stop

# Step 2 — fine-tune a small model (runs in background)
python3.12 02_train_model.py \
  --dataset dataset.csv \
  --model "ibm-granite/granite-3.3-2b-instruct" \
  --output ./checkpoints \
  --system-prompt "You are an expert in OpenShift Virtualization networking." \
  --epochs 1 \
  --learning-rate 5e-5 \
  --background
tail -f train_model.log

# Step 3 — test the fine-tuned model
python3.12 03_test_model.py \
  --checkpoint ./checkpoints \
  --base-model "ibm-granite/granite-3.3-2b-instruct" \
  --question "How do I expose a VM with a Kubernetes service?" \
  --system-prompt "You are an expert in OpenShift Virtualization networking."

# Step 4 — merge LoRA adapter into a standalone model
python3.12 04_merge_model.py --checkpoint ./checkpoints --output ./merged-model
```

## Step 0: Convert Documentation to Markdown

`00_convert_docs.py` converts `.adoc` (AsciiDoc) and `.pdf` files to
Markdown. AsciiDoc files are converted using **pypandoc** and PDF files
using **docling**. A compatible version of `pandoc` (>= 2.15) is
automatically downloaded on first run if the system version is missing
or too old.

By default all input files are merged into a single Markdown file,
which is what step 1 expects.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `inputs` (positional) | Yes | | One or more files or directories to convert (directories are scanned recursively) |
| `-o` / `--output` | No | `corpus.md` | Output file path (or directory when using `--no-merge`) |
| `--no-merge` | No | | Write individual `.md` files to the output directory instead of merging |
| `--separator` | No | `\n\n---\n\n` | Text separator between documents when merging |

### Examples

```bash
python3.12 00_convert_docs.py docs/ -o corpus.md

python3.12 00_convert_docs.py guide.adoc appendix.pdf -o corpus.md

python3.12 00_convert_docs.py docs/ --no-merge -o converted/
```

## Serving a Local LLM

`00_serve_model.py` starts a [vLLM](https://github.com/vllm-project/vllm)
server in the **background** that exposes an OpenAI-compatible API.
The server writes logs to `vllm_server.log`.

> **Important:** The local LLM and fine-tuning (step 2) both need the
> GPU. Stop the vLLM server with `--stop` before running step 2.

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
# Start with the 14B preset (recommended for quality)
python3.12 00_serve_model.py --preset 14b

# Watch the log until ready (~2 minutes)
tail -f vllm_server.log

# Check server status
python3.12 00_serve_model.py --status

# Stop the server (frees GPU for training)
python3.12 00_serve_model.py --stop
```

> **Tip:** The `--model` value for `01_generate_dataset.py` must be
> prefixed with `hosted_vllm/` followed by the exact model name served
> by vLLM. The `--token` can be any non-empty string (vLLM doesn't
> authenticate by default).

## Step 1: Generate a Synthetic Dataset

`01_generate_dataset.py` reads a Markdown document, chunks it, and
sends each chunk to the local LLM through an SDG Hub flow to produce
question-answer pairs.

The default **Key Facts** flow decomposes each chunk into atomic facts
and generates multiple QA pairs per fact, producing a rich training
dataset without requiring hand-crafted in-context learning examples.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | | LLM identifier (e.g. `hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ`) |
| `--url` | Yes | | OpenAI-compatible API base URL |
| `--token` | Yes | | API key / bearer token (use `dummy` for local vLLM) |
| `--input` | Yes | | Path to the source Markdown document |
| `--output` | No | `dataset.csv` | Output CSV path |
| `--domain` | No | `General` | Knowledge domain label |
| `--outline` | No | | Short topic description |
| `--flow` | No | `Key Facts Knowledge Tuning Dataset Generation Flow` | SDG Hub flow name |
| `--max-chunk-chars` | No | `2500` | Max characters per document chunk |
| `--max-concurrency` | No | `10` | Max concurrent LLM requests (use `2`–`4` for local GPU) |
| `--timeout` | No | `600` | Per-request timeout in seconds (use `1800` for local GPU) |
| `--keep-cot` | No | | Keep reasoning / chain-of-thought tags in output |
| `--batch-size` | No | `25` | Chunks per batch; results are checkpointed after each batch |
| `--resume` | No | | Resume from the last checkpoint instead of starting over |
| `--background` | No | | Run in background with output logged to `generate_dataset.log` |
| `--status` | No | | Show the status of a background generation process |
| `--stop` | No | | Stop a running background generation process |

### Examples

```bash
# Run dataset generation in background with resume support
python3.12 01_generate_dataset.py \
  --model "hosted_vllm/Qwen/Qwen2.5-14B-Instruct-AWQ" \
  --url http://localhost:8000/v1 \
  --token dummy \
  --input corpus.md \
  --output dataset.csv \
  --domain "Infrastructure" \
  --outline "OpenShift Virtualization Networking" \
  --max-concurrency 4 --timeout 1800 \
  --resume --background

# Monitor progress
tail -f generate_dataset.log

# Check status
python3.12 01_generate_dataset.py --status

# Stop if needed
python3.12 01_generate_dataset.py --stop
```

> **Resilient processing:** The script processes chunks in batches
> (default 25). After each batch, results are saved to a checkpoint
> file. If the process fails or is stopped, re-run the same command
> with `--resume` to continue from the last completed batch.
>
> **Local GPU tuning:** When using the local vLLM server on a single
> GPU, set `--max-concurrency 2`–`4` to avoid GPU contention and
> `--timeout 1800` to allow for slower local inference.

## Step 2: Fine-Tune a Model

`02_train_model.py` converts the CSV from step 1 to JSONL messages
format and fine-tunes a small model using Training Hub's LoRA + SFT
algorithm (backed by Unsloth for speed).

On the L4 (24 GB), QLoRA 4-bit quantization is enabled by default,
which lets you fine-tune models up to ~7 B parameters comfortably.

> **Avoiding catastrophic forgetting:** When fine-tuning instruct
> models, use a low learning rate (`5e-5`) and fewer epochs (`1`–`2`).
> Aggressive settings (e.g. `2e-4` / `3` epochs on small datasets) can
> cause the model to overfit and lose its general instruction-following
> ability, producing shorter and less coherent answers than the original
> model.

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
| `--background` | No | | Run in background with output logged to `train_model.log` |
| `--status` | No | | Show the status of a background training process |
| `--stop` | No | | Stop a running background training process |

### Recommended Models for the L4 24 GB

> **Tip:** Always prefer an **instruct** model over a base model.
> Instruct models already know how to follow instructions, so
> fine-tuning only needs to add domain knowledge.

| Model | Params | Notes |
|-------|--------|-------|
| `ibm-granite/granite-3.3-2b-instruct` | 2 B | Recommended — instruction-tuned, strong domain adaptation |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | Strong multilingual instruct model |
| `ibm-granite/granite-4.0-1b` | 2 B | Base model only — fast training but requires more data |
| `ibm-granite/granite-4.0-350m` | 0.4 B | Ultra-lightweight base model |

With QLoRA 4-bit, larger models (7 B+) also fit in 24 GB.

### Examples

```bash
# Run training in background
python3.12 02_train_model.py \
  --dataset dataset.csv \
  --model "ibm-granite/granite-3.3-2b-instruct" \
  --output ./checkpoints \
  --system-prompt "You are an expert in OpenShift Virtualization networking." \
  --epochs 1 \
  --learning-rate 5e-5 \
  --background

# Monitor progress
tail -f train_model.log

# Check status
python3.12 02_train_model.py --status

# Stop if needed
python3.12 02_train_model.py --stop
```

## Step 3: Test the Fine-Tuned Model

`03_test_model.py` loads a reference model and the LoRA-adapted model,
sends the same question to each, and prints the answers side by side so
you can see the effect of fine-tuning.

By default the reference model is auto-detected from the checkpoint's
`adapter_config.json`. Use `--base-model` to override it — for example,
to compare against the original instruct model before fine-tuning.

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
python3.12 03_test_model.py \
  --checkpoint ./checkpoints \
  --base-model "ibm-granite/granite-3.3-2b-instruct" \
  --question "How do I expose a VM with a Kubernetes service?" \
  --system-prompt "You are an expert in OpenShift Virtualization networking."

python3.12 03_test_model.py \
  --checkpoint ./checkpoints \
  --base-model "ibm-granite/granite-3.3-2b-instruct" \
  --question "What is the difference between masquerade and bridge networking?" \
  --system-prompt "You are an expert in OpenShift Virtualization networking."

python3.12 03_test_model.py \
  --checkpoint ./checkpoints \
  --question "Explain how to configure SR-IOV for a virtual machine." \
  --max-new-tokens 512
```

## Step 4: Merge and Export the Model

`04_merge_model.py` merges the LoRA adapter into the base model weights
and saves a standalone model directory. The merged model can be loaded
directly with `transformers` or served via an inference engine like
vLLM without requiring PEFT at inference time.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | No | `./checkpoints` | Path to the LoRA checkpoint directory |
| `--output` | No | `./merged-model` | Output directory for the merged model |
| `--no-quantize` | No | | Disable 4-bit quantization when loading |

### Examples

```bash
python3.12 04_merge_model.py

python3.12 04_merge_model.py \
  --checkpoint ./checkpoints \
  --output ./my-merged-model
```

## Managing the Container

### Reconnecting after SSH disconnect

The container keeps running after you disconnect (thanks to `loginctl
enable-linger` and the `sleep infinity` entrypoint). To reconnect:

```bash
ssh cloud-user@<your-host>
podman exec -it sdg-container /bin/bash
cd /workspace/sdg-finetune-pipeline
```

Your background processes (vLLM server, dataset generation) continue
running inside the container between sessions.

### Useful commands

```bash
# Check container status
podman ps

# Check vLLM server status
python3.12 00_serve_model.py --status

# Check dataset generation status
python3.12 01_generate_dataset.py --status

# Check training status
python3.12 02_train_model.py --status

# Stop everything and remove the container
python3.12 02_train_model.py --stop
python3.12 01_generate_dataset.py --stop
python3.12 00_serve_model.py --stop
exit
podman stop sdg-container
podman rm sdg-container
```

### Recovering from podman errors

If podman shows `"invalid internal status"` after a session disconnect
(usually means lingering was not enabled):

```bash
podman system migrate
podman start sdg-container
podman exec -it sdg-container /bin/bash
```

## Project Structure

```
00_convert_docs.py      # Step 0 — document conversion (.adoc / .pdf -> .md)
00_serve_model.py       # Serve a local LLM with vLLM
01_generate_dataset.py  # Step 1 — synthetic data generation
02_train_model.py       # Step 2 — LoRA + SFT fine-tuning
03_test_model.py        # Step 3 — compare base vs fine-tuned model answers
04_merge_model.py       # Step 4 — merge LoRA adapter and export standalone model
Containerfile           # UBI9 container image with pipeline dependencies
README.md
```

## License

Apache-2.0
