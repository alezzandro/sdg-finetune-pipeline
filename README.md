# sdg-finetune-pipeline

A three-step pipeline that converts technical documentation (AsciiDoc / PDF)
to Markdown, generates a synthetic QA dataset from it using a large remote LLM,
then fine-tunes a smaller local model on that dataset with LoRA + SFT.

| Step | Script | Tool | Purpose |
|------|--------|------|---------|
| 0 | `00_convert_docs.py` | [pypandoc](https://github.com/JessicaTegworthy/pypandoc) / [docling](https://github.com/DS4SD/docling) | Convert `.adoc` and `.pdf` files to Markdown |
| 1 | `01_generate_dataset.py` | [SDG Hub](https://github.com/instructlab/sdg) | Extract QA pairs from the Markdown corpus via a remote LLM |
| 2 | `02_train_model.py` | [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) | Fine-tune a small model on the generated dataset (LoRA + SFT) |

## Target Environment

This pipeline is designed for **RHEL AI 1.5** on an **AWS g6.xlarge** instance
(1 x NVIDIA L4 24 GB), though it works on any CUDA-capable machine.

| Component | Requirement |
|-----------|-------------|
| OS | RHEL AI 1.5 (or any Linux with CUDA) |
| GPU | NVIDIA L4 24 GB (or equivalent) |
| Python | 3.12 |
| System | `pandoc` (for AsciiDoc conversion) |
| Remote LLM | Any OpenAI-compatible endpoint (for step 1 only) |

## Quick Start

```bash
# Create a virtual environment
python3.12 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install sdg-hub
pip install training-hub[lora]
pip install pypandoc docling

# Make sure pandoc is installed on the system
# Fedora/RHEL: sudo dnf install pandoc
# Ubuntu/Debian: sudo apt install pandoc

# Step 0 — convert AsciiDoc docs to a single Markdown file
python 00_convert_docs.py docs/ -o corpus.md

# Step 1 — generate dataset (uses a remote LLM, no GPU needed)
python 01_generate_dataset.py \
  --model "openai/qwen3-14b" \
  --url "$LLM_ENDPOINT" \
  --token "$API_KEY" \
  --input corpus.md \
  --output dataset.csv \
  --domain "Infrastructure" \
  --outline "OpenShift Virtualization Networking"

# Step 2 — fine-tune a small model on the GPU
python 02_train_model.py \
  --dataset dataset.csv \
  --model "ibm-granite/granite-4.0-1b" \
  --output ./checkpoints \
  --system-prompt "You are an expert in OpenShift Virtualization networking." \
  --epochs 3 \
  --max-seq-len 512
```

## Step 0: Convert Documentation to Markdown

`00_convert_docs.py` converts `.adoc` (AsciiDoc) and `.pdf` files to Markdown.
AsciiDoc files are converted using **pypandoc** (requires `pandoc` on the
system) and PDF files are converted using **docling**.

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

## Step 1: Generate a Synthetic Dataset

`01_generate_dataset.py` reads a Markdown document, chunks it, and sends each
chunk to a remote LLM through an SDG Hub flow to produce question-answer pairs.

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

## Step 2: Fine-Tune a Model

`02_train_model.py` converts the CSV from step 1 to JSONL messages format and
fine-tunes a small model using Training Hub's LoRA + SFT algorithm (backed by
Unsloth for speed).

On the L4 (24 GB), QLoRA 4-bit quantization is enabled by default, which lets
you fine-tune models up to ~7 B parameters comfortably.

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

| Model | Params | Notes |
|-------|--------|-------|
| `ibm-granite/granite-4.0-1b` | 2 B | Fast training, good baseline |
| `ibm-granite/granite-4.0-350m` | 0.4 B | Ultra-lightweight |
| `ibm-granite/granite-3.3-2b-instruct` | 2 B | Previous-gen Granite |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5 B | Strong multilingual model |

With QLoRA 4-bit, larger models (7 B+) also fit in 24 GB.

## Project Structure

```
00_convert_docs.py      # Step 0 — document conversion (.adoc / .pdf -> .md)
01_generate_dataset.py  # Step 1 — synthetic data generation
02_train_model.py       # Step 2 — LoRA + SFT fine-tuning
corpus.md               # Example source document (output of step 0, input to step 1)
README.md
```

## License

Apache-2.0
