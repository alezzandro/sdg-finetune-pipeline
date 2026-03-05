import os
import json
import hashlib
import argparse
import re
import pandas as pd
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

os.environ['LITELLM_LOG'] = 'INFO'

REASONING_TAG_PATTERN = re.compile(
    r'<(think|reasoning|reflection)>.*?</\1>', flags=re.DOTALL
)


def strip_reasoning_traces(text):
    """Remove chain-of-thought tags emitted by reasoning models."""
    if not isinstance(text, str):
        return text
    return REASONING_TAG_PATTERN.sub('', text).strip()


def preprocess_technical_text(text):
    """Strip base64 images, data URIs, and leftover AsciiDoc variables."""
    text = re.sub(r'<figure>.*?</figure>', '', text, flags=re.DOTALL)
    text = re.sub(r'<img\s[^>]*data:image/[^>]*>', '', text, flags=re.DOTALL)
    text = re.sub(r'!\[.*?\]\(data:image/[^)]+\)', '', text)
    text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+', '', text)
    text = re.sub(r'\{[A-Za-z][\w-]*\}', '', text)
    return text


def _split_oversized(text, max_chars):
    """Sub-split a text block that exceeds max_chars by paragraph boundaries."""
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 < max_chars:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            if len(para) >= max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars].strip())
            else:
                current = para
                continue
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text(text, max_chars=2500):
    """Split markdown text into chunks that fit within a model's context window."""
    text = preprocess_technical_text(text)

    sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
    chunks = []
    current_chunk = ""
    for section in sections:
        if not section.strip():
            continue
        if len(current_chunk) + len(section) < max_chars:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(section) >= max_chars:
                chunks.extend(_split_oversized(section, max_chars))
                current_chunk = ""
                continue
            current_chunk = section
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic QA dataset from a markdown document using SDG Hub."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="LLM model identifier (e.g. openai/qwen3-14b)")
    parser.add_argument("--url", type=str, required=True,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--token", type=str, required=True,
                        help="API key / bearer token")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the source markdown document")
    parser.add_argument("--output", type=str, default="dataset.csv",
                        help="Output CSV path (default: dataset.csv)")
    parser.add_argument("--domain", type=str, default="General",
                        help="Knowledge domain label (default: General)")
    parser.add_argument("--outline", type=str, default="",
                        help="Short document outline / topic description")
    parser.add_argument("--flow", type=str,
                        default="Key Facts Knowledge Tuning Dataset Generation Flow",
                        help="SDG Hub flow name to use")
    parser.add_argument("--max-chunk-chars", type=int, default=2500,
                        help="Max characters per document chunk (default: 2500)")
    parser.add_argument("--max-concurrency", type=int, default=10,
                        help="Max concurrent LLM requests (default: 10)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-request timeout in seconds (default: 600)")
    parser.add_argument("--keep-cot", action="store_true",
                        help="Keep reasoning/chain-of-thought tags in output")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Chunks per batch; results are checkpointed after each batch (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint instead of starting over")
    args = parser.parse_args()

    # --- 1. Load and chunk the source document ---
    with open(args.input, "r", encoding="utf-8") as f:
        full_text = f.read()

    document_chunks = chunk_text(full_text, max_chars=args.max_chunk_chars)
    print(f"Preprocessed input into {len(document_chunks)} chunks "
          f"(max {args.max_chunk_chars} chars each).")

    # --- 2. Build input dataset (deduplicated) ---
    seen = set()
    data_list = []
    for chunk in document_chunks:
        if chunk in seen:
            continue
        seen.add(chunk)
        data_list.append({
            "document": chunk,
            "document_outline": args.outline,
            "domain": args.domain,
        })

    if len(seen) < len(document_chunks):
        print(f"Removed {len(document_chunks) - len(seen)} duplicate chunks.")

    # --- 3. Checkpoint / resume logic ---
    input_hash = hashlib.sha256(full_text.encode()).hexdigest()[:12]
    checkpoint_path = args.output + f".checkpoint.{input_hash}.csv"
    meta_path = checkpoint_path + ".meta"

    chunks_done = 0
    if args.resume and os.path.exists(meta_path):
        with open(meta_path, "r") as mf:
            meta = json.load(mf)
        chunks_done = meta.get("chunks_processed", 0)
        print(f"Resuming: {chunks_done} / {len(data_list)} input chunks already processed.")
    elif not args.resume:
        for f in (checkpoint_path, meta_path):
            if os.path.exists(f):
                os.remove(f)
        print("Existing checkpoint discarded (use --resume to continue from it).")

    remaining = data_list[chunks_done:]
    if not remaining:
        print("All chunks already processed. Finalizing output...")
        _finalize(checkpoint_path, meta_path, args)
        return

    print(f"{len(remaining)} chunks remaining to process.")

    # --- 4. Load and configure the SDG flow ---
    print(f"Loading flow: {args.flow}")
    FlowRegistry.discover_flows()
    flow_path = FlowRegistry.get_flow_path(args.flow)
    if flow_path is None:
        print(f"Error: flow '{args.flow}' not found. Use FlowRegistry.discover_flows() to list available flows.")
        return

    flow = Flow.from_yaml(flow_path)
    flow.set_model_config(
        model=args.model,
        api_base=args.url,
        api_key=args.token,
        timeout=args.timeout,
    )

    # --- 5. Generate in batches with checkpoints ---
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(remaining))
        batch = remaining[start:end]
        batch_ds = Dataset.from_list(batch)

        print(f"\n--- Batch {batch_idx + 1}/{total_batches} "
              f"(chunks {start + 1}–{end} of {len(remaining)}) ---")
        try:
            result = flow.generate(batch_ds, max_concurrency=args.max_concurrency)
        except Exception as e:
            print(f"\nBatch {batch_idx + 1} failed: {e}")
            print(f"Progress saved to {checkpoint_path}. "
                  f"Re-run with --resume to continue.")
            return

        batch_df = result.to_pandas()
        write_header = not os.path.exists(checkpoint_path)
        batch_df.to_csv(checkpoint_path, mode="a", index=False, header=write_header)

        absolute_done = (len(data_list) - len(remaining)) + end
        with open(meta_path, "w") as mf:
            json.dump({"chunks_processed": absolute_done}, mf)
        print(f"Checkpoint saved ({end} / {len(remaining)} chunks done).")

    # --- 6. Finalize ---
    _finalize(checkpoint_path, meta_path, args)


def _finalize(checkpoint_path, meta_path, args):
    """Merge checkpoint into the final output CSV and clean up."""
    df = pd.read_csv(checkpoint_path)

    if not args.keep_cot:
        print("Stripping reasoning traces from output columns...")
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(strip_reasoning_traces)

    df.to_csv(args.output, index=False)
    for f in (checkpoint_path, meta_path):
        if os.path.exists(f):
            os.remove(f)
    print(f"\nDone — {len(df)} rows saved to {args.output}")


if __name__ == "__main__":
    main()
