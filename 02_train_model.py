import os
import sys
import json
import signal
import argparse
import subprocess
import tempfile
import time

# Unsloth must be imported before transformers for its optimizations to apply.
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import pandas as pd

LOGFILE = "train_model.log"
PIDFILE = "train_model.pid"


def detect_device():
    """Detect the best available compute device and return a status summary."""
    import torch

    info = {
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "device_name": "CPU",
        "quantization_supported": False,
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["quantization_supported"] = True

    return info


def csv_to_jsonl(csv_path, output_path, system_prompt=None):
    """Convert step-1 CSV (question/response columns) to JSONL messages format.

    Returns the number of training examples written.
    """
    df = pd.read_csv(csv_path)

    required = {"question", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["question", "response"])

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": str(row["question"])})
            messages.append({"role": "assistant", "content": str(row["response"])})
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1

    return count


def _read_pid():
    """Read the PID from the pidfile, return None if missing or stale."""
    if not os.path.exists(PIDFILE):
        return None
    with open(PIDFILE) as f:
        try:
            pid = int(f.read().strip())
        except ValueError:
            return None
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        os.remove(PIDFILE)
        return None


def _stop_process():
    """Stop a running background training process."""
    pid = _read_pid()
    if pid is None:
        print("No running training process found.")
        return
    print(f"Stopping training process (PID {pid})...")
    os.kill(pid, signal.SIGTERM)
    for _ in range(15):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except OSError:
            break
    else:
        print("Process did not exit gracefully, sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
    if os.path.exists(PIDFILE):
        os.remove(PIDFILE)
    print("Training process stopped.")


def _show_status():
    """Show the status of a background training process."""
    pid = _read_pid()
    if pid is None:
        print("No running training process found.")
        return
    print(f"Training process is running (PID {pid}).")
    if os.path.exists(LOGFILE):
        log_path = os.path.abspath(LOGFILE)
        print(f"Log file: {log_path}")
        with open(LOGFILE) as f:
            lines = f.readlines()
        progress = [l for l in lines if "Training complete" in l
                    or "loss" in l.lower() or "epoch" in l.lower()
                    or "step" in l.lower()]
        if progress:
            print(f"Last progress: {progress[-1].strip()}")
        else:
            print("Status: starting up...")


def _cleanup_pidfile():
    """Remove pidfile if it belongs to this process."""
    if not os.path.exists(PIDFILE):
        return
    with open(PIDFILE) as f:
        try:
            pid = int(f.read().strip())
        except ValueError:
            return
    if pid == os.getpid():
        os.remove(PIDFILE)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on a QA dataset using Training Hub (LoRA + SFT)."
    )
    parser.add_argument("--dataset", type=str,
                        help="Path to the CSV dataset from step 1 (must have question/response columns)")
    parser.add_argument("--model", type=str, default="ibm-granite/granite-4.0-1b",
                        help="HuggingFace model ID or local path (default: ibm-granite/granite-4.0-1b)")
    parser.add_argument("--output", type=str, default="./checkpoints",
                        help="Directory to save the fine-tuned model (default: ./checkpoints)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt prepended to each training example")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Max sequence length (default: 512)")
    parser.add_argument("--micro-batch-size", type=int, default=2,
                        help="Batch size per device (default: 2)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable QLoRA 4-bit quantization")
    parser.add_argument("--background", action="store_true",
                        help="Run in background with output logged to train_model.log")
    parser.add_argument("--status", action="store_true",
                        help="Show the status of a background training process")
    parser.add_argument("--stop", action="store_true",
                        help="Stop a running background training process")
    args = parser.parse_args()

    if args.status:
        _show_status()
        return

    if args.stop:
        _stop_process()
        return

    if args.background:
        existing_pid = _read_pid()
        if existing_pid is not None:
            print(f"ERROR: Training is already running (PID {existing_pid}).",
                  file=sys.stderr)
            print("Stop it first with: python3.12 02_train_model.py --stop",
                  file=sys.stderr)
            sys.exit(1)

        log_path = os.path.abspath(LOGFILE)
        bg_args = [a for a in sys.argv if a != "--background"]
        with open(LOGFILE, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable] + bg_args,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with open(PIDFILE, "w") as pf:
            pf.write(str(proc.pid))

        print(f"Training started in background (PID {proc.pid}).")
        print(f"\n{'='*60}")
        print(f"  Log file: {log_path}")
        print(f"")
        print(f"  Monitor progress:")
        print(f"    tail -f {LOGFILE}")
        print(f"")
        print(f"  Check status:")
        print(f"    python3.12 02_train_model.py --status")
        print(f"")
        print(f"  Stop the process:")
        print(f"    python3.12 02_train_model.py --stop")
        print(f"{'='*60}")
        return

    if not args.dataset:
        parser.error("the following arguments are required: --dataset")

    # --- 1. Detect compute device ---
    device_info = detect_device()
    print(f"PyTorch {device_info['pytorch_version']}")
    print(f"Device:  {device_info['device']} ({device_info['device_name']})")

    use_4bit = device_info["quantization_supported"] and not args.no_quantize

    if device_info["device"] == "cpu":
        print("WARNING: No CUDA GPU detected. Training will run on CPU (slow).")
        use_4bit = False

    print(f"Quantization: {'QLoRA 4-bit' if use_4bit else 'disabled'}")

    # --- 2. Convert CSV to JSONL messages format ---
    jsonl_path = os.path.join(tempfile.gettempdir(), "training_messages.jsonl")
    num_examples = csv_to_jsonl(args.dataset, jsonl_path, args.system_prompt)
    print(f"\nConverted {num_examples} QA pairs from CSV to JSONL messages format.")
    print(f"Training data: {jsonl_path}")

    if num_examples == 0:
        print("ERROR: No valid training examples found. Check your CSV.")
        return

    # --- 3. Fine-tune with Training Hub ---
    print(f"\nStarting LoRA + SFT training:")
    print(f"  Model:          {args.model}")
    print(f"  LoRA rank:      {args.lora_r}")
    print(f"  LoRA alpha:     {args.lora_alpha}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Max seq length: {args.max_seq_len}")
    print(f"  Batch size:     {args.micro_batch_size} (x{args.gradient_accumulation_steps} accum)")
    print(f"  Output:         {args.output}")
    print()

    from training_hub import lora_sft

    lora_kwargs = dict(
        model_path=args.model,
        data_path=jsonl_path,
        ckpt_output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset_type="chat_template",
        field_messages="messages",
        logging_steps=1,
        save_total_limit=2,
    )

    if use_4bit:
        lora_kwargs["load_in_4bit"] = True

    try:
        result = lora_sft(**lora_kwargs)
        print(f"\nTraining complete. Model saved to: {args.output}")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise
    finally:
        _cleanup_pidfile()


if __name__ == "__main__":
    main()
