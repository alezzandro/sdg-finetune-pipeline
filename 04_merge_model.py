"""Merge a LoRA adapter into the base model and save a standalone copy.

The merged model can be loaded directly with transformers without requiring
PEFT, making it easier to deploy or serve via an inference engine like vLLM.
"""

import argparse

try:
    import unsloth  # noqa: F401
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model and export a standalone model."
    )
    parser.add_argument("--checkpoint", type=str, default="./checkpoints",
                        help="Path to the LoRA checkpoint directory (default: ./checkpoints)")
    parser.add_argument("--output", type=str, default="./merged-model",
                        help="Output directory for the merged model (default: ./merged-model)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization when loading the checkpoint")
    args = parser.parse_args()

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    use_4bit = not args.no_quantize

    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Output:      {args.output}")
    print()

    print("Loading LoRA checkpoint...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint,
        device_map="auto",
        load_in_4bit=use_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    print("Merging LoRA adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output}...")
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone — standalone model saved to {args.output}")
    print("You can load it directly with AutoModelForCausalLM.from_pretrained() or serve it with vLLM.")


if __name__ == "__main__":
    main()
