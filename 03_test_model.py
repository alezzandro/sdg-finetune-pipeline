"""Compare answers from a reference model and the fine-tuned (LoRA) model.

Loads the LoRA checkpoint, then loads a reference model (either specified via
--base-model or auto-resolved from the adapter config) and generates answers
to the same question from both models so you can see the effect of fine-tuning
side by side.
"""

import argparse

try:
    import unsloth  # noqa: F401
except ImportError:
    pass


def load_tokenizer_and_model(model_path, load_in_4bit=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_4bit=load_in_4bit,
    )
    return tokenizer, model


def load_peft_model(checkpoint_path, load_in_4bit=True):
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        load_in_4bit=load_in_4bit,
    )
    return tokenizer, model


def resolve_base_model(checkpoint_path):
    """Read the PEFT adapter_config.json to find the base model identifier."""
    import json
    import os

    config_path = os.path.join(checkpoint_path, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["base_model_name_or_path"]


def generate_answer(model, tokenizer, messages, max_new_tokens=256,
                    repetition_penalty=1.2):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
    )
    new_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare a reference model vs fine-tuned model answers."
    )
    parser.add_argument("--checkpoint", type=str, default="./checkpoints",
                        help="Path to the LoRA checkpoint directory (default: ./checkpoints)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="HuggingFace model ID for the reference model "
                             "(default: auto-detected from adapter config). "
                             "Use an instruct model for a meaningful comparison.")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask both models")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt prepended to the conversation")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization when loading models")
    args = parser.parse_args()

    use_4bit = not args.no_quantize

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.question})

    adapter_base = resolve_base_model(args.checkpoint)
    base_model_id = args.base_model or adapter_base
    if args.base_model:
        print(f"Reference model:  {base_model_id}  (user-specified)")
    else:
        print(f"Reference model:  {base_model_id}  (from adapter config)")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Question:         {args.question}")
    print()

    # --- Reference model ---
    print("Loading reference model...")
    base_tokenizer, base_model = load_tokenizer_and_model(base_model_id, load_in_4bit=use_4bit)
    base_answer = generate_answer(base_model, base_tokenizer, messages, args.max_new_tokens)

    del base_model
    import torch
    torch.cuda.empty_cache()

    # --- Fine-tuned model ---
    print("Loading fine-tuned model...")
    ft_tokenizer, ft_model = load_peft_model(args.checkpoint, load_in_4bit=use_4bit)
    ft_answer = generate_answer(ft_model, ft_tokenizer, messages, args.max_new_tokens)

    del ft_model
    torch.cuda.empty_cache()

    # --- Display ---
    separator = "-" * 72
    print(f"\n{'=' * 72}")
    print(f"REFERENCE MODEL ANSWER  ({base_model_id})")
    print(separator)
    print(base_answer)
    print(f"\n{'=' * 72}")
    print("FINE-TUNED MODEL ANSWER  (LoRA adapter)")
    print(separator)
    print(ft_answer)
    print("=" * 72)


if __name__ == "__main__":
    main()
