import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal smoke test that loads Llama 2 7B and generates a single patient summary."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HF model id to load (defaults to the chat-tuned 7B Llama 2 checkpoint).",
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        default=Path("data/ann-pt-summ/1.0.1/mimic-iv-note-ext-di-bhc/dataset/valid.json"),
        help="JSONL file with `text` and `summary` fields (defaults to the shipped validation split).",
    )
    parser.add_argument(
        "--example_index",
        type=int,
        default=0,
        help="0-based index of the example to summarize from the JSONL file.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for the summary.",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token for gated Llama checkpoints (falls back to HF_TOKEN env var).",
    )
    return parser.parse_args()


def load_example(path: Path, idx: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file at {path}")
    with path.open() as f:
        for line_no, line in enumerate(f):
            if line_no == idx:
                return json.loads(line)
    raise ValueError(f"Index {idx} is out of range for {path}")


def main() -> None:
    args = parse_args()
    example = load_example(args.data_file, args.example_index)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        token=args.hf_token,
    )
    model.eval()

    prompt = (
        "You are a helpful clinician writing plain-language discharge instructions.\n\n"
        f"Brief Hospital Course:\n{example['text']}\n\n"
        "Patient summary:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = decoded.split("Patient summary:")[-1].strip()

    print("=== Prompted reference summary ===")
    print(example["summary"])
    print("\n=== Model output ===")
    print(summary)


if __name__ == "__main__":
    main()
