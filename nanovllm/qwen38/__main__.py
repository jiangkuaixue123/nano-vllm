"""Run the dedicated BF16/block-FP8 text prototype on one visible CUDA device."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-context-tokens", type=int, required=True)
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from nanovllm.qwen38.loader import load_checkpoint
    from nanovllm.qwen38.runner import RequestRunner

    model, _ = load_checkpoint(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    generation = json.loads((args.model / "generation_config.json").read_bytes())
    eos = generation["eos_token_id"]
    if type(eos) is int:
        eos = [eos]
    runner = RequestRunner(model, tokenizer, max_context_tokens=args.max_context_tokens,
                           vocab_size=model.config.vocab_size)
    kwargs = {"max_tokens": args.max_new_tokens, "eos_token_ids": eos}
    result = (runner.chat([{"role": "user", "content": args.prompt}], **kwargs)
              if args.chat else runner.generate(args.prompt, **kwargs))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
