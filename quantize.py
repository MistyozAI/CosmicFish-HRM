import argparse
import os
import sys
import torch
import tiktoken

from model import HRMCosmicFish, HRMCosmicFishConfig


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise ValueError("No config found in checkpoint")

    config = checkpoint["config"]

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("No model weights found in checkpoint")

    cleaned = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

    model = HRMCosmicFish(config)
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()

    return model, config


def run_forward_test(model, device):
    tokenizer = tiktoken.get_encoding("gpt2")
    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "def add(a, b):",
    ]

    results = []
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            logits, _, steps_taken, _ = model(idx)
            top_token = logits[0, -1, :].argmax().item()
            next_word = tokenizer.decode([top_token])
            avg_steps = steps_taken.float().mean().item()
            results.append((prompt, next_word, avg_steps))

    return results


def check_dtypes(model):
    dtypes = {}
    for name, param in model.named_parameters():
        t = str(param.dtype)
        dtypes[t] = dtypes.get(t, 0) + 1
    return dtypes


def get_model_size_mb(state_dict):
    total_bytes = sum(v.nelement() * v.element_size() for v in state_dict.values())
    return total_bytes / (1024 * 1024)


def quantize_state_dict(state_dict):
    quantized = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            quantized[k] = v.to(torch.float16)
        else:
            quantized[k] = v
    return quantized


def main():
    parser = argparse.ArgumentParser(description="Quantize CosmicFish-HRM to fp16")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)

    if args.out is None:
        base, ext = os.path.splitext(args.model)
        args.out = base + "_fp16" + ext

    print(f"Loading model from: {args.model}")
    model, config = load_model(args.model, args.device)

    original_state_dict = model.state_dict()
    original_size = get_model_size_mb(original_state_dict)
    original_dtypes = check_dtypes(model)

    print(f"\nPre-quantization dtype distribution: {original_dtypes}")
    print(f"Pre-quantization model size: {original_size:.2f} MB")

    print("\nRunning pre-quantization forward test...")
    pre_results = run_forward_test(model, args.device)
    for prompt, next_word, avg_steps in pre_results:
        print(f"  [{avg_steps:.1f} steps] \"{prompt}\" -> \"{next_word}\"")

    print("\nQuantizing to fp16...")
    quantized_sd = quantize_state_dict(original_state_dict)
    quantized_size = get_model_size_mb(quantized_sd)

    model.load_state_dict(quantized_sd)
    model.eval()

    quantized_dtypes = check_dtypes(model)
    print(f"\nPost-quantization dtype distribution: {quantized_dtypes}")
    print(f"Post-quantization model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {original_size - quantized_size:.2f} MB ({(1 - quantized_size / original_size) * 100:.1f}%)")

    print("\nRunning post-quantization forward test...")
    post_results = run_forward_test(model, args.device)
    passed = True
    for (pre_prompt, pre_word, pre_steps), (post_prompt, post_word, post_steps) in zip(pre_results, post_results):
        match = "OK" if pre_word == post_word else "DIFF"
        if pre_word != post_word:
            passed = False
        print(f"  [{match}] \"{pre_prompt}\" -> pre: \"{pre_word}\" | post: \"{post_word}\"")

    if not passed:
        print("\nWarning: some top token predictions differ after quantization. This may be acceptable.")
    else:
        print("\nAll predictions match.")

    checkpoint = {
        "model": quantized_sd,
        "config": config,
    }

    torch.save(checkpoint, args.out)
    print(f"\nSaved quantized model to: {args.out}")


if __name__ == "__main__":
    main()
