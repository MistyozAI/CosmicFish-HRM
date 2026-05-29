import os
import sys
import json
import shutil
import argparse
import logging

import torch
import tiktoken

from safetensors.torch import save_file, load_file
from model import HRMCosmicFish, HRMCosmicFishConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = "HF_release"


def get_args():
    parser = argparse.ArgumentParser(description="Convert CosmicFish-HRM checkpoint to Hugging Face release format")
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def load_checkpoint(model_path):
    logger.info(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if "config" not in checkpoint:
        logger.error("No config found in checkpoint")
        sys.exit(1)

    config = checkpoint["config"]

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        logger.error("No model weights found in checkpoint")
        sys.exit(1)

    cleaned = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

    logger.info(f"Loaded {len(cleaned)} tensors from checkpoint")
    return config, cleaned


def save_safetensors(state_dict, output_dir):
    path = os.path.join(output_dir, "model.safetensors")
    contiguous = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(contiguous, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"Saved model.safetensors ({size_mb:.1f} MB)")
    return path


def create_config_json(config, output_dir):
    config_dict = {
        "model_type": "cosmicfish_hrm",
        "architectures": ["HRMCosmicFish"],
        "vocab_size": config.vocab_size,
        "n_embd": config.n_embd,
        "block_size": config.block_size,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "n_input_layers": config.n_input_layers,
        "n_output_layers": config.n_output_layers,
        "hrm_H_layers": config.hrm_H_layers,
        "hrm_L_layers": config.hrm_L_layers,
        "hrm_H_cycles": config.hrm_H_cycles,
        "hrm_L_cycles": config.hrm_L_cycles,
        "hrm_max_steps": config.hrm_max_steps,
        "hrm_exploration_prob": config.hrm_exploration_prob,
        "dropout": config.dropout,
        "bias": config.bias,
        "use_rotary": config.use_rotary,
        "use_gqa": config.use_gqa,
        "use_swiglu": config.use_swiglu,
        "eps": config.eps,
        "torch_dtype": "float32",
        "transformers_version": "4.41.0",
        "pad_token_id": 50256,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }

    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Created config.json")
    return config_dict


def create_tokenizer_files(output_dir):
    enc = tiktoken.get_encoding("gpt2")

    tokenizer_config = {
        "tokenizer_class": "GPT2Tokenizer",
        "vocab_size": enc.n_vocab,
        "model_max_length": 512,
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "add_prefix_space": False,
    }

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }

    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    logger.info("Created tokenizer_config.json and special_tokens_map.json")


def create_model_card(config_dict, output_dir):
    total_params = 82.77

    card = f"""---
license: apache-2.0
tags:
  - text-generation
  - causal-lm
  - cosmicfish
  - hrm
  - adaptive-reasoning
  - custom-architecture
language:
  - en
---

# CosmicFish-HRM

CosmicFish-HRM is a compact 82.77M parameter causal language model built around a Hierarchical Reasoning Module (HRM) that dynamically allocates reasoning compute during inference. Developed at Mistyoz AI.

Rather than applying a fixed number of transformer layers to every input, CosmicFish-HRM iterates through high-level and low-level reasoning cycles and uses a learned halting head to decide when to stop. Harder inputs trigger deeper reasoning trajectories while simpler ones halt early.

## Architecture

```
Input Blocks (Transformer) -> HRM Core (H + L levels, variable steps) -> Output Blocks (Transformer) -> LM Head
```

| Hyperparameter | Value |
|---|---|
| Parameters | 82.77M |
| Embedding dimension | 448 |
| Vocabulary size | 50,304 |
| Context length | 512 |
| Input layers | 6 |
| Output layers | 6 |
| Attention heads | 8 (4 KV, GQA) |
| HRM H-layers | 4 |
| HRM L-layers | 4 |
| Max HRM steps | 16 |

**Key components:**
- Grouped-Query Attention (GQA) with RoPE
- SwiGLU feedforward layers
- RMSNorm (pre-norm for I/O blocks, post-norm inside HRM)
- Learned halt/continue Q-head controlling per-input reasoning depth
- Step penalty in training loss encouraging efficient halting

## Usage

This model uses a custom architecture and requires `trust_remote_code=True`.

```python
import torch
import json
import tiktoken
from safetensors.torch import load_file
from modeling_hrm_cosmicfish import HRMCosmicFish, HRMCosmicFishConfig

with open("config.json") as f:
    cfg = json.load(f)

config = HRMCosmicFishConfig(
    vocab_size=cfg["vocab_size"],
    n_embd=cfg["n_embd"],
    block_size=cfg["block_size"],
    n_head=cfg["n_head"],
    n_kv_head=cfg["n_kv_head"],
    n_input_layers=cfg["n_input_layers"],
    n_output_layers=cfg["n_output_layers"],
    hrm_H_layers=cfg["hrm_H_layers"],
    hrm_L_layers=cfg["hrm_L_layers"],
    hrm_H_cycles=cfg["hrm_H_cycles"],
    hrm_L_cycles=cfg["hrm_L_cycles"],
    hrm_max_steps=cfg["hrm_max_steps"],
    dropout=0.0,
)

state_dict = load_file("model.safetensors")
model = HRMCosmicFish(config)
model.load_state_dict(state_dict)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

prompt = "Artificial intelligence is"
tokens = tokenizer.encode(prompt)
idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

with torch.no_grad():
    output = model.generate(idx, max_new_tokens=50, temperature=0.7, top_k=40)

print(tokenizer.decode(output[0].tolist()))
```

## Training

CosmicFish-HRM was trained on the 10B-token CosmicSet dataset spanning web text, Wikipedia, code, mathematics, and research papers. Training used cosine LR decay with linear warmup, bfloat16 mixed precision, and gradient clipping.

## Citation

```bibtex
@misc{{cosmicfish-hrm,
  title={{CosmicFish-HRM: Adaptive Reasoning via Hierarchical Recurrent Mechanisms in Compact Language Models}},
  author={{Venkat Akhil Lakkapragada}},
  year={{2026}},
  howpublished={{\\url{{https://huggingface.co/MistyozAI/CosmicFish-HRM}}}}
}}
```

---

Mistyoz AI, Hyderabad
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)

    logger.info("Created README.md")


def create_example_usage(output_dir):
    code = '''import torch
import json
import tiktoken
from safetensors.torch import load_file
from modeling_hrm_cosmicfish import HRMCosmicFish, HRMCosmicFishConfig


def load_model(model_dir, device="cpu"):
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)

    config = HRMCosmicFishConfig(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        n_head=cfg["n_head"],
        n_kv_head=cfg["n_kv_head"],
        n_input_layers=cfg["n_input_layers"],
        n_output_layers=cfg["n_output_layers"],
        hrm_H_layers=cfg["hrm_H_layers"],
        hrm_L_layers=cfg["hrm_L_layers"],
        hrm_H_cycles=cfg["hrm_H_cycles"],
        hrm_L_cycles=cfg["hrm_L_cycles"],
        hrm_max_steps=cfg["hrm_max_steps"],
        dropout=0.0,
    )

    state_dict = load_file(f"{model_dir}/model.safetensors")
    model = HRMCosmicFish(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    return model, tokenizer


def generate(model, tokenizer, prompt, device="cpu", max_new_tokens=100, temperature=0.7, top_k=40):
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(output[0].tolist())


if __name__ == "__main__":
    model, tokenizer = load_model(".")
    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "def fibonacci(n):",
    ]
    for prompt in prompts:
        result = generate(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {result}")
        print()
'''

    with open(os.path.join(output_dir, "example_usage.py"), "w") as f:
        f.write(code)

    logger.info("Created example_usage.py")


def test_safetensors(state_dict, config, output_dir):
    logger.info("Testing safetensors weights by loading and running a forward pass")

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    loaded_sd = load_file(safetensors_path)

    config.dropout = 0.0
    model = HRMCosmicFish(config)
    model.load_state_dict(loaded_sd)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompts = [
        "The capital of France is",
        "Photosynthesis is the process by which",
        "def add(a, b):",
    ]

    original_model = HRMCosmicFish(config)
    original_model.load_state_dict(state_dict)
    original_model.eval()

    all_passed = True

    with torch.no_grad():
        for prompt in prompts:
            tokens = enc.encode(prompt)
            idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

            orig_logits, _, orig_steps, _ = original_model(idx)
            st_logits, _, st_steps, _ = model(idx)

            orig_top = orig_logits[0, -1, :].argmax().item()
            st_top = st_logits[0, -1, :].argmax().item()

            max_diff = (orig_logits - st_logits).abs().max().item()
            match = orig_top == st_top
            status = "PASS" if match else "FAIL"

            if not match:
                all_passed = False

            logger.info(
                f"  [{status}] \"{prompt}\" -> "
                f"orig: \"{enc.decode([orig_top])}\" | "
                f"st: \"{enc.decode([st_top])}\" | "
                f"max_logit_diff: {max_diff:.6f} | "
                f"steps: {orig_steps.item()} -> {st_steps.item()}"
            )

    if all_passed:
        logger.info("Safetensors test passed: all predictions match original checkpoint")
    else:
        logger.warning("Safetensors test: some predictions differ. Review logit diffs above.")

    return all_passed


def main():
    args = get_args()

    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    try:
        from safetensors.torch import save_file as _sf_check
    except ImportError:
        logger.error("safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    config, state_dict = load_checkpoint(args.model)

    save_safetensors(state_dict, OUTPUT_DIR)
    config_dict = create_config_json(config, OUTPUT_DIR)
    create_tokenizer_files(OUTPUT_DIR)
    create_model_card(config_dict, OUTPUT_DIR)
    create_example_usage(OUTPUT_DIR)

    if os.path.exists("model.py"):
        shutil.copy2("model.py", os.path.join(OUTPUT_DIR, "modeling_hrm_cosmicfish.py"))
        logger.info("Copied model.py as modeling_hrm_cosmicfish.py")
    else:
        logger.warning("model.py not found in current directory")

    test_safetensors(state_dict, config, OUTPUT_DIR)

    logger.info("Release preparation complete")
    logger.info(f"Files in {OUTPUT_DIR}:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.info(f"  {fname} ({size_mb:.2f} MB)")

    logger.info("Next steps:")
    logger.info("  1. Review HF_release/ contents")
    logger.info("  2. Run: huggingface-cli login")
    logger.info("  3. Run: huggingface-cli upload MistyozAI/CosmicFish-HRM HF_release/")


if __name__ == "__main__":
    main()
