import torch
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
        "What is the capital of France?",
        "What is artificial intelligence?",
        "What does def fibonacci(n): do?",
]
    for prompt in prompts:
        result = generate(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {result}")
        print()
