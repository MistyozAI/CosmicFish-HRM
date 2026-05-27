import argparse
import os
import sys
import time
import torch
import tiktoken
import re
import textwrap
from termcolor import colored

from model import HRMCosmicFish, HRMCosmicFishConfig


def clean_text(text):
    text = text.replace('ï¿½', "'")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    return text.strip()


def get_repetition_penalty_logits(input_ids, logits, penalty=1.2):
    for input_ids_slice in input_ids:
        for token_id in set(input_ids_slice.tolist()):
            logits[:, token_id] /= penalty
    return logits


def apply_diversity_penalty(logits, recent_tokens, penalty=1.5):
    if len(recent_tokens) > 10:
        pairs = [(recent_tokens[i], recent_tokens[i + 1]) for i in range(len(recent_tokens) - 1)]
        pair_counts = {}
        for pair in pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        for pair, count in pair_counts.items():
            if count > 2:
                recent_context = (recent_tokens[-2], recent_tokens[-1])
                if recent_context == pair:
                    third_token = recent_tokens[pairs.index(pair) + 2]
                    logits[:, third_token] /= penalty

    return logits


def generate_text(model, tokenizer, prompt, device, args):
    cleaned_prompt = clean_text(prompt)
    tokens = tokenizer.encode(cleaned_prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = cleaned_prompt
    recent_tokens = []
    current_hrm_max = model.config.hrm_max_steps
    hrm_steps_per_token = []

    print(f"\n{'=' * 80}")
    print(f"Generating with:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  HRM Max Steps: {current_hrm_max}")
    if args.show_hrm_steps:
        print(f"  [HRM Debug Mode: ON]")
    print(f"{'=' * 80}\n")

    print(colored("PROMPT:", "green"))
    for line in textwrap.wrap(cleaned_prompt, width=80):
        print(line)
    print("\n" + "-" * 80 + "\n")

    print(colored("GENERATION:", "blue"))
    print(cleaned_prompt, end="", flush=True)

    start_time = time.time()
    total_hrm_steps = 0

    with torch.no_grad():
        for gen_step in range(args.max_tokens):
            if input_ids.size(1) > model.config.block_size:
                context = input_ids[:, -model.config.block_size:]
            else:
                context = input_ids

            logits, _, steps_taken, _ = model(context)

            actual_steps = steps_taken.item() if isinstance(steps_taken, torch.Tensor) else steps_taken
            total_hrm_steps += actual_steps
            hrm_steps_per_token.append(actual_steps)

            if args.show_hrm_steps and gen_step < 10:
                print(f"\n[HRM] Token {gen_step}: {actual_steps} steps", end="", flush=True)

            logits = logits[:, -1, :] / args.temperature
            logits = get_repetition_penalty_logits(context, logits, 1.2)
            logits = apply_diversity_penalty(logits, recent_tokens)

            if len(recent_tokens) >= 2:
                if recent_tokens[-1] == recent_tokens[-2]:
                    logits[:, recent_tokens[-1]] /= 2.0
                if len(recent_tokens) >= 4 and recent_tokens[-1] == recent_tokens[-3] and recent_tokens[-2] == recent_tokens[-4]:
                    logits[:, recent_tokens[-1]] /= 2.0
                    logits[:, recent_tokens[-2]] /= 2.0

            if args.top_k > 0:
                values, _ = torch.topk(logits, min(args.top_k, logits.shape[-1]))
                logits[logits < values[:, [-1]]] = float('-inf')

            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            recent_tokens.append(next_token.item())
            if len(recent_tokens) > 50:
                recent_tokens.pop(0)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == 50256 or (len(recent_tokens) > 20 and len(set(recent_tokens[-10:])) < 3):
                break

            new_text = tokenizer.decode([next_token.item()])
            cleaned_text = clean_text(new_text)

            if cleaned_text and not (cleaned_text.startswith(('.', ',', '!', '?', ';', ':')) or generated_text.endswith(' ')):
                cleaned_text = ' ' + cleaned_text

            if args.show_hrm_steps and gen_step < 10:
                print(f" → {cleaned_text}", end="", flush=True)
            else:
                print(cleaned_text, end="", flush=True)

            generated_text += cleaned_text

    end_time = time.time()
    generation_time = end_time - start_time
    total_tokens = len(hrm_steps_per_token)
    tokens_per_sec = total_tokens / generation_time if generation_time > 0 else 0
    avg_hrm_steps = total_hrm_steps / total_tokens if total_tokens > 0 else 0

    if hrm_steps_per_token:
        min_steps = min(hrm_steps_per_token)
        max_steps = max(hrm_steps_per_token)
        step_counts = {}
        for steps in hrm_steps_per_token:
            step_counts[steps] = step_counts.get(steps, 0) + 1

    print("\n\n" + "-" * 80)
    print(f"Generated {total_tokens} tokens in {generation_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")
    print(f"HRM Steps:")
    print(f"  Average: {avg_hrm_steps:.2f}")
    print(f"  Range: {min_steps} - {max_steps}")
    print(f"  Max allowed: {current_hrm_max}")

    if args.show_hrm_steps:
        print(f"\nHRM Step Distribution:")
        for step_count in sorted(step_counts.keys()):
            percentage = (step_counts[step_count] / total_tokens) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {step_count:2d} steps: {bar} {step_counts[step_count]:3d} ({percentage:5.1f}%)")

    print("=" * 80)

    return clean_text(generated_text)


def load_model(model_path, device, force_hrm_steps=None):
    print(f"Loading model from {model_path}...")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("No config found in checkpoint")

        original_hrm_steps = config.hrm_max_steps

        if force_hrm_steps is not None:
            config.hrm_max_steps = force_hrm_steps
            print(f"Overriding HRM max steps: {original_hrm_steps} → {force_hrm_steps}")

        model = HRMCosmicFish(config)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            raise ValueError("No model weights found in checkpoint")

        cleaned_state_dict = {
            k.replace('module.', '').replace('_orig_mod.', ''): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(cleaned_state_dict)
        model.to(device)
        model.eval()

        print(f"Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
        print(f"  Input blocks: {config.n_input_layers}")
        print(f"  HRM Core: H={config.hrm_H_layers} L={config.hrm_L_layers} (max {config.hrm_max_steps} steps)")
        print(f"  Output blocks: {config.n_output_layers}")

        config.original_hrm_max_steps = original_hrm_steps

        return model, config

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test HRM-Enhanced CosmicFish model")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--force_hrm_steps", type=int, default=None)
    parser.add_argument("--show_hrm_steps", action="store_true")

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    model, config = load_model(args.model_path, args.device, args.force_hrm_steps)
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.interactive:
        print("\n" + "=" * 80)
        print("HRM-CosmicFish Interactive Mode")
        print("=" * 80)
        print("Commands: 'quit'/'exit' to exit")
        print("          '/temp X'   — set temperature")
        print("          '/tokens X' — set max tokens")
        print("          '/hrm X'    — set HRM max steps")
        print("          '/topk X'   — set top-k")
        print("          '/topp X'   — set top-p")
        print("          '/debug'    — toggle HRM debug mode")
        print("=" * 80 + "\n")

        while True:
            try:
                prompt = input(colored("\nPrompt: ", "yellow"))

                if prompt.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break

                if prompt.startswith('/temp '):
                    try:
                        args.temperature = float(prompt.split()[1])
                        print(f"Temperature set to {args.temperature}")
                    except:
                        print("Invalid temperature")
                    continue

                if prompt.startswith('/tokens '):
                    try:
                        args.max_tokens = int(prompt.split()[1])
                        print(f"Max tokens set to {args.max_tokens}")
                    except:
                        print("Invalid token count")
                    continue

                if prompt.startswith('/hrm '):
                    try:
                        new_hrm_steps = int(prompt.split()[1])
                        if new_hrm_steps < 1:
                            print("HRM steps must be >= 1")
                            continue
                        model.config.hrm_max_steps = new_hrm_steps
                        print(f"HRM max steps set to {new_hrm_steps}")
                        if hasattr(config, 'original_hrm_max_steps'):
                            print(f"  (Model was trained with {config.original_hrm_max_steps} steps)")
                    except:
                        print("Invalid HRM steps")
                    continue

                if prompt.startswith('/topk '):
                    try:
                        args.top_k = int(prompt.split()[1])
                        print(f"Top-k set to {args.top_k}")
                    except:
                        print("Invalid top-k")
                    continue

                if prompt.startswith('/topp '):
                    try:
                        args.top_p = float(prompt.split()[1])
                        print(f"Top-p set to {args.top_p}")
                    except:
                        print("Invalid top-p")
                    continue

                if prompt.strip() == '/debug':
                    args.show_hrm_steps = not args.show_hrm_steps
                    print(f"HRM debug mode: {'ON' if args.show_hrm_steps else 'OFF'}")
                    continue

                if not prompt.strip():
                    continue

                generate_text(model, tokenizer, prompt, args.device, args)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                import traceback
                traceback.print_exc()

    else:
        if args.file:
            with open(args.file, 'r') as f:
                prompt = f.read().strip()
        elif args.prompt:
            prompt = args.prompt
        else:
            print("Please provide --prompt, --file, or use --interactive")
            return

        generate_text(model, tokenizer, prompt, args.device, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)