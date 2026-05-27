import os
import sys
import time
import argparse
import torch
import numpy as np
from termcolor import colored
import logging
import readline
import re
import textwrap
import random
from collections import defaultdict
import tiktoken

from model import HRMCosmicFish, HRMCosmicFishConfig
from torch.serialization import add_safe_globals

add_safe_globals([HRMCosmicFishConfig])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = "Below is a conversation between a helpful AI assistant and a human. The assistant is knowledgeable, friendly, and provides detailed and accurate responses.\n\n"


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty=1.2):
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score > 0, score / self.penalty, score * self.penalty)
        scores.scatter_(1, input_ids, score)
        return scores


class ChatSession:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.history = []
        self.history_tokens = []
        self.max_history_tokens = config.max_history_tokens
        self.prompt_template = config.prompt_template
        self.human_prefix = config.human_prefix
        self.assistant_prefix = config.assistant_prefix
        self.end_of_turn = config.end_of_turn
        self.block_size = config.block_size
        self.debug_mode = config.debug_mode
        self.repetition_penalty = config.repetition_penalty
        self.min_tokens_to_generate = config.min_tokens_to_generate

        self.hrm_forced_steps = None
        self.original_hrm_max_steps = self.model.config.hrm_max_steps

        self.max_retries = 20

        self.fallback_responses = [
            "I'd be happy to help with that. Could you provide more details?",
            "That's interesting. What specific aspects would you like to know about?",
            "I can help with that. Could you clarify what you're looking for?",
            "Let me help you with that. What particular information do you need?",
            "I understand. Could you be more specific about what you'd like to know?"
        ]

        self.generation_failure_message = "I'm having difficulty generating a response. Could you try rephrasing?"

        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0
        self.total_hrm_steps_used = 0

        self.end_markers = [
            f"{self.human_prefix}",
            "Human:",
            "\nHuman:",
            "\nH:",
            "H:",
            "<|endoftext|>",
            "Below is a conversation",
            "\nA:",
            "A:",
            "</s>",
            "User:",
            "\nUser:"
        ]

        if config.display_welcome:
            self._print_welcome_message()

    def _print_welcome_message(self):
        hrm_mode = f"auto (max {self.original_hrm_max_steps})" if self.hrm_forced_steps is None else str(self.hrm_forced_steps)
        print(colored(f"""
{'=' * 80}
Welcome to CosmicFish-HRM

Model: {self.model.get_num_params() / 1e6:.1f}M parameters
Max HRM Steps: {self.original_hrm_max_steps} | Current HRM Mode: {hrm_mode}

Commands: /help /clear /exit /stats /save /load
          /temp [val]  /penalty [val]  /hrm [n|auto]  /debug
{'=' * 80}
""", 'cyan'))

    def _format_prompt(self, user_input):
        formatted_prompt = self.prompt_template
        for entry in self.history:
            role, text = entry
            if role == "human":
                formatted_prompt += f"{self.human_prefix}{text}{self.end_of_turn}"
            else:
                formatted_prompt += f"{self.assistant_prefix}{text}{self.end_of_turn}"
        formatted_prompt += f"{self.human_prefix}{user_input}{self.end_of_turn}{self.assistant_prefix}"
        return formatted_prompt

    def _tokenize(self, text):
        return self.tokenizer.encode(text)

    def _update_history(self, user_input, response):
        self.history.append(("human", user_input))
        self.history.append(("assistant", response))

        user_tokens     = self._tokenize(f"{self.human_prefix}{user_input}{self.end_of_turn}")
        response_tokens = self._tokenize(f"{self.assistant_prefix}{response}{self.end_of_turn}")

        self.history_tokens.extend(user_tokens)
        self.history_tokens.extend(response_tokens)

        self.total_prompt_tokens    += len(user_tokens)
        self.total_generated_tokens += len(response_tokens)

        self._trim_history_if_needed()

    def _trim_history_if_needed(self):
        if len(self.history_tokens) > self.max_history_tokens:
            while len(self.history_tokens) > self.max_history_tokens and len(self.history) >= 2:
                self.history = self.history[2:]
                user_turn      = self.history[0][1]
                assistant_turn = self.history[1][1]
                user_tokens      = len(self._tokenize(f"{self.human_prefix}{user_turn}{self.end_of_turn}"))
                assistant_tokens = len(self._tokenize(f"{self.assistant_prefix}{assistant_turn}{self.end_of_turn}"))
                self.history_tokens = self.history_tokens[user_tokens + assistant_tokens:]

    def _should_stop_generation(self, text):
        for marker in self.end_markers:
            if marker in text:
                return True
        return False

    def _clean_token_text(self, text):
        return text.replace("<|endoftext|>", "")

    def _is_repetitive(self, tokens, window=10):
        if len(tokens) < window:
            return False
        recent = tokens[-window:]
        if len(set(recent)) < 3:
            return True
        for pattern_len in [2, 3, 4]:
            if len(recent) >= pattern_len * 2:
                pattern      = tuple(recent[-pattern_len:])
                prev_pattern = tuple(recent[-pattern_len*2:-pattern_len])
                if pattern == prev_pattern:
                    return True
        return False

    def _set_hrm_steps(self, steps):
        self.model.config.hrm_max_steps = steps
        self.model.hrm_core.config.hrm_max_steps = steps

    def _restore_hrm_steps(self):
        self.model.config.hrm_max_steps = self.original_hrm_max_steps
        self.model.hrm_core.config.hrm_max_steps = self.original_hrm_max_steps

    def generate_response(self, user_input):
        if self.hrm_forced_steps is not None:
            self._set_hrm_steps(self.hrm_forced_steps)

        try:
            full_prompt   = self._format_prompt(user_input)
            prompt_tokens = self._tokenize(full_prompt)
            input_ids     = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

            if self.debug_mode:
                print(f"\n[DEBUG] Prompt tokens: {len(prompt_tokens)}")
                print(f"[DEBUG] HRM mode: {'auto' if self.hrm_forced_steps is None else self.hrm_forced_steps} (model max: {self.model.config.hrm_max_steps})")

            generated_tokens      = []
            accumulated_text      = ""
            repetition_processor  = RepetitionPenaltyLogitsProcessor(self.repetition_penalty)
            total_hrm_steps       = 0

            with torch.no_grad():
                for step in range(self.config.max_new_tokens):
                    context = input_ids[:, -self.block_size:] if input_ids.size(1) > self.block_size else input_ids

                    logits, _, steps_taken, _ = self.model(context)
                    total_hrm_steps += steps_taken.item()

                    logits = logits[:, -1, :] / self.config.temperature
                    logits = repetition_processor(context, logits)

                    if self.config.top_k > 0:
                        v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float('-inf')

                    probs      = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    if next_token.item() == 50256:
                        break

                    token_text = self._clean_token_text(self.tokenizer.decode([next_token.item()]))
                    generated_tokens.append(next_token.item())
                    accumulated_text += token_text

                    if self._should_stop_generation(accumulated_text):
                        for marker in self.end_markers:
                            if marker in accumulated_text:
                                accumulated_text = accumulated_text.split(marker)[0]
                                break
                        break

                    if self._is_repetitive(generated_tokens):
                        if self.debug_mode:
                            print("\n[DEBUG] Detected repetition, stopping")
                        break

                    yield (token_text, accumulated_text, False)

                    input_ids = torch.cat([input_ids, next_token], dim=1)

                    if step < self.min_tokens_to_generate:
                        continue

            final_response = accumulated_text.strip()
            for marker in self.end_markers:
                if final_response.endswith(marker.strip()):
                    final_response = final_response[:-len(marker.strip())].strip()

            self.total_hrm_steps_used += total_hrm_steps

            if self.debug_mode:
                avg_steps = total_hrm_steps / len(generated_tokens) if generated_tokens else 0
                print(f"\n[DEBUG] Generated {len(generated_tokens)} tokens | Total HRM steps: {total_hrm_steps} | Avg steps/token: {avg_steps:.1f}")

            self._update_history(user_input, final_response)
            yield (None, final_response, True)

        finally:
            if self.hrm_forced_steps is not None:
                self._restore_hrm_steps()

    def execute_command(self, command):
        command_lower = command.lower().strip()

        if command_lower in ['/exit', '/quit', '/q']:
            print(colored("Goodbye!", 'cyan'))
            return False

        elif command_lower == '/help':
            self._print_welcome_message()

        elif command_lower == '/clear':
            self.history = []
            self.history_tokens = []
            print(colored("Conversation history cleared.", 'yellow'))

        elif command_lower == '/stats':
            self._print_stats()

        elif command_lower == '/debug':
            self.debug_mode = not self.debug_mode
            print(colored(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}.", 'yellow'))

        elif command_lower.startswith('/temp '):
            try:
                temp = float(command.split()[1])
                if 0.1 <= temp <= 2.0:
                    self.config.temperature = temp
                    print(colored(f"Temperature set to {temp}", 'yellow'))
                else:
                    print(colored("Temperature must be between 0.1 and 2.0", 'red'))
            except:
                print(colored("Usage: /temp [value]", 'red'))

        elif command_lower.startswith('/penalty '):
            try:
                penalty = float(command.split()[1])
                if 1.0 <= penalty <= 2.0:
                    self.repetition_penalty = penalty
                    print(colored(f"Repetition penalty set to {penalty}", 'yellow'))
                else:
                    print(colored("Penalty must be between 1.0 and 2.0", 'red'))
            except:
                print(colored("Usage: /penalty [value]", 'red'))

        elif command_lower.startswith('/hrm '):
            try:
                hrm_arg = command.split()[1].lower()
                if hrm_arg == 'auto':
                    self.hrm_forced_steps = 8
                    print(colored(f"HRM mode set to AUTO (model will use up to {self.original_hrm_max_steps} steps)", 'yellow'))
                else:
                    steps = int(hrm_arg)
                    if 0 <= steps <= 9999:
                        self.hrm_forced_steps = steps
                        print(colored(f"HRM forced to {steps} step(s)", 'yellow'))
                        if steps == 0:
                            print(colored("Warning: HRM with 0 steps means no iterative reasoning!", 'red'))
                    else:
                        print(colored("HRM steps must be between 0 and 9999", 'red'))
            except:
                print(colored("Usage: /hrm [number] or /hrm auto", 'red'))

        elif command_lower.startswith('/save '):
            try:
                self._save_conversation(command.split(maxsplit=1)[1])
            except:
                print(colored("Usage: /save [filename]", 'red'))

        elif command_lower.startswith('/load '):
            try:
                self._load_conversation(command.split(maxsplit=1)[1])
            except:
                print(colored("Usage: /load [filename]", 'red'))

        else:
            print(colored(f"Unknown command: {command}", 'red'))
            print(colored("Type /help for available commands", 'yellow'))

        return True

    def _print_stats(self):
        avg_hrm  = self.total_hrm_steps_used / self.total_generated_tokens if self.total_generated_tokens > 0 else 0
        hrm_mode = "AUTO" if self.hrm_forced_steps is None else f"FORCED ({self.hrm_forced_steps})"
        print(colored(f"""
{'=' * 60}
CONVERSATION STATISTICS
{'=' * 60}
Prompt tokens:      {self.total_prompt_tokens:,}
Generated tokens:   {self.total_generated_tokens:,}
Total HRM steps:    {self.total_hrm_steps_used:,}
Avg HRM steps/tok:  {avg_hrm:.2f}
Turns:              {len(self.history) // 2}
History tokens:     {len(self.history_tokens):,}

Temperature:        {self.config.temperature}
Repetition penalty: {self.repetition_penalty}
HRM mode:           {hrm_mode}
Model max HRM steps:{self.original_hrm_max_steps}
Top-k:              {self.config.top_k}
{'=' * 60}
""", 'cyan'))

    def _save_conversation(self, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("HRM-CosmicFish Conversation\n")
                f.write(f"{'=' * 80}\n\n")
                for role, text in self.history:
                    prefix = "Human: " if role == "human" else "Assistant: "
                    f.write(f"{prefix}{text}\n\n")
            print(colored(f"Conversation saved to {filename}", 'green'))
        except Exception as e:
            print(colored(f"Error saving conversation: {e}", 'red'))

    def _load_conversation(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')

            self.history = []
            self.history_tokens = []

            current_role = None
            current_text = []

            for line in lines:
                if line.startswith('Human: '):
                    if current_role and current_text:
                        self.history.append((current_role, '\n'.join(current_text).strip()))
                    current_role = 'human'
                    current_text = [line[7:]]
                elif line.startswith('Assistant: '):
                    if current_role and current_text:
                        self.history.append((current_role, '\n'.join(current_text).strip()))
                    current_role = 'assistant'
                    current_text = [line[11:]]
                elif line.strip() and current_role:
                    current_text.append(line)

            if current_role and current_text:
                self.history.append((current_role, '\n'.join(current_text).strip()))

            print(colored(f"Conversation loaded from {filename} ({len(self.history)//2} turns)", 'green'))
        except Exception as e:
            print(colored(f"Error loading conversation: {e}", 'red'))


def main():
    parser = argparse.ArgumentParser(description="Chat with CosmicFish-HRM model")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--min_tokens", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--human_prefix", type=str, default="Human: ")
    parser.add_argument("--assistant_prefix", type=str, default="Assistant: ")
    parser.add_argument("--end_of_turn", type=str, default="\n\n")
    parser.add_argument("--instruction", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max_history", type=int, default=1024)
    parser.add_argument("--no_welcome", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Loading HRM-CosmicFish model from {args.model_path}...")
    try:
        try:
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=True: {e}")
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("No configuration found in checkpoint")

        model = HRMCosmicFish(config)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            raise ValueError("Could not find model weights in checkpoint")

        cleaned_state_dict = {
            k.removeprefix('_orig_mod.').removeprefix('module.'): v
            for k, v in state_dict.items()
        }

        try:
            model.load_state_dict(cleaned_state_dict)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}, attempting flexible loading...")
            missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}")

        model.to(device)
        model.eval()

        block_size = config.block_size

        print(f"Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
        print(f"  Input blocks: {config.n_input_layers} | HRM: H={config.hrm_H_layers} L={config.hrm_L_layers} (max {config.hrm_max_steps} steps) | Output blocks: {config.n_output_layers}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        return

    class ChatConfig:
        def __init__(self, args, block_size):
            self.device               = args.device
            self.temperature          = args.temperature
            self.max_new_tokens       = args.max_tokens
            self.min_tokens_to_generate = args.min_tokens
            self.top_k                = args.top_k
            self.human_prefix         = args.human_prefix
            self.assistant_prefix     = args.assistant_prefix
            self.end_of_turn          = args.end_of_turn
            self.prompt_template      = args.instruction
            self.max_history_tokens   = args.max_history
            self.display_welcome      = not args.no_welcome
            self.block_size           = block_size
            self.debug_mode           = args.debug
            self.repetition_penalty   = args.repetition_penalty

    chat = ChatSession(model, tokenizer, ChatConfig(args, block_size))

    print(colored("\nHRM-CosmicFish initialized. Type your message (or /help for commands).\n", 'cyan'))

    while True:
        try:
            user_input = input(colored("You: ", 'green'))

            if user_input.startswith('/'):
                if not chat.execute_command(user_input):
                    break
                continue

            if not user_input.strip():
                continue

            live_buffer    = ""
            final_response = None

            response_generator = chat.generate_response(user_input)

            try:
                print(colored("CosmicFish: ", 'blue'), end="")
                sys.stdout.flush()

                for token, live_text, is_done in response_generator:
                    if is_done:
                        final_response = live_text
                        if not live_buffer:
                            print(final_response, end="")
                        break

                    if token:
                        if "<|endoftext|>" in token:
                            token = token.replace("<|endoftext|>", "")
                            if token:
                                print(token, end="", flush=True)
                            break
                        print(token, end="", flush=True)
                        live_buffer += token

            except KeyboardInterrupt:
                print("\n[Generation interrupted]")

            print()

        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt. Type /exit to quit or continue chatting.")

        except Exception as e:
            print(colored(f"\nError: {str(e)}", 'red'))
            logger.error(f"Error in chat loop: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)