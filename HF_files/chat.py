import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from termcolor import colored
import logging
import readline
import re
import textwrap
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import json

try:
    from safetensors.torch import load_file
except ImportError:
    print("safetensors not installed. Run: pip install safetensors")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

try:
    from transformers import GPT2Tokenizer
except ImportError:
    print("transformers not installed. Run: pip install transformers")
    sys.exit(1)

HF_REPO = "MistyozAI/CosmicFish-HRM"


@dataclass
class HRMCosmicFishConfig:
    vocab_size: int = 50304
    n_embd: int = 448
    block_size: int = 512
    n_input_layers: int = 6
    n_output_layers: int = 6
    n_head: int = 8
    hrm_H_layers: int = 4
    hrm_L_layers: int = 4
    hrm_H_cycles: int = 2
    hrm_L_cycles: int = 2
    hrm_max_steps: int = 16
    hrm_exploration_prob: float = 0.1
    dropout: float = 0.1
    bias: bool = False
    use_rotary: bool = True
    use_gqa: bool = True
    use_swiglu: bool = True
    n_kv_head: int = 4
    eps: float = 1e-5
    forward_dtype: str = "bfloat16"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    freqs_cis = freqs_cis[:, :, :xq_.shape[2], :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.use_gqa else config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, freqs_cis=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        if config.use_swiglu:
            self.gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.up   = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.act  = nn.SiLU()
        else:
            self.c_fc   = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.act    = nn.GELU()
        self.dropout    = nn.Dropout(config.dropout)
        self.use_swiglu = config.use_swiglu

    def forward(self, x):
        if self.use_swiglu:
            return self.dropout(self.down(self.act(self.up(x)) * self.gate(x)))
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.eps)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.eps)
        self.mlp  = MLP(config)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


class HRMReasoningBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.eps)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.eps)
        self.mlp  = MLP(config)

    def forward(self, x, freqs_cis=None):
        x = self.ln_1(x + self.attn(x, freqs_cis))
        x = self.ln_2(x + self.mlp(x))
        return x


class HRMReasoningLevel(nn.Module):
    def __init__(self, config, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([HRMReasoningBlock(config) for _ in range(n_layers)])

    def forward(self, hidden_states, input_injection, freqs_cis=None):
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis)
        return hidden_states


class HRMCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.H_level = HRMReasoningLevel(config, config.hrm_H_layers)
        self.L_level = HRMReasoningLevel(config, config.hrm_L_layers)
        self.H_init  = nn.Parameter(torch.randn(config.n_embd) * 0.02)
        self.L_init  = nn.Parameter(torch.randn(config.n_embd) * 0.02)
        self.q_head  = nn.Linear(config.n_embd, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

    def forward(self, x, freqs_cis=None, training=False):
        B, T, C   = x.size()
        device    = x.device
        z_H       = self.H_init.expand(B, T, C)
        z_L       = self.L_init.expand(B, T, C)
        steps_taken = torch.zeros(B, dtype=torch.long, device=device)
        halted    = torch.zeros(B, dtype=torch.bool, device=device)
        q_logits_list = []

        for step in range(self.config.hrm_max_steps):
            if halted.all():
                break
            with torch.set_grad_enabled(step == self.config.hrm_max_steps - 1):
                for _h in range(self.config.hrm_H_cycles):
                    for _l in range(self.config.hrm_L_cycles):
                        z_L = self.L_level(z_L, z_H + x, freqs_cis)
                    z_H = self.H_level(z_H, z_L, freqs_cis)
            q_input  = z_H.mean(dim=1)
            q_logits = self.q_head(q_input.float())
            q_logits_list.append(q_logits)

            if self.config.hrm_max_steps > 1:
                q_halt     = q_logits[:, 0]
                q_continue = q_logits[:, 1]
                if not training:
                    q_halt = q_halt + 0.35
                should_halt = q_halt > q_continue
                halted = halted | should_halt

            steps_taken = torch.where(halted, steps_taken, steps_taken + 1)
            if step == self.config.hrm_max_steps - 1:
                halted = torch.ones_like(halted)

        return z_H, steps_taken, (q_logits_list[-1] if q_logits_list else None)


class HRMCosmicFish(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte    = nn.Embedding(config.vocab_size, config.n_embd)

        if config.use_rotary:
            self.freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size)
        else:
            self.freqs_cis = None
            self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.drop          = nn.Dropout(config.dropout)
        self.input_blocks  = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_input_layers)])
        self.hrm_core      = HRMCore(config)
        self.output_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_output_layers)])
        self.ln_f          = RMSNorm(config.n_embd, eps=config.eps)
        self.lm_head       = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight    = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down.weight'):
                total = config.n_input_layers + config.n_output_layers + config.hrm_H_layers + config.hrm_L_layers
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * total))

        print(f"Model initialized with {self.get_num_params() / 1e6:.2f}M parameters")
        print(f"  Input blocks: {config.n_input_layers} layers")
        print(f"  HRM Core: H={config.hrm_H_layers} L={config.hrm_L_layers} (max {config.hrm_max_steps} steps)")
        print(f"  Output blocks: {config.n_output_layers} layers")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'wpe'):
            n_params -= self.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        B, T   = idx.size()
        x      = self.wte(idx)

        if self.config.use_rotary:
            freqs_cis = self.freqs_cis.to(device) if self.freqs_cis is not None else None
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x   = x + self.wpe(pos)
            freqs_cis = None

        x = self.drop(x)
        for block in self.input_blocks:
            x = block(x, freqs_cis)
        x, steps_taken, q_logits = self.hrm_core(x, freqs_cis, training=self.training)
        for block in self.output_blocks:
            x = block(x, freqs_cis)
        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = loss + 0.01 * steps_taken.float().mean()

        return logits, loss, steps_taken, q_logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx

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

    print(f"Downloading CosmicFish-HRM from Hugging Face ({HF_REPO})...")
    try:
        cache_dir = snapshot_download(repo_id=HF_REPO)
        logger.info(f"Model cached at: {cache_dir}")

        config_path  = os.path.join(cache_dir, "config.json")
        weights_path = os.path.join(cache_dir, "model.safetensors")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {cache_dir}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"model.safetensors not found in {cache_dir}")

        with open(config_path) as f:
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
            hrm_exploration_prob=cfg["hrm_exploration_prob"],
            dropout=0.0,
            bias=cfg["bias"],
            use_rotary=cfg["use_rotary"],
            use_gqa=cfg["use_gqa"],
            use_swiglu=cfg["use_swiglu"],
            eps=cfg["eps"],
        )

        model = HRMCosmicFish(config)

        state_dict = load_file(weights_path, device=device)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}, attempting flexible loading...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
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
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        return

    class ChatConfig:
        def __init__(self, args, block_size, device):
            self.device               = device
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

    chat = ChatSession(model, tokenizer, ChatConfig(args, block_size, device))

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