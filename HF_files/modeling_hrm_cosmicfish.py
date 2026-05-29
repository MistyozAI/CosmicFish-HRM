import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


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
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    # xq, xk: [B, n_heads, T, head_dim], freqs_cis: [T, head_dim/2]
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
        assert config.n_embd % config.n_head == 0

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

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

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
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool(), float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        if config.use_swiglu:
            self.gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.act = nn.SiLU()
        else:
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.act = nn.GELU()

        self.dropout = nn.Dropout(config.dropout)
        self.use_swiglu = config.use_swiglu

    def forward(self, x):
        if self.use_swiglu:
            return self.dropout(self.down(self.act(self.up(x)) * self.gate(x)))
        else:
            return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.eps)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.eps)
        self.mlp = MLP(config)

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
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None):
        # Post-norm architecture for HRM
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
        self.config = config

        self.H_level = HRMReasoningLevel(config, config.hrm_H_layers)
        self.L_level = HRMReasoningLevel(config, config.hrm_L_layers)

        self.H_init = nn.Parameter(torch.randn(config.n_embd) * 0.02)
        self.L_init = nn.Parameter(torch.randn(config.n_embd) * 0.02)

        self.q_head = nn.Linear(config.n_embd, 2, bias=True)  # [halt, continue]

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)  # Bias towards halting

    def forward(self, x, freqs_cis=None, training=False):
        B, T, C = x.size()
        device = x.device

        z_H = self.H_init.expand(B, T, C)
        z_L = self.L_init.expand(B, T, C)

        steps_taken = torch.zeros(B, dtype=torch.long, device=device)
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        q_logits_list = []

        for step in range(self.config.hrm_max_steps):
            if halted.all():
                break

            with torch.set_grad_enabled(step == self.config.hrm_max_steps - 1):
                for _h in range(self.config.hrm_H_cycles):
                    for _l in range(self.config.hrm_L_cycles):
                        z_L = self.L_level(z_L, z_H + x, freqs_cis)
                    z_H = self.H_level(z_H, z_L, freqs_cis)

            q_input = z_H.mean(dim=1)  # [B, n_embd]
            q_logits = self.q_head(q_input.float())  # [B, 2]
            q_logits_list.append(q_logits)

            if self.config.hrm_max_steps > 1:
                q_halt = q_logits[:, 0]
                q_continue = q_logits[:, 1]

                if not training:
                    q_halt = q_halt + 0.35  # tune this value (try 1.0, 2.0, 3.0)

                should_halt = q_halt > q_continue

                if training and torch.rand(1).item() < self.config.hrm_exploration_prob:
                    min_steps = torch.randint(2, self.config.hrm_max_steps + 1, (1,)).item()
                    should_halt = should_halt & (steps_taken >= min_steps)

                halted = halted | should_halt

            steps_taken = torch.where(halted, steps_taken, steps_taken + 1)

            if step == self.config.hrm_max_steps - 1:
                halted = torch.ones_like(halted)

        output_q_logits = q_logits_list[-1] if q_logits_list else None
        return z_H, steps_taken, output_q_logits


class HRMCosmicFish(nn.Module):
    """
    Architecture: Input Blocks → HRM Reasoning Core → Output Blocks → LM Head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        if config.use_rotary:
            self.freqs_cis = precompute_freqs_cis(
                config.n_embd // config.n_head,
                config.block_size
            )
        else:
            self.freqs_cis = None
            self.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        self.input_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_input_layers)
        ])

        self.hrm_core = HRMCore(config)

        self.output_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_output_layers)
        ])

        self.ln_f = RMSNorm(config.n_embd, eps=config.eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down.weight'):
                total_layers = config.n_input_layers + config.n_output_layers + config.hrm_H_layers + config.hrm_L_layers
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * total_layers))

        print(f"Model initialized with {self.get_num_params() / 1e6:.2f}M parameters")
        print(f"  Input blocks: {config.n_input_layers} layers")
        print(f"  HRM Core: H={config.hrm_H_layers} L={config.hrm_L_layers} (max {config.hrm_max_steps} steps)")
        print(f"  Output blocks: {config.n_output_layers} layers")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'wpe'):
            n_params -= self.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        x = self.wte(idx)

        if self.config.use_rotary:
            freqs_cis = self.freqs_cis.to(device) if self.freqs_cis is not None else None
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.wpe(pos)
            freqs_cis = None

        x = self.drop(x)

        for block in self.input_blocks:
            x = block(x, freqs_cis)

        x, steps_taken, q_logits = self.hrm_core(x, freqs_cis, training=self.training)

        for block in self.output_blocks:
            x = block(x, freqs_cis)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            step_penalty = 0.01 * steps_taken.float().mean()  # penalize using more steps
            loss = task_loss + step_penalty

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

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx