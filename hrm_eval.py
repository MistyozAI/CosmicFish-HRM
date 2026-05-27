"""
hrm_eval.py — Logits-based benchmark evaluation for CosmicFish-HRM
Currently: HellaSwag, PIQA, WinoGrande, Natural Questions (NQ) - More to come...
"""

import os
import sys
import json
import re
import math
import time
import argparse
import logging
import urllib.request
import urllib.error
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
import tiktoken
from torch.serialization import add_safe_globals

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import HRMCosmicFish, HRMCosmicFishConfig

add_safe_globals([HRMCosmicFishConfig])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@contextmanager
def patched_halt_bias(model: HRMCosmicFish, halt_bias: float):
    """Temporarily override the inference halt bias in HRMCore.forward."""
    original_forward = model.hrm_core.__class__.forward

    def patched_forward(self, x, freqs_cis=None, training=False):
        B, T, C = x.size()
        device = x.device

        z_H = self.H_init.expand(B, T, C)
        z_L = self.L_init.expand(B, T, C)

        steps_taken   = torch.zeros(B, dtype=torch.long, device=device)
        halted        = torch.zeros(B, dtype=torch.bool,  device=device)
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
                    q_halt = q_halt + halt_bias

                should_halt = q_halt > q_continue

                if training and torch.rand(1).item() < self.config.hrm_exploration_prob:
                    min_steps   = torch.randint(2, self.config.hrm_max_steps + 1, (1,)).item()
                    should_halt = should_halt & (steps_taken >= min_steps)

                halted = halted | should_halt

            steps_taken = torch.where(halted, steps_taken, steps_taken + 1)

            if step == self.config.hrm_max_steps - 1:
                halted = torch.ones_like(halted)

        output_q_logits = q_logits_list[-1] if q_logits_list else None
        return z_H, steps_taken, output_q_logits

    model.hrm_core.__class__.forward = patched_forward
    try:
        yield
    finally:
        model.hrm_core.__class__.forward = original_forward


@dataclass
class MCExample:
    context: str
    choices: List[str]
    label: int


def _download(url: str, dest: str) -> None:
    log.info(f"  Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {e}") from e


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("[header]", "")
    text = text.replace("[step]", "")
    text = text.replace("[substeps]", "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def load_hellaswag(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    log.info("Loading HellaSwag …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "hellaswag_val.jsonl")

    if not os.path.exists(cache_path):
        url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        try:
            _download(url, cache_path)
            log.info(f"  Saved to {cache_path}")
        except Exception as e:
            log.warning(f"  GitHub download failed ({e}), trying HuggingFace parquet …")
            try:
                import pandas as pd
                pq_url  = "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet"
                pq_path = cache_path + ".parquet"
                _download(pq_url, pq_path)
                df = pd.read_parquet(pq_path)
                with open(cache_path, "w") as f:
                    for _, row in df.iterrows():
                        f.write(json.dumps(row.to_dict()) + "\n")
                os.remove(pq_path)
                log.info(f"  Saved {len(df)} examples to {cache_path}")
            except Exception as e2:
                log.error(f"  All download methods failed: {e2}")
                log.error("  Install pandas + pyarrow: pip install pandas pyarrow")
                sys.exit(1)

    examples: List[MCExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item    = json.loads(line)
            ctx     = _hellaswag_preprocess(item["ctx"])
            choices = [_hellaswag_preprocess(e) for e in item["endings"]]
            label   = int(item["label"])
            examples.append(MCExample(context=ctx, choices=choices, label=label))

    if quick:
        examples = examples[:100]
    log.info(f"  HellaSwag: {len(examples)} examples")
    return examples


def load_piqa(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    log.info("Loading PIQA …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "piqa_val.jsonl")

    if not os.path.exists(cache_path):
        goals_url   = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl"
        labels_url  = "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst"
        goals_path  = os.path.join(cache_dir, "piqa_valid.jsonl")
        labels_path = os.path.join(cache_dir, "piqa_valid_labels.lst")

        try:
            _download(goals_url, goals_path)
            _download(labels_url, labels_path)
        except Exception as e:
            log.warning(f"  GitHub download failed ({e}), trying HuggingFace parquet …")
            try:
                import pandas as pd
                pq_url  = "https://huggingface.co/datasets/ybisk/piqa/resolve/main/data/validation-00000-of-00001.parquet"
                pq_path = cache_path + ".parquet"
                _download(pq_url, pq_path)
                df = pd.read_parquet(pq_path)
                with open(cache_path, "w") as f:
                    for _, row in df.iterrows():
                        f.write(json.dumps(row.to_dict()) + "\n")
                os.remove(pq_path)
                log.info(f"  Saved {len(df)} examples via parquet to {cache_path}")
                goals_path = None
            except Exception as e2:
                log.error(f"  All download methods failed: {e2}")
                sys.exit(1)

        if goals_path is not None and os.path.exists(goals_path):
            with open(goals_path) as gf, open(labels_path) as lf, open(cache_path, "w") as out:
                for goal_line, label_line in zip(gf, lf):
                    item = json.loads(goal_line)
                    item["label"] = int(label_line.strip())
                    out.write(json.dumps(item) + "\n")
            log.info(f"  Saved to {cache_path}")

    examples: List[MCExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item  = json.loads(line)
            goal  = item.get("goal",  item.get("ctx",     "")).strip()
            sol1  = item.get("sol1",  item.get("choice1", "")).strip()
            sol2  = item.get("sol2",  item.get("choice2", "")).strip()
            label = int(item["label"])
            if not goal or not sol1 or not sol2:
                continue
            examples.append(MCExample(context=goal, choices=[sol1, sol2], label=label))

    if quick:
        examples = examples[:100]
    log.info(f"  PIQA: {len(examples)} examples")
    return examples


def load_winogrande(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[MCExample]:
    log.info("Loading WinoGrande …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "winogrande_val.jsonl")

    if not os.path.exists(cache_path):
        parquet_url = (
            "https://huggingface.co/datasets/allenai/winogrande/resolve/main/"
            "data/validation-00000-of-00001.parquet"
        )
        downloaded = False
        try:
            import pandas as pd
            pq_path = cache_path + ".parquet"
            _download(parquet_url, pq_path)
            df = pd.read_parquet(pq_path)
            with open(cache_path, "w") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")
            os.remove(pq_path)
            log.info(f"  Saved {len(df)} examples to {cache_path}")
            downloaded = True
        except Exception as e:
            log.warning(f"  Parquet download failed ({e}), trying zip …")

        if not downloaded:
            try:
                import zipfile
                data_url = "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"
                zip_path = os.path.join(cache_dir, "winogrande.zip")
                _download(data_url, zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    candidates = [n for n in zf.namelist() if "dev" in n and n.endswith(".jsonl")]
                    if not candidates:
                        candidates = [n for n in zf.namelist() if n.endswith(".jsonl")]
                    if not candidates:
                        raise RuntimeError("No JSONL found in WinoGrande zip")
                    chosen = next((c for c in candidates if "xl" in c and "dev" in c), candidates[0])
                    log.info(f"  Extracting {chosen}")
                    with zf.open(chosen) as src, open(cache_path, "wb") as dst:
                        dst.write(src.read())
                os.remove(zip_path)
            except Exception as e2:
                log.error(f"  All download methods failed: {e2}")
                log.error("  Try: pip install pandas pyarrow")
                sys.exit(1)

    examples: List[MCExample] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item     = json.loads(line)
            sentence = item["sentence"]
            opt1     = item["option1"].strip()
            opt2     = item["option2"].strip()
            answer   = str(item.get("answer", item.get("label", "1")))
            label    = int(answer) - 1  # "1"/"2" → 0/1

            parts = sentence.split("_", 1)
            if len(parts) == 2:
                prefix, suffix = parts[0], parts[1]
                ctx     = prefix.rstrip()
                choices = [opt1 + suffix, opt2 + suffix]
            else:
                ctx     = ""
                choices = [sentence.replace("_", opt1), sentence.replace("_", opt2)]

            examples.append(MCExample(context=ctx, choices=choices, label=label))

    if quick:
        examples = examples[:100]
    log.info(f"  WinoGrande: {len(examples)} examples")
    return examples


@dataclass
class NQExample:
    question:     str
    gold_answers: List[str]


def load_nq(quick: bool = False, cache_dir: str = ".benchmark_cache") -> List[NQExample]:
    log.info("Loading Natural Questions (dev, short answers) …")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "nq_dev.jsonl")

    if not os.path.exists(cache_path):
        parquet_url = (
            "https://huggingface.co/datasets/google-research-datasets/nq_open"
            "/resolve/refs%2Fconvert%2Fparquet/nq_open/validation/0000.parquet"
        )
        fallback_url = (
            "https://huggingface.co/datasets/nq_open"
            "/resolve/refs%2Fconvert%2Fparquet/nq_open/validation/0000.parquet"
        )
        downloaded = False

        for url in [parquet_url, fallback_url]:
            if downloaded:
                break
            try:
                import pandas as pd
                import numpy as np
                pq_path = cache_path + ".parquet"
                _download(url, pq_path)
                df = pd.read_parquet(pq_path)
                os.remove(pq_path)
                with open(cache_path, "w") as f:
                    for _, row in df.iterrows():
                        d = row.to_dict()
                        for k, v in d.items():
                            if isinstance(v, np.ndarray):
                                d[k] = v.tolist()
                            elif hasattr(v, "tolist"):
                                d[k] = v.tolist()
                        f.write(json.dumps(d) + "\n")
                log.info(f"  Saved {len(df)} rows to {cache_path}")
                downloaded = True
            except Exception as e:
                log.warning(f"  Download failed ({url}): {e}")

        if not downloaded:
            log.error("  All NQ download attempts failed.")
            log.error("  Try: pip install pandas pyarrow")
            sys.exit(1)

    examples: List[NQExample] = []
    skipped = 0
    with open(cache_path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if line_num == 0:
                log.info(f"  NQ first-row keys: {list(item.keys())}")
                for k, v in item.items():
                    log.info(f"    {k}: {type(v).__name__} = {repr(v)[:120]}")

            question = str(item.get("question", "")).strip()
            if not question:
                skipped += 1
                continue

            raw = item.get("answer", item.get("answers", []))
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = [raw]
            if not isinstance(raw, list):
                raw = list(raw)

            gold = [str(a).strip() for a in raw if str(a).strip()]
            if not gold:
                skipped += 1
                continue

            examples.append(NQExample(question=question, gold_answers=gold))

    log.info(f"  NQ: {len(examples)} examples loaded  ({skipped} skipped — no short answer)")
    if quick:
        examples = examples[:100]
    return examples


def evaluate_nq(
    model: "HRMCosmicFish",
    config: "HRMCosmicFishConfig",
    tokenizer,
    examples: List[NQExample],
    device: str,
    batch_log_every: int = 50,
) -> Dict:
    import string

    def _normalise(s: str) -> str:
        s = s.lower().strip()
        for article in (" a ", " an ", " the "):
            if s.startswith(article.strip()):
                s = s[len(article.strip()):].strip()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = " ".join(s.split())
        return s

    correct   = 0
    total     = len(examples)
    t0        = time.time()
    all_steps: List[int] = []

    PROMPT_TEMPLATE = "Question: {q}\nAnswer:"
    MAX_NEW_TOKENS  = 20
    EOS_ID          = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    NEWLINE_IDS     = set(tokenizer.encode("\n"))

    for i, ex in enumerate(examples):
        prompt      = PROMPT_TEMPLATE.format(q=ex.question)
        context_ids = tokenizer.encode(prompt)

        generated_ids: List[int] = []
        token_steps:   List[int] = []

        for _ in range(MAX_NEW_TOKENS):
            all_ids   = context_ids + generated_ids
            max_len   = config.block_size
            if len(all_ids) > max_len:
                all_ids = all_ids[-max_len:]

            input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                targets = torch.full_like(input_ids, -1)
                logits, _, steps_taken, _ = model(input_ids, targets=targets)

            next_token_logits = logits[0, -1, :]
            next_id           = int(next_token_logits.argmax().item())
            steps             = int(steps_taken[0].item())
            token_steps.append(steps)

            if next_id == EOS_ID or next_id in NEWLINE_IDS:
                break
            generated_ids.append(next_id)

        if token_steps:
            all_steps.append(sum(token_steps) // len(token_steps))

        generated_text = tokenizer.decode(generated_ids).strip()
        pred_norm      = _normalise(generated_text)
        gold_norms     = [_normalise(a) for a in ex.gold_answers]

        if pred_norm and any(pred_norm == g for g in gold_norms):
            correct += 1

        if (i + 1) % batch_log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc     = correct / (i + 1) * 100
            speed   = (i + 1) / elapsed
            mean_s  = sum(all_steps) / len(all_steps) if all_steps else 0.0
            log.info(
                f"  [nq] {i+1}/{total}  "
                f"acc={acc:.1f}%  {speed:.1f} ex/s  "
                f"hrm_steps_mean={mean_s:.2f}"
            )

    if total == 0:
        log.error("  NQ: 0 examples — delete .benchmark_cache/nq_dev.jsonl and retry.")
        return {
            "benchmark": "nq", "accuracy": 0.0, "correct": 0, "total": 0,
            "elapsed_s": 0.0, "hrm_steps_mean": 0.0, "hrm_steps_stdev": 0.0,
            "hrm_steps_min": 0, "hrm_steps_max": 0, "hrm_steps_all": [],
        }

    accuracy = correct / total
    elapsed  = time.time() - t0

    import statistics
    steps_mean  = statistics.mean(all_steps)  if all_steps          else 0.0
    steps_stdev = statistics.stdev(all_steps) if len(all_steps) > 1 else 0.0
    steps_min   = min(all_steps)              if all_steps          else 0
    steps_max   = max(all_steps)              if all_steps          else 0

    return {
        "benchmark":       "nq",
        "accuracy":        accuracy,
        "correct":         correct,
        "total":           total,
        "elapsed_s":       round(elapsed, 1),
        "hrm_steps_mean":  steps_mean,
        "hrm_steps_stdev": steps_stdev,
        "hrm_steps_min":   steps_min,
        "hrm_steps_max":   steps_max,
        "hrm_steps_all":   all_steps,
    }


def load_model(checkpoint_path: str, device: str) -> Tuple[HRMCosmicFish, HRMCosmicFishConfig]:
    log.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = None
    for key in ("config", "hrm_config", "model_config", "cosmicconf"):
        if key in checkpoint:
            config = checkpoint[key]
            break

    if config is None:
        log.warning("No config found in checkpoint — using defaults")
        config = HRMCosmicFishConfig()

    if not isinstance(config, HRMCosmicFishConfig):
        if isinstance(config, dict):
            config = HRMCosmicFishConfig(**{
                k: v for k, v in config.items()
                if k in HRMCosmicFishConfig.__dataclass_fields__
            })
        else:
            log.warning(f"Unexpected config type {type(config)}, using defaults")
            config = HRMCosmicFishConfig()

    log.info(
        f"  Config: embd={config.n_embd}  heads={config.n_head}  "
        f"block={config.block_size}  "
        f"H_layers={config.hrm_H_layers}  L_layers={config.hrm_L_layers}  "
        f"max_steps={config.hrm_max_steps}  "
        f"RoPE={config.use_rotary}  GQA={config.use_gqa}  SwiGLU={config.use_swiglu}"
    )

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = HRMCosmicFish(config)

    raw_sd = None
    for key in ("model_state_dict", "model", "state_dict"):
        if key in checkpoint:
            raw_sd = checkpoint[key]
            break
    if raw_sd is None:
        raise ValueError("Cannot find model weights in checkpoint.")

    sd = {}
    for k, v in raw_sd.items():
        k = k.removeprefix("_orig_mod.").removeprefix("module.")
        sd[k] = v

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning(f"  Missing keys  : {len(missing)}  (first 5: {missing[:5]})")
    if unexpected:
        log.warning(f"  Unexpected keys: {len(unexpected)}  (first 5: {unexpected[:5]})")

    model.to(device)
    model.eval()
    params_m = model.get_num_params() / 1e6
    log.info(f"  Model ready — {params_m:.2f}M parameters")
    log.info(f"  HRM Core: H={config.hrm_H_layers} L={config.hrm_L_layers} (max {config.hrm_max_steps} steps)")
    return model, config


def score_sequence(
    model: HRMCosmicFish,
    config: HRMCosmicFishConfig,
    tokenizer,
    context_ids: List[int],
    continuation_ids: List[int],
    device: str,
) -> Tuple[float, int]:
    """Average per-token log-likelihood of continuation given context. Returns (mean_log_prob, hrm_steps)."""
    all_ids = context_ids + continuation_ids
    max_len = config.block_size

    if len(all_ids) > max_len:
        overflow    = len(all_ids) - max_len
        context_ids = context_ids[overflow:]
        all_ids     = context_ids + continuation_ids

    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        targets = torch.full_like(input_ids, -1)
        logits, _, steps_taken, _ = model(input_ids, targets=targets)

    hrm_steps = int(steps_taken[0].item())

    cont_start   = len(context_ids) - 1
    cont_end     = len(all_ids) - 1
    logits_cont  = logits[0, cont_start:cont_end, :]
    targets_cont = torch.tensor(continuation_ids, dtype=torch.long, device=device)

    log_probs       = F.log_softmax(logits_cont, dim=-1)
    token_log_probs = log_probs[
        torch.arange(len(continuation_ids), device=device),
        targets_cont
    ]

    return token_log_probs.mean().item(), hrm_steps


def evaluate_benchmark(
    model: HRMCosmicFish,
    config: HRMCosmicFishConfig,
    tokenizer,
    examples: List[MCExample],
    benchmark_name: str,
    device: str,
    batch_log_every: int = 50,
) -> Dict:
    correct   = 0
    total     = len(examples)
    t0        = time.time()
    all_steps: List[int] = []

    for i, ex in enumerate(examples):
        context_ids = tokenizer.encode(ex.context)
        scores      = []
        ex_steps    = []

        for choice in ex.choices:
            cont = " " + choice if not choice.startswith(" ") else choice
            continuation_ids = tokenizer.encode(cont)

            if len(continuation_ids) == 0:
                scores.append(-1e9)
                ex_steps.append(0)
                continue

            score, steps = score_sequence(
                model, config, tokenizer, context_ids, continuation_ids, device
            )
            scores.append(score)
            ex_steps.append(steps)

        all_steps.extend(ex_steps)

        pred = scores.index(max(scores))
        if pred == ex.label:
            correct += 1

        if (i + 1) % batch_log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            acc     = correct / (i + 1) * 100
            speed   = (i + 1) / elapsed
            mean_s  = sum(all_steps) / len(all_steps) if all_steps else 0.0
            log.info(
                f"  [{benchmark_name}] {i+1}/{total}  "
                f"acc={acc:.1f}%  {speed:.1f} ex/s  "
                f"hrm_steps_mean={mean_s:.2f}"
            )

    accuracy = correct / total
    elapsed  = time.time() - t0

    import statistics
    steps_mean  = statistics.mean(all_steps)  if all_steps          else 0.0
    steps_stdev = statistics.stdev(all_steps) if len(all_steps) > 1 else 0.0
    steps_min   = min(all_steps)              if all_steps          else 0
    steps_max   = max(all_steps)              if all_steps          else 0

    return {
        "benchmark":       benchmark_name,
        "accuracy":        accuracy,
        "correct":         correct,
        "total":           total,
        "elapsed_s":       round(elapsed, 1),
        "hrm_steps_mean":  steps_mean,
        "hrm_steps_stdev": steps_stdev,
        "hrm_steps_min":   steps_min,
        "hrm_steps_max":   steps_max,
        "hrm_steps_all":   all_steps,
    }


def make_steps_chart(
    results: List[Dict],
    hrm_max_steps: int,
    halt_bias: float,
    out_path: str = "hrm_steps_chart.png",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        log.warning("matplotlib not installed — skipping chart.  pip install matplotlib")
        return

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         12,
        "axes.linewidth":    1.2,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  5,
        "ytick.major.size":  5,
    })

    names  = [r["benchmark"].upper() for r in results]
    means  = np.array([r["hrm_steps_mean"]  for r in results])
    stdevs = np.array([r["hrm_steps_stdev"] for r in results])
    n      = len(names)

    fig_w  = max(12, n * 3.8)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    x     = np.arange(n)
    width = 0.50

    palette = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66", "#77BEDB"]
    colors  = [palette[i % len(palette)] for i in range(n)]

    bars = ax.bar(
        x, means,
        yerr=stdevs,
        width=width,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=8,
        error_kw={"elinewidth": 2.0, "ecolor": "#333333", "capthick": 2.0},
        zorder=3,
    )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    for bar, mean, std in zip(bars, means, stdevs):
        label_y = bar.get_height() + std + hrm_max_steps * 0.022
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{mean:.2f} ± {std:.2f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#222222",
        )

    ax.axhline(
        hrm_max_steps, color="#CC3333", linestyle="--", linewidth=1.5,
        label=f"Max steps ({hrm_max_steps})", zorder=2,
    )
    ax.axhline(
        1, color="#888888", linestyle=":", linewidth=1.2,
        label="Min steps (1)", zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean HRM Reasoning Steps", fontsize=13, labelpad=10)
    ax.set_xlabel("Benchmark", fontsize=13, labelpad=10)
    ax.set_ylim(0, hrm_max_steps * 1.32)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    bias_str = f"halt_bias = {halt_bias:+.2f}"
    ax.set_title(
        f"HRM-CosmicFish: Mean Reasoning Steps per Benchmark\n"
        f"(error bars = ±1 std dev across all forward passes  |  {bias_str})",
        fontsize=13, pad=16,
    )

    ax.legend(fontsize=10, framealpha=0.7, edgecolor="#cccccc", loc="upper right")

    plt.tight_layout(pad=1.8)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    log.info(f"  Chart saved → {out_path}")
    plt.close()


BENCHMARK_MAP = {
    "hellaswag":  load_hellaswag,
    "piqa":       load_piqa,
    "winogrande": load_winogrande,
    "nq":         load_nq,
}

RANDOM_BASELINE = {
    "hellaswag":  25.0,   # 4 choices
    "piqa":       50.0,   # 2 choices
    "winogrande": 50.0,   # 2 choices
    "nq":         None,   # open-ended
}


def main():
    parser = argparse.ArgumentParser(
        description="Logit-based benchmark eval for HRM-Enhanced CosmicFish"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmarks", nargs="+",
                        default=["hellaswag", "piqa", "winogrande", "nq"],
                        choices=list(BENCHMARK_MAP.keys()))
    parser.add_argument("--device", default="auto",
                        help="cpu / cuda / mps / auto  (default: auto)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 100 examples per benchmark")
    parser.add_argument("--cache_dir", default=".benchmark_cache")
    parser.add_argument("--no_chart", action="store_true")
    parser.add_argument("--chart_out", default="hrm_steps_chart.png")
    parser.add_argument(
        "--halt_bias", type=float, default=0.35,
        help="Inference halt bias added to q_halt. Lower = more steps. Default: 0.35"
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    log.info(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)

    tokenizer = tiktoken.get_encoding("gpt2")
    log.info("Tokenizer: gpt2 (tiktoken)")

    if args.quick:
        log.info("Quick mode: 100 examples per benchmark")

    log.info(f"Halt bias: {args.halt_bias:+.3f}  (model.py default: +0.350)")

    results = []
    with patched_halt_bias(model, args.halt_bias):
        for name in args.benchmarks:
            log.info(f"\n{'='*60}")
            log.info(f"Benchmark: {name.upper()}")
            log.info(f"{'='*60}")

            if name == "nq":
                examples = load_nq(quick=args.quick, cache_dir=args.cache_dir)
                result   = evaluate_nq(
                    model=model, config=config, tokenizer=tokenizer,
                    examples=examples, device=device,
                )
            else:
                loader   = BENCHMARK_MAP[name]
                examples = loader(quick=args.quick, cache_dir=args.cache_dir)
                result   = evaluate_benchmark(
                    model=model, config=config, tokenizer=tokenizer,
                    examples=examples, benchmark_name=name, device=device,
                )
            results.append(result)

    print("\n" + "=" * 72)
    print(f"{'ACCURACY RESULTS':^72}")
    print("=" * 72)
    print(f"  {'Benchmark':<14}  {'Accuracy':>8}  {'Correct':>8}  {'Total':>6}  {'vs. Random':>10}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}")

    total_correct  = 0
    total_examples = 0
    for r in results:
        baseline = RANDOM_BASELINE.get(r["benchmark"])
        if baseline is not None:
            delta_str = f"{r['accuracy'] * 100 - baseline:>+.1f}pp"
        else:
            delta_str = "   n/a"
        print(
            f"  {r['benchmark']:<14}  "
            f"{r['accuracy']*100:>7.1f}%  "
            f"{r['correct']:>8}  "
            f"{r['total']:>6}  "
            f"{delta_str:>10}"
        )
        total_correct  += r["correct"]
        total_examples += r["total"]

    if len(results) > 1:
        overall = total_correct / total_examples * 100
        print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*10}")
        print(f"  {'OVERALL':<14}  {overall:>7.1f}%  {total_correct:>8}  {total_examples:>6}")

    print("=" * 72)

    print()
    print("=" * 72)
    header = f"HRM REASONING STEPS  (halt_bias = {args.halt_bias:+.3f})"
    print(f"{header:^72}")
    print("=" * 72)
    print(f"  {'Benchmark':<14}  {'Mean':>8}  {'Std Dev':>8}  {'Min':>5}  {'Max':>5}  {'# Passes':>9}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*9}")
    for r in results:
        n_passes = len(r["hrm_steps_all"])
        print(
            f"  {r['benchmark']:<14}  "
            f"{r['hrm_steps_mean']:>8.3f}  "
            f"{r['hrm_steps_stdev']:>8.3f}  "
            f"{r['hrm_steps_min']:>5}  "
            f"{r['hrm_steps_max']:>5}  "
            f"{n_passes:>9}"
        )
    print(f"  (max possible steps per pass = {config.hrm_max_steps})")
    print("=" * 72)

    if args.quick:
        print("\n  Quick mode was ON — results are from 100-example subsets")

    if not args.no_chart:
        print()
        make_steps_chart(
            results,
            hrm_max_steps=config.hrm_max_steps,
            halt_bias=args.halt_bias,
            out_path=args.chart_out,
        )

    print()


if __name__ == "__main__":
    main()