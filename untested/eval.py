# Note: This script has not been tested yet!
import os
import sys
import json
import re
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

import torch
import tiktoken
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset

from model import HRMCosmicFish, HRMCosmicFishConfig
from torch.serialization import add_safe_globals

add_safe_globals([HRMCosmicFishConfig])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    model_path: str
    benchmark: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_new_tokens: int = 100
    temperature: float = 0.1
    top_k: int = 5
    num_shots: int = 5
    output_dir: str = "eval_results"
    save_predictions: bool = True
    limit: Optional[int] = None
    subjects: Optional[List[str]] = None
    show_hrm_steps: bool = True


def load_checkpoint(model_path: str, device: str) -> Tuple[HRMCosmicFish, HRMCosmicFishConfig]:
    logger.info(f"Loading model from {model_path}...")
    try:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            logger.warning(f"weights_only=True failed: {e}, falling back")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("No config found in checkpoint")

        model = HRMCosmicFish(config)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            raise ValueError("No model weights found in checkpoint")

        cleaned_state_dict = {
            k.removeprefix('_orig_mod.').removeprefix('module.'): v
            for k, v in state_dict.items()
        }

        try:
            model.load_state_dict(cleaned_state_dict)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}, attempting flexible loading...")
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")

        model.to(device)
        model.eval()

        logger.info(f"Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
        logger.info(f"  Input blocks: {config.n_input_layers} | HRM: H={config.hrm_H_layers} L={config.hrm_L_layers} (max {config.hrm_max_steps} steps) | Output blocks: {config.n_output_layers} | Block size: {config.block_size}")

        return model, config

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


class BaseEvaluator:
    def __init__(self, model: HRMCosmicFish, config: HRMCosmicFishConfig, eval_config: EvalConfig):
        self.model = model
        self.config = config
        self.eval_config = eval_config
        self.device = eval_config.device
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.instruction_prefix = "Below is a conversation between a helpful AI assistant and a human. The assistant is knowledgeable, friendly, and provides detailed and accurate responses.\n\n"
        self.stats = {
            'total_examples': 0,
            'correct': 0,
            'total_hrm_steps': 0,
            'total_tokens_generated': 0,
            'predictions': []
        }

    def format_prompt(self, question: str, choices: Optional[List[str]] = None,
                      few_shot_examples: Optional[List[Dict]] = None) -> str:
        raise NotImplementedError

    def extract_answer(self, generation: str, choices: Optional[List[str]] = None) -> str:
        raise NotImplementedError

    def generate_answer(self, prompt: str) -> Tuple[str, int, int]:
        prompt_tokens = self.tokenizer.encode(prompt)
        max_prompt_length = self.config.block_size - self.eval_config.max_new_tokens
        if len(prompt_tokens) > max_prompt_length:
            prompt_tokens = prompt_tokens[-max_prompt_length:]
            logger.warning(f"Truncated prompt to {max_prompt_length} tokens")

        input_ids       = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        total_hrm_steps = 0
        generated_tokens = []

        with torch.no_grad():
            for _ in range(self.eval_config.max_new_tokens):
                context = input_ids[:, -self.config.block_size:] if input_ids.size(1) > self.config.block_size else input_ids

                logits, _, steps_taken, _ = self.model(context)
                total_hrm_steps += steps_taken.item() if isinstance(steps_taken, torch.Tensor) else steps_taken

                logits = logits[:, -1, :] / self.eval_config.temperature
                if self.eval_config.top_k > 0:
                    v, _ = torch.topk(logits, min(self.eval_config.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs      = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == 50256:
                    break

                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if len(generated_tokens) > 5:
                    recent_text = self.tokenizer.decode(generated_tokens[-5:])
                    if '\n' in recent_text or '.' in recent_text:
                        full_text = self.tokenizer.decode(generated_tokens)
                        if re.search(r'^[A-D][\.\)]?\s', full_text.strip()):
                            break

        return self.tokenizer.decode(generated_tokens), total_hrm_steps, len(generated_tokens)

    def evaluate(self, dataset) -> Dict:
        raise NotImplementedError

    def save_results(self, output_path: str):
        results = {
            'benchmark':              self.__class__.__name__,
            'model_path':             self.eval_config.model_path,
            'accuracy':               self.stats['correct'] / max(self.stats['total_examples'], 1),
            'total_examples':         self.stats['total_examples'],
            'correct':                self.stats['correct'],
            'avg_hrm_steps':          self.stats['total_hrm_steps'] / max(self.stats['total_examples'], 1),
            'avg_tokens_generated':   self.stats['total_tokens_generated'] / max(self.stats['total_examples'], 1),
            'predictions':            self.stats['predictions'] if self.eval_config.save_predictions else []
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


class ARCEvaluator(BaseEvaluator):
    def __init__(self, model, config, eval_config, difficulty="easy"):
        super().__init__(model, config, eval_config)
        self.difficulty   = difficulty
        self.dataset_name = f"ARC-{difficulty.capitalize()}"

    def format_prompt(self, question, choices, choice_labels):
        prompt = self.instruction_prefix
        prompt += f"Human: {question}\n\nChoices:\n"
        for label, choice in zip(choice_labels, choices):
            prompt += f"{label}. {choice}\n"
        prompt += "\nPlease select the correct answer (A, B, C, or D).\n\nAssistant: The answer is"
        return prompt

    def extract_answer(self, generation, choice_labels):
        generation = generation.strip().upper()
        match = re.match(r'^([A-D])', generation)
        if match:
            return match.group(1)
        match = re.search(r'(?:answer is|is)\s*([A-D])', generation, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r'([A-D])', generation)
        if match:
            return match.group(1)
        return choice_labels[0] if choice_labels else 'A'

    def evaluate(self, dataset) -> Dict:
        logger.info(f"Evaluating {self.dataset_name}...")
        if self.eval_config.limit:
            dataset = dataset.select(range(min(self.eval_config.limit, len(dataset))))

        for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {self.dataset_name}")):
            question      = example['question']
            choices       = example['choices']['text']
            choice_labels = example['choices']['label']
            correct_answer = example['answerKey']

            generation, hrm_steps, tokens_gen = self.generate_answer(
                self.format_prompt(question, choices, choice_labels)
            )
            predicted_answer = self.extract_answer(generation, choice_labels)
            is_correct       = (predicted_answer == correct_answer)

            self.stats['total_examples'] += 1
            if is_correct:
                self.stats['correct'] += 1
            self.stats['total_hrm_steps']        += hrm_steps
            self.stats['total_tokens_generated'] += tokens_gen

            if self.eval_config.save_predictions:
                self.stats['predictions'].append({
                    'index': idx, 'question': question,
                    'choices': {label: text for label, text in zip(choice_labels, choices)},
                    'correct_answer': correct_answer, 'predicted_answer': predicted_answer,
                    'generation': generation, 'is_correct': is_correct, 'hrm_steps': hrm_steps
                })

            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx+1}/{len(dataset)} | Acc: {self.stats['correct']/self.stats['total_examples']:.2%}")

        accuracy = self.stats['correct'] / self.stats['total_examples']
        avg_hrm  = self.stats['total_hrm_steps'] / self.stats['total_examples']
        logger.info(f"{self.dataset_name}: {accuracy:.2%} ({self.stats['correct']}/{self.stats['total_examples']}) | Random baseline: 25.00% | Avg HRM steps: {avg_hrm:.2f}")
        return self.stats


class MMLUEvaluator(BaseEvaluator):
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]

    def __init__(self, model, config, eval_config, subjects=None):
        super().__init__(model, config, eval_config)
        if subjects is None or subjects == ['all']:
            self.subjects = self.SUBJECTS
        else:
            self.subjects = [s for s in subjects if s in self.SUBJECTS] or self.SUBJECTS
        self.subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    def format_prompt(self, question, choices, few_shot_examples=None):
        prompt = self.instruction_prefix + "Human: "
        if few_shot_examples:
            for ex in few_shot_examples:
                prompt += f"Question: {ex['question']}\n"
                for i, choice in enumerate(ex['choices']):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += f"Answer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:\n\nAssistant: The answer is"
        return prompt

    def extract_answer(self, generation):
        generation = generation.strip().upper()
        match = re.match(r'^([A-D])', generation)
        if match:
            return match.group(1)
        match = re.search(r'(?:answer is|is)\s*([A-D])', generation, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r'([A-D])', generation)
        if match:
            return match.group(1)
        return 'A'

    def evaluate_subject(self, subject):
        logger.info(f"Loading MMLU subject: {subject}")
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            few_shot_examples = None
            if self.eval_config.num_shots > 0:
                dev_dataset = load_dataset("cais/mmlu", subject, split="dev")
                few_shot_examples = [
                    {'question': ex['question'], 'choices': ex['choices'], 'answer': chr(65 + ex['answer'])}
                    for ex in dev_dataset.select(range(min(self.eval_config.num_shots, len(dev_dataset))))
                ]

            if self.eval_config.limit:
                dataset = dataset.select(range(min(self.eval_config.limit, len(dataset))))

            subject_correct = 0
            for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {subject}", leave=False)):
                question       = example['question']
                choices        = example['choices']
                correct_answer = chr(65 + example['answer'])

                generation, hrm_steps, tokens_gen = self.generate_answer(
                    self.format_prompt(question, choices, few_shot_examples)
                )
                predicted_answer = self.extract_answer(generation)
                is_correct       = (predicted_answer == correct_answer)

                self.stats['total_examples'] += 1
                if is_correct:
                    self.stats['correct'] += 1
                    subject_correct += 1
                self.stats['total_hrm_steps']        += hrm_steps
                self.stats['total_tokens_generated'] += tokens_gen
                self.subject_stats[subject]['total'] += 1
                if is_correct:
                    self.subject_stats[subject]['correct'] += 1

                if self.eval_config.save_predictions:
                    self.stats['predictions'].append({
                        'subject': subject, 'index': idx, 'question': question,
                        'choices': choices, 'correct_answer': correct_answer,
                        'predicted_answer': predicted_answer, 'generation': generation,
                        'is_correct': is_correct, 'hrm_steps': hrm_steps
                    })

            subject_acc = subject_correct / len(dataset) if len(dataset) > 0 else 0
            logger.info(f"  {subject}: {subject_acc:.2%} ({subject_correct}/{len(dataset)})")

        except Exception as e:
            logger.error(f"Error evaluating subject {subject}: {e}")

    def evaluate(self) -> Dict:
        logger.info(f"Starting MMLU evaluation on {len(self.subjects)} subjects...")
        for subject in self.subjects:
            self.evaluate_subject(subject)

        accuracy = self.stats['correct'] / self.stats['total_examples'] if self.stats['total_examples'] > 0 else 0
        avg_hrm  = self.stats['total_hrm_steps'] / self.stats['total_examples'] if self.stats['total_examples'] > 0 else 0

        logger.info(f"MMLU Overall: {accuracy:.2%} ({self.stats['correct']}/{self.stats['total_examples']}) | Random baseline: 25.00% | Avg HRM steps: {avg_hrm:.2f}")

        subject_accs = sorted(
            [(s, stats['correct'] / stats['total'] if stats['total'] > 0 else 0)
             for s, stats in self.subject_stats.items()],
            key=lambda x: x[1], reverse=True
        )
        logger.info("Top 10 subjects:")
        for subj, acc in subject_accs[:10]:
            logger.info(f"  {subj}: {acc:.2%}")
        logger.info("Bottom 10 subjects:")
        for subj, acc in subject_accs[-10:]:
            logger.info(f"  {subj}: {acc:.2%}")

        return self.stats


class GSM8KEvaluator(BaseEvaluator):
    def format_prompt(self, question):
        return (
            self.instruction_prefix
            + f"Human: Solve this math problem step by step.\n\n{question}\n\n"
            + "Please show your work and provide the final numerical answer.\n\nAssistant:"
        )

    def extract_answer(self, generation) -> Optional[float]:
        match = re.search(r'####\s*([\d,\.]+)', generation)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except:
                pass

        for pattern in [
            r'(?:answer is|answer:|final answer is|final answer:)\s*\$?\s*([\d,\.]+)',
            r'(?:equals?|=)\s*\$?\s*([\d,\.]+)\s*$',
            r'\$\s*([\d,\.]+)\s*$'
        ]:
            match = re.search(pattern, generation, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1).replace(',', ''))
                except:
                    pass

        numbers = re.findall(r'[\d,]+\.?\d*', generation)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except:
                pass

        return None

    def evaluate(self, dataset) -> Dict:
        logger.info("Evaluating GSM8K...")
        if self.eval_config.limit:
            dataset = dataset.select(range(min(self.eval_config.limit, len(dataset))))

        for idx, example in enumerate(tqdm(dataset, desc="Evaluating GSM8K")):
            question    = example['question']
            answer_text = example['answer']
            match       = re.search(r'####\s*([\d,\.]+)', answer_text)
            if not match:
                logger.warning(f"Could not extract answer from: {answer_text}")
                continue
            correct_answer = float(match.group(1).replace(',', ''))

            original_max_tokens = self.eval_config.max_new_tokens
            self.eval_config.max_new_tokens = 200
            generation, hrm_steps, tokens_gen = self.generate_answer(self.format_prompt(question))
            self.eval_config.max_new_tokens = original_max_tokens

            predicted_answer = self.extract_answer(generation)
            is_correct = predicted_answer is not None and abs(predicted_answer - correct_answer) < 1e-2

            self.stats['total_examples'] += 1
            if is_correct:
                self.stats['correct'] += 1
            self.stats['total_hrm_steps']        += hrm_steps
            self.stats['total_tokens_generated'] += tokens_gen

            if self.eval_config.save_predictions:
                self.stats['predictions'].append({
                    'index': idx, 'question': question,
                    'correct_answer': correct_answer, 'predicted_answer': predicted_answer,
                    'generation': generation, 'is_correct': is_correct, 'hrm_steps': hrm_steps
                })

            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx+1}/{len(dataset)} | Acc: {self.stats['correct']/self.stats['total_examples']:.2%}")

        accuracy = self.stats['correct'] / self.stats['total_examples']
        avg_hrm  = self.stats['total_hrm_steps'] / self.stats['total_examples']
        logger.info(f"GSM8K: {accuracy:.2%} ({self.stats['correct']}/{self.stats['total_examples']}) | Avg HRM steps: {avg_hrm:.2f}")
        return self.stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRM-CosmicFish on benchmarks")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=['arc-easy', 'arc-challenge', 'mmlu', 'gsm8k', 'all'])
    parser.add_argument("--subjects", type=str, nargs='+', default=['all'])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_save_predictions", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_config = EvalConfig(
        model_path=args.model_path,
        benchmark=args.benchmark,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        num_shots=args.num_shots,
        output_dir=args.output_dir,
        save_predictions=not args.no_save_predictions,
        limit=args.limit,
        subjects=args.subjects
    )

    model, config = load_checkpoint(args.model_path, args.device)

    timestamp          = time.strftime('%Y%m%d_%H%M%S')
    benchmarks_to_run  = ['arc-easy', 'arc-challenge', 'mmlu', 'gsm8k'] if args.benchmark == 'all' else [args.benchmark]
    all_results        = {}

    for benchmark in benchmarks_to_run:
        logger.info(f"\nRunning benchmark: {benchmark}\n")

        if benchmark == 'arc-easy':
            dataset   = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            evaluator = ARCEvaluator(model, config, eval_config, difficulty="easy")
            results   = evaluator.evaluate(dataset)
            evaluator.save_results(os.path.join(args.output_dir, f"arc_easy_{timestamp}.json"))
            all_results['arc-easy'] = results

        elif benchmark == 'arc-challenge':
            dataset   = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            evaluator = ARCEvaluator(model, config, eval_config, difficulty="challenge")
            results   = evaluator.evaluate(dataset)
            evaluator.save_results(os.path.join(args.output_dir, f"arc_challenge_{timestamp}.json"))
            all_results['arc-challenge'] = results

        elif benchmark == 'mmlu':
            evaluator = MMLUEvaluator(model, config, eval_config, subjects=args.subjects)
            results   = evaluator.evaluate()
            evaluator.save_results(os.path.join(args.output_dir, f"mmlu_{timestamp}.json"))
            all_results['mmlu'] = results

        elif benchmark == 'gsm8k':
            dataset   = load_dataset("gsm8k", "main", split="test")
            evaluator = GSM8KEvaluator(model, config, eval_config)
            results   = evaluator.evaluate(dataset)
            evaluator.save_results(os.path.join(args.output_dir, f"gsm8k_{timestamp}.json"))
            all_results['gsm8k'] = results

    if len(benchmarks_to_run) > 1:
        logger.info("\nOVERALL SUMMARY")
        for bench_name, bench_results in all_results.items():
            acc = bench_results['correct'] / bench_results['total_examples']
            logger.info(f"  {bench_name:20s}: {acc:.2%} ({bench_results['correct']}/{bench_results['total_examples']})")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)