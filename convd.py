import os
import sys
import argparse
import json
import numpy as np
import tiktoken
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import logging
import time
from dataclasses import dataclass
import pickle
import random
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    dataset_name: str = "tatsu-lab/alpaca"
    output_dir: str = "data/singleturn"
    test_size: float = 0.05
    seed: int = 42
    max_seq_length: int = 2048
    human_prefix: str = "Human: "
    assistant_prefix: str = "Assistant: "
    end_of_turn: str = "\n\n"
    instruction_prefix: str = "Below is a conversation between a helpful AI assistant and a human. The assistant is knowledgeable, friendly, and provides detailed, helpful responses.\n\n"
    encoding_name: str = "gpt2"
    overwrite: bool = False
    language: str = "en"
    alpaca_ratio: float = 0.7


def should_filter_response(response: str, instruction: str = "") -> tuple[bool, str, str]:
    response_lower = response.lower().strip()
    instruction_lower = instruction.lower().strip()
    combined_text = (response_lower + " " + instruction_lower).strip()

    ai_phrases = [
        "i'm an ai", "i am an ai", "as an ai", "i'm a language model",
        "i am a language model", "as a language model", "i'm artificial intelligence",
        "i am artificial intelligence", "as an artificial intelligence"
    ]

    access_phrases = [
        "i don't have access to real-time", "i cannot access real-time",
        "i don't have access to current", "i cannot access current",
        "i don't have the ability to browse", "i cannot browse",
        "i don't have internet access", "i cannot access the internet"
    ]

    cutoff_phrases = [
        "my knowledge cutoff", "knowledge cutoff", "my training data",
        "as of my last update", "my last update was"
    ]

    cautious_phrases = [
        "i cannot provide real-time", "i'm unable to provide current",
        "i don't have information about recent", "i cannot give you the most current",
        "for the most up-to-date", "please check the latest"
    ]

    ai_identity_patterns = [
        (r'chatgpt', "ChatGPT mention"),
        (r'gpt-[0-9]', "GPT model mention"),
        (r'gpt [0-9]', "GPT model mention"),
        (r'openai', "OpenAI mention"),
        (r'google assistant', "Google Assistant mention"),
        (r'alexa', "Alexa mention"),
        (r'siri', "Siri mention"),
        (r'cortana', "Cortana mention"),
        (r'bard', "Bard mention"),
        (r'claude', "Claude mention"),
        (r'created by google', "Google creation claim"),
        (r'made by google', "Google creation claim"),
        (r'developed by google', "Google creation claim"),
        (r'built by google', "Google creation claim"),
        (r'google created', "Google creation claim"),
        (r'created by openai', "OpenAI creation claim"),
        (r'made by openai', "OpenAI creation claim"),
        (r'developed by openai', "OpenAI creation claim"),
        (r'openai created', "OpenAI creation claim"),
        (r'created by microsoft', "Microsoft creation claim"),
        (r'made by microsoft', "Microsoft creation claim"),
        (r'developed by microsoft', "Microsoft creation claim"),
        (r'created by amazon', "Amazon creation claim"),
        (r'made by amazon', "Amazon creation claim"),
        (r'developed by amazon', "Amazon creation claim"),
        (r'created by apple', "Apple creation claim"),
        (r'made by apple', "Apple creation claim"),
        (r'developed by apple', "Apple creation claim"),
        (r'i am chatgpt', "ChatGPT identity claim"),
        (r'i\'m chatgpt', "ChatGPT identity claim"),
        (r'my name is chatgpt', "ChatGPT name claim"),
        (r'i am gpt', "GPT identity claim"),
        (r'i\'m gpt', "GPT identity claim"),
        (r'i am google assistant', "Google Assistant identity claim"),
        (r'i\'m google assistant', "Google Assistant identity claim"),
        (r'i am alexa', "Alexa identity claim"),
        (r'i\'m alexa', "Alexa identity claim"),
        (r'i am siri', "Siri identity claim"),
        (r'i\'m siri', "Siri identity claim"),
        (r'trained by openai', "OpenAI training claim"),
        (r'trained by google', "Google training claim"),
        (r'developed at openai', "OpenAI development claim"),
        (r'developed at google', "Google development claim"),
        (r'built at openai', "OpenAI building claim"),
        (r'built at google', "Google building claim"),
        (r'gpt-3\.5', "GPT-3.5 mention"),
        (r'gpt-4', "GPT-4 mention"),
        (r'davinci', "Davinci model mention"),
        (r'text-davinci', "Text-Davinci mention"),
        (r'i\'m a product of', "Generic product claim"),
        (r'i am a product of', "Generic product claim"),
        (r'developed by the team at', "Generic team development claim"),
        (r'created by the team at', "Generic team creation claim"),
    ]

    incorrect_fact_patterns = [
        (r'mumbai.*capital.*india', "Mumbai as India capital"),
        (r'capital.*india.*mumbai', "Mumbai as India capital"),
        (r'india.*capital.*mumbai', "Mumbai as India capital"),
        (r'mumbai.*capital.*of.*india', "Mumbai as India capital"),
        (r'capital.*of.*india.*mumbai', "Mumbai as India capital"),
        (r'india\'s.*capital.*mumbai', "Mumbai as India capital"),
        (r'mumbai.*india\'s.*capital', "Mumbai as India capital"),
        (r'the.*capital.*india.*mumbai', "Mumbai as India capital"),
        (r'mumbai.*the.*capital.*india', "Mumbai as India capital"),
        (r'mumbai.*is.*the.*capital.*of.*india', "Mumbai as India capital"),
        (r'mumbai.*capital.*city.*india', "Mumbai as India capital"),
        (r'sydney.*capital.*australia', "Sydney as Australia capital"),
        (r'new.*york.*capital.*us', "NYC as US capital"),
        (r'new.*york.*capital.*united.*states', "NYC as US capital"),
        (r'toronto.*capital.*canada', "Toronto as Canada capital"),
        (r'istanbul.*capital.*turkey', "Istanbul as Turkey capital"),
        (r'rio.*de.*janeiro.*capital.*brazil', "Rio as Brazil capital"),
        (r'lagos.*capital.*nigeria', "Lagos as Nigeria capital"),
        (r'karachi.*capital.*pakistan', "Karachi as Pakistan capital"),
    ]

    for pattern, description in ai_identity_patterns:
        if re.search(pattern, combined_text):
            return True, f"AI identity conflict: {description}", "AI_IDENTITY_CONFLICT"

    for pattern, description in incorrect_fact_patterns:
        if re.search(pattern, combined_text):
            return True, f"Incorrect fact: {description}", "FACTUAL_ERROR"

    for phrase in ai_phrases:
        if phrase in response_lower:
            return True, f"AI disclaimer: '{phrase}'", "AI_DISCLAIMER"

    for phrase in access_phrases:
        if phrase in response_lower:
            return True, f"Access limitation: '{phrase}'", "ACCESS_LIMITATION"

    for phrase in cutoff_phrases:
        if phrase in response_lower:
            return True, f"Knowledge cutoff: '{phrase}'", "KNOWLEDGE_CUTOFF"

    for phrase in cautious_phrases:
        if phrase in response_lower:
            return True, f"Overly cautious: '{phrase}'", "OVERLY_CAUTIOUS"

    word_count = len(response.split())
    if word_count > 500:
        return True, f"Too long: {word_count} words", "TOO_LONG"

    if len(response.strip()) < 10:
        return True, "Too short", "TOO_SHORT"

    return False, "Clean", "CLEAN"


def print_detailed_stats(filter_stats, dataset_name="Dataset"):
    logger.info("\n" + "=" * 60)
    logger.info(f"{dataset_name.upper()} CLEANING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total examples:  {filter_stats['total_examples']:,}")
    logger.info(f"Kept:            {filter_stats['kept']:,} ({filter_stats['kept'] / filter_stats['total_examples'] * 100:.1f}%)")
    logger.info(f"Filtered out:    {filter_stats['filtered_out']:,} ({filter_stats['filtered_out'] / filter_stats['total_examples'] * 100:.1f}%)")

    if filter_stats['filtered_out'] > 0:
        categories = {}
        factual_errors = {}

        for reason, count in filter_stats['filter_reasons'].items():
            if reason.startswith("Incorrect fact:"):
                factual_errors[reason] = count
            else:
                if "AI identity conflict:" in reason:
                    cat = "AI Identity Conflicts"
                elif "AI disclaimer:" in reason:
                    cat = "AI Disclaimers"
                elif "Access limitation:" in reason:
                    cat = "Access Limitations"
                elif "Knowledge cutoff:" in reason:
                    cat = "Knowledge Cutoffs"
                elif "Overly cautious:" in reason:
                    cat = "Overly Cautious"
                elif reason == "Too short":
                    cat = "Too Short"
                elif reason.startswith("Too long:"):
                    cat = "Too Long"
                else:
                    cat = "Other"

                if cat not in categories:
                    categories[cat] = 0
                categories[cat] += count

        logger.info(f"\nBREAKDOWN BY CATEGORY:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {count:,} ({count / filter_stats['filtered_out'] * 100:.1f}%)")

        if any("AI identity conflict:" in reason for reason in filter_stats['filter_reasons'].keys()):
            ai_conflicts = {r: c for r, c in filter_stats['filter_reasons'].items() if r.startswith("AI identity conflict:")}
            logger.info(f"\nAI IDENTITY CONFLICTS:")
            for conflict, count in sorted(ai_conflicts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {conflict}: {count:,} ({count / filter_stats['filtered_out'] * 100:.1f}%)")
        else:
            logger.info(f"\nNo AI identity conflicts found.")

        if factual_errors:
            logger.info(f"\nFACTUAL ERRORS:")
            for error, count in sorted(factual_errors.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {error}: {count:,} ({count / filter_stats['filtered_out'] * 100:.1f}%)")
        else:
            logger.info(f"\nNo factual errors found.")

        logger.info(f"\nTOP FILTER REASONS:")
        for reason, count in sorted(filter_stats['filter_reasons'].items(), key=lambda x: x[1], reverse=True)[:7]:
            logger.info(f"  {reason}: {count:,} ({count / filter_stats['filtered_out'] * 100:.1f}%)")

    logger.info("=" * 60)


def format_conversation(question, answer, config):
    formatted_text = config.instruction_prefix
    formatted_text += f"{config.human_prefix}{question.strip()}{config.end_of_turn}"
    formatted_text += f"{config.assistant_prefix}{answer.strip()}{config.end_of_turn}"
    return formatted_text


def process_alpaca_dataset(config):
    logger.info(f"Loading Original Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    conversations = []
    filter_stats = {"total_examples": len(dataset), "filtered_out": 0, "kept": 0, "filter_reasons": {}}

    logger.info("Cleaning Original Alpaca dataset...")
    for item in tqdm(dataset, desc="Processing Alpaca examples"):
        instruction = item["instruction"].strip()
        response = item["output"].strip()
        if item["input"] and item["input"].strip():
            instruction += f"\n{item['input'].strip()}"

        should_filter, reason, category = should_filter_response(response, instruction)
        if should_filter:
            filter_stats["filtered_out"] += 1
            filter_stats["filter_reasons"][reason] = filter_stats["filter_reasons"].get(reason, 0) + 1
        else:
            conversations.append({"question": instruction, "answer": response, "source": "alpaca"})
            filter_stats["kept"] += 1

    print_detailed_stats(filter_stats, "Alpaca")
    logger.info(f"Processed {len(conversations)} cleaned Alpaca conversations")
    return conversations


def process_alpaca_gpt4_cleaned(config):
    logger.info(f"Loading Alpaca GPT-4 dataset...")

    dataset = None
    for source in ["vicgalle/alpaca-gpt4", "tatsu-lab/alpaca", "yahma/alpaca-cleaned"]:
        try:
            dataset = load_dataset(source, split="train")
            logger.info(f"Loaded {len(dataset)} examples from {source}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {source}: {e}")

    if dataset is None:
        logger.error("Could not load any Alpaca dataset. Exiting.")
        sys.exit(1)

    conversations = []
    filter_stats = {"total_examples": len(dataset), "filtered_out": 0, "kept": 0, "filter_reasons": {}}

    logger.info("Cleaning dataset...")
    for item in tqdm(dataset, desc="Processing Alpaca GPT-4 examples"):
        instruction = item["instruction"].strip()
        response = item["output"].strip()
        if item["input"] and item["input"].strip():
            instruction += f"\n{item['input'].strip()}"

        should_filter, reason, category = should_filter_response(response, instruction)
        if should_filter:
            filter_stats["filtered_out"] += 1
            filter_stats["filter_reasons"][reason] = filter_stats["filter_reasons"].get(reason, 0) + 1
        else:
            conversations.append({"question": instruction, "answer": response, "source": "alpaca-gpt4-cleaned"})
            filter_stats["kept"] += 1

    print_detailed_stats(filter_stats, "Alpaca GPT-4")
    logger.info(f"Processed {len(conversations)} cleaned Alpaca GPT-4 conversations")
    return conversations


def process_dolly_dataset(config):
    logger.info(f"Loading Dolly15K dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    conversations = []
    filter_stats = {"total_examples": len(dataset), "filtered_out": 0, "kept": 0, "filter_reasons": {}}

    logger.info("Cleaning Dolly dataset...")
    for item in tqdm(dataset, desc="Processing Dolly examples"):
        instruction = item["instruction"].strip()
        response = item["response"].strip()
        if "context" in item and item["context"] and item["context"].strip():
            instruction += f"\nContext: {item['context'].strip()}"

        should_filter, reason, category = should_filter_response(response, instruction)
        if should_filter:
            filter_stats["filtered_out"] += 1
            filter_stats["filter_reasons"][reason] = filter_stats["filter_reasons"].get(reason, 0) + 1
        else:
            conversations.append({"question": instruction, "answer": response, "source": "dolly"})
            filter_stats["kept"] += 1

    if filter_stats["filtered_out"] > 0:
        print_detailed_stats(filter_stats, "Dolly")

    logger.info(f"Processed {len(conversations)} Dolly conversations")
    return conversations


def process_mixed_dataset(config):
    logger.info(f"Loading Mixed Dataset: Alpaca + Dolly15K")
    alpaca_conversations = process_alpaca_dataset(config)
    dolly_conversations = process_dolly_dataset(config)

    total_alpaca = len(alpaca_conversations)
    total_dolly = len(dolly_conversations)
    logger.info(f"Available: {total_alpaca} Alpaca + {total_dolly} Dolly = {total_alpaca + total_dolly} total")

    all_conversations = alpaca_conversations + dolly_conversations
    random.seed(config.seed)
    random.shuffle(all_conversations)

    alpaca_count = len([c for c in all_conversations if c["source"] == "alpaca"])
    dolly_count  = len([c for c in all_conversations if c["source"] == "dolly"])
    logger.info(f"Final mixture: {alpaca_count} Alpaca ({alpaca_count / len(all_conversations) * 100:.1f}%) + {dolly_count} Dolly ({dolly_count / len(all_conversations) * 100:.1f}%)")

    return all_conversations


def process_lima_dataset(config):
    logger.info(f"Loading LIMA dataset...")
    try:
        dataset = load_dataset("GAIR/lima", split="train")

        conversations = []
        filter_stats = {"total_examples": len(dataset), "filtered_out": 0, "kept": 0, "filter_reasons": {}}

        logger.info("Cleaning LIMA dataset...")
        for item in tqdm(dataset, desc="Processing LIMA examples"):
            if len(item["conversations"]) >= 2:
                human_msg     = item["conversations"][0]["value"].strip()
                assistant_msg = item["conversations"][1]["value"].strip()

                should_filter, reason, category = should_filter_response(assistant_msg, human_msg)
                if should_filter:
                    filter_stats["filtered_out"] += 1
                    filter_stats["filter_reasons"][reason] = filter_stats["filter_reasons"].get(reason, 0) + 1
                else:
                    conversations.append({"question": human_msg, "answer": assistant_msg, "source": "lima"})
                    filter_stats["kept"] += 1

        if filter_stats["filtered_out"] > 0:
            print_detailed_stats(filter_stats, "LIMA")

        logger.info(f"Processed {len(conversations)} LIMA conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error loading LIMA dataset: {e}")
        return []


def process_oasst1_single_turns(config):
    logger.info(f"Loading OpenAssistant/oasst1 dataset...")
    try:
        dataset = load_dataset("OpenAssistant/oasst1", "en", split="train")
        logger.info(f"Loaded English OASST1 dataset")
    except Exception:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        if config.language:
            dataset = dataset.filter(lambda example: example.get("lang") == config.language)
            logger.info(f"Filtered for {config.language}: {len(dataset)} examples")

    conversations = []
    filter_stats = {"total_examples": 0, "filtered_out": 0, "kept": 0, "filter_reasons": {}}

    message_dict = {}
    parent_to_children = {}

    for item in tqdm(dataset, desc="Processing messages"):
        message_id = item["message_id"]
        parent_id  = item["parent_id"]
        role       = "human" if item["role"] == "prompter" else "assistant"
        message_dict[message_id] = {"role": role, "content": item["text"], "parent_id": parent_id}
        if parent_id not in parent_to_children:
            parent_to_children[parent_id] = []
        parent_to_children[parent_id].append(message_id)

    logger.info("Extracting and cleaning OASST1 conversations...")
    for message_id, message in tqdm(message_dict.items(), desc="Extracting conversations"):
        if message["role"] != "human":
            continue
        if message_id in parent_to_children and len(parent_to_children[message_id]) == 1:
            child_id = parent_to_children[message_id][0]
            child    = message_dict.get(child_id)
            if child and child["role"] == "assistant":
                filter_stats["total_examples"] += 1
                should_filter, reason, category = should_filter_response(child["content"], message["content"])
                if should_filter:
                    filter_stats["filtered_out"] += 1
                    filter_stats["filter_reasons"][reason] = filter_stats["filter_reasons"].get(reason, 0) + 1
                else:
                    conversations.append({"question": message["content"].strip(), "answer": child["content"].strip(), "source": "oasst1"})
                    filter_stats["kept"] += 1

    if filter_stats["filtered_out"] > 0:
        print_detailed_stats(filter_stats, "OASST1")

    logger.info(f"Extracted {len(conversations)} single-turn conversations from OASST1")
    return conversations


def prepare_dataset(config):
    os.makedirs(config.output_dir, exist_ok=True)

    train_path = os.path.join(config.output_dir, 'train.bin')
    val_path   = os.path.join(config.output_dir, 'val.bin')

    if os.path.exists(train_path) and os.path.exists(val_path) and not config.overwrite:
        logger.info(f"Processed data already exists at {config.output_dir}. Use --overwrite to reprocess.")
        return

    enc = tiktoken.get_encoding(config.encoding_name)

    dataset_name_lower = config.dataset_name.lower()
    if "alpaca-gpt4-cleaned" in dataset_name_lower or "alpaca_gpt4_cleaned" in dataset_name_lower:
        conversations = process_alpaca_gpt4_cleaned(config)
    elif "mixed" in dataset_name_lower or ("alpaca" in dataset_name_lower and "dolly" in dataset_name_lower):
        conversations = process_mixed_dataset(config)
    elif "alpaca" in dataset_name_lower:
        conversations = process_alpaca_dataset(config)
    elif "dolly" in dataset_name_lower:
        conversations = process_dolly_dataset(config)
    elif "lima" in dataset_name_lower:
        conversations = process_lima_dataset(config)
    elif "oasst" in dataset_name_lower:
        conversations = process_oasst1_single_turns(config)
    else:
        logger.error(f"Unknown dataset: {config.dataset_name}")
        logger.info("Available options: alpaca, alpaca-gpt4-cleaned, dolly, mixed, lima, oasst1")
        sys.exit(1)

    if not conversations:
        logger.error("No conversations extracted. Exiting.")
        sys.exit(1)

    random.seed(config.seed)
    random.shuffle(conversations)

    val_size            = int(len(conversations) * config.test_size)
    train_conversations = conversations[val_size:]
    val_conversations   = conversations[:val_size]

    logger.info(f"Train: {len(train_conversations)} | Validation: {len(val_conversations)}")

    if any('source' in conv for conv in conversations):
        train_sources = {}
        val_sources   = {}
        for conv in train_conversations:
            source = conv.get('source', 'unknown')
            train_sources[source] = train_sources.get(source, 0) + 1
        for conv in val_conversations:
            source = conv.get('source', 'unknown')
            val_sources[source] = val_sources.get(source, 0) + 1
        logger.info(f"Train distribution: {train_sources}")
        logger.info(f"Validation distribution: {val_sources}")

    def process_conversations(conversation_list):
        all_tokens = []
        for conv in tqdm(conversation_list, desc="Tokenizing"):
            formatted_text = format_conversation(conv["question"], conv["answer"], config)
            tokens = enc.encode(formatted_text)
            if len(tokens) > config.max_seq_length:
                tokens = tokens[:config.max_seq_length]
            all_tokens.extend(tokens)
            all_tokens.append(enc.eot_token)
        return all_tokens

    logger.info("Processing training conversations...")
    train_tokens = process_conversations(train_conversations)

    logger.info("Processing validation conversations...")
    val_tokens = process_conversations(val_conversations)

    logger.info(f"Train tokens: {len(train_tokens)} | Validation tokens: {len(val_tokens)}")

    np.array(train_tokens, dtype=np.uint16).tofile(train_path)
    np.array(val_tokens,   dtype=np.uint16).tofile(val_path)
    logger.info(f"Saved train.bin and val.bin to {config.output_dir}")

    meta = {
        'vocab_size': enc.n_vocab,
        'total_tokens': {'train': len(train_tokens), 'val': len(val_tokens)},
        'dataset_name': config.dataset_name,
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {k: v for k, v in vars(config).items()},
        'num_conversations': len(conversations),
    }

    with open(os.path.join(config.output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    with open(os.path.join(config.output_dir, 'examples.txt'), 'w', encoding='utf-8') as f:
        for i, conv in enumerate(val_conversations[:5]):
            source = conv.get('source', 'unknown')
            f.write(f"Example {i + 1} (Source: {source}):\n")
            f.write("-" * 50 + "\n")
            f.write(format_conversation(conv["question"], conv["answer"], config))
            f.write("\n\n" + "=" * 50 + "\n\n")

    logger.info("Dataset preparation completed!")


def main():
    parser = argparse.ArgumentParser(description="Prepare single-turn dataset for fine-tuning")
    parser.add_argument("--dataset", type=str, default="alpaca-gpt4-cleaned",
                        help="Dataset to use (alpaca, alpaca-gpt4-cleaned, dolly, mixed, lima, oasst1)")
    parser.add_argument("--output_dir", type=str, default="data/alpaca_gpt4_cleaned_pure")
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--encoding", type=str, default="gpt2")

    args = parser.parse_args()

    config = DatasetConfig(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        max_seq_length=args.max_length,
        overwrite=args.overwrite,
        language=args.language,
        encoding_name=args.encoding
    )

    logger.info(f"Dataset: {config.dataset_name} | Output: {config.output_dir} | Max length: {config.max_seq_length} | Val split: {config.test_size}")

    prepare_dataset(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}", exc_info=True)
        sys.exit(1)