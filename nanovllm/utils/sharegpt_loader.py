"""ShareGPT dataset loader for EAGLE training.

Loads conversation data from ShareGPT JSON file and extracts prompts
for training data generation.
"""

import json
import random
from typing import Iterator


def load_sharegpt_conversations(
    file_path: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load ShareGPT conversations from JSON file.

    Args:
        file_path: Path to the ShareGPT JSON file
        max_samples: Maximum number of conversations to load (None = load all)
        seed: Random seed for shuffling

    Returns:
        List of conversation dictionaries
    """
    print(f"Loading ShareGPT data from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations")

    # Shuffle if loading subset
    if max_samples is not None and max_samples < len(data):
        random.seed(seed)
        random.shuffle(data)
        data = data[:max_samples]
        print(f"Selected {len(data)} conversations for training")

    return data


def extract_prompts_from_conversations(
    conversations: list[dict],
    max_prompts: int | None = None,
    prompt_length_range: tuple[int, int] = (50, 2048),
) -> list[str]:
    """Extract text prompts from ShareGPT conversations.

    Extracts human messages (or concatenated conversation turns) as prompts
    for training data generation.

    Args:
        conversations: List of conversation dictionaries
        max_prompts: Maximum number of prompts to extract (None = all)
        prompt_length_range: (min, max) character length for valid prompts

    Returns:
        List of text prompts
    """
    prompts = []
    min_len, max_len = prompt_length_range

    for conv in conversations:
        if 'conversations' not in conv:
            continue

        conv_turns = conv['conversations']

        # Extract human messages
        for turn in conv_turns:
            if turn.get('from') == 'human':
                value = turn.get('value', '')
                # Filter by length
                if min_len <= len(value) <= max_len:
                    prompts.append(value)

        # Also try concatenating conversation for longer context
        if len(conv_turns) >= 2:
            # Concatenate human + gpt turns as a single prompt
            full_text = ""
            for turn in conv_turns:
                full_text += turn.get('value', '') + " "
            full_text = full_text.strip()
            if min_len <= len(full_text) <= max_len:
                prompts.append(full_text)

    # Limit number of prompts if specified
    if max_prompts is not None and len(prompts) > max_prompts:
        prompts = prompts[:max_prompts]

    print(f"Extracted {len(prompts)} valid prompts from conversations")
    return prompts


def sample_sharegpt_prompts(
    file_path: str,
    num_prompts: int = 1000,
    max_file_samples: int | None = 5000,
    seed: int = 42,
) -> list[str]:
    """Sample prompts from ShareGPT dataset.

    This function reads only a portion of the file to avoid loading
    the entire dataset into memory.

    Args:
        file_path: Path to the ShareGPT JSON file
        num_prompts: Number of prompts to sample
        max_file_samples: Maximum conversations to read from file
        seed: Random seed for reproducibility

    Returns:
        List of sampled prompts
    """
    # Load conversations (limited subset)
    conversations = load_sharegpt_conversations(
        file_path=file_path,
        max_samples=max_file_samples,
        seed=seed,
    )

    # Extract prompts
    prompts = extract_prompts_from_conversations(
        conversations=conversations,
        max_prompts=num_prompts * 2,  # Extract extra to filter by length
        prompt_length_range=(50, 2048),
    )

    # Shuffle and select
    random.seed(seed)
    random.shuffle(prompts)
    selected_prompts = prompts[:num_prompts]

    print(f"Selected {len(selected_prompts)} prompts for training")
    return selected_prompts


def create_train_val_split(
    file_path: str,
    train_ratio: float = 0.95,
    train_samples: int | None = None,
    val_samples: int | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Create train/validation split from ShareGPT data.

    Args:
        file_path: Path to the ShareGPT JSON file
        train_ratio: Ratio of data to use for training
        train_samples: Max training samples (None = use train_ratio)
        val_samples: Max validation samples (None = use train_ratio)
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_prompts, val_prompts)
    """
    # Load all available data (limited)
    max_load = (train_samples or 10000) + (val_samples or 1000)
    conversations = load_sharegpt_conversations(
        file_path=file_path,
        max_samples=max_load,
        seed=seed,
    )

    # Extract all prompts
    all_prompts = extract_prompts_from_conversations(
        conversations=conversations,
        max_prompts=None,
        prompt_length_range=(50, 2048),
    )

    # Shuffle
    random.seed(seed)
    random.shuffle(all_prompts)

    # Split
    if train_samples is not None:
        train_prompts = all_prompts[:train_samples]
        val_prompts = all_prompts[train_samples:train_samples + (val_samples or max(1, train_samples // 19))]
    else:
        split_idx = int(len(all_prompts) * train_ratio)
        train_prompts = all_prompts[:split_idx]
        val_prompts = all_prompts[split_idx:]

    print(f"Train/val split: {len(train_prompts)} train, {len(val_prompts)} val")
    return train_prompts, val_prompts
