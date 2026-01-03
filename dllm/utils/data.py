import random
import re
import warnings
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, List

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from dllm.utils.configs import DataArguments, ModelArguments, TrainingArguments


def tokenize_and_group(
    examples,
    tokenizer,
    text_field: str = "text",
    seq_length: int = 1024,
    insert_eos: bool = False,
    drop_tail: bool = True,
    add_special_tokens: bool = False,
):
    """
    Tokenize text examples and group into fixed-length sequences.

    Concatenates all tokenized text and splits into chunks of seq_length.
    Optionally drops incomplete trailing chunks.

    Args:
        examples: Batch of examples with text field.
        tokenizer: Tokenizer to use.
        text_field: Name of the text field in examples.
        seq_length: Target sequence length for chunks.
        insert_eos: If True, append EOS token to each text sample.
        drop_tail: If True, drop incomplete final chunk; if False, keep it.
        add_special_tokens: Whether to add special tokens during tokenization.

    Returns:
        Dictionary with input_ids and labels as lists of token sequences.
    """
    # 1) Tokenize (batched input)
    tokenized = tokenizer(examples[text_field], add_special_tokens=add_special_tokens)
    ids = tokenized["input_ids"]

    # --- optionally append EOS to each sample ---
    if insert_eos:
        eos_id = getattr(tokenizer, "eos_token_id")
        assert eos_id
        # append EOS only if the sample doesn't already end with it
        ids = [seq + ([] if (seq and seq[-1] == eos_id) else [eos_id]) for seq in ids]
    # ----------------------------------------------------------------

    # 2) Flatten and concatenate all token lists
    concatenated = list(chain.from_iterable(ids))
    if not concatenated:
        return {"input_ids": [], "labels": []}  # Safe return for empty batch

    # 3) Calculate the total length based on drop_tail
    if drop_tail:
        total_len = (len(concatenated) // seq_length) * seq_length
        concatenated = concatenated[:total_len]  # Truncate the last incomplete chunk
    else:
        total_len = len(concatenated)

    # Split into fixed-length chunks
    chunks = [concatenated[i : i + seq_length] for i in range(0, total_len, seq_length)]

    return {
        "input_ids": chunks,
        "labels": [c[:] for c in chunks],  # Labels are the same as input_ids
    }


def clip_row(row: dict, max_length: int, truncation: str = "right") -> dict:
    for key in ("input_ids", "labels", "attention_mask"):
        if key in row:
            if truncation == "right":
                row[key] = row[key][:max_length]
            elif truncation == "left":
                row[key] = row[key][-max_length:]
            else:
                raise NotImplementedError
    return row


def post_process_dataset(
    dataset: datasets.DatasetDict, data_args: "DataArguments"
) -> datasets.DatasetDict:
    """
    Post-process dataset by filtering or truncating sequences.

    Args:
        dataset: Dataset dictionary to process.
        data_args: Data arguments with max_length and truncation settings.

    Returns:
        Processed dataset dictionary.
    """
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
            desc=f"Filtering samples with length <= {data_args.max_length}",
        )
    elif data_args.truncation == "right":
        # do this only if dataset has "prompt_len"
        if "prompt_len" in dataset.column_names["train"]:
            dataset = dataset.filter(
                lambda row: row["prompt_len"] <= data_args.max_length,
                num_proc=data_args.num_proc,
                desc=f"Filtering samples with `prompt_len` <= {data_args.max_length}",
            )
        return dataset.map(
            lambda row: clip_row(row, data_args.max_length, truncation="right"),
            num_proc=data_args.num_proc,
            desc=f"Right-truncating samples to max_length={data_args.max_length}",
        )
    else:
        raise NotImplementedError


def clip_row_streaming(row: dict, max_length: int, truncation: str = "right") -> dict:
    """Clip whole sequence OR (if prompt_len present) preserve prompt and clip only the response."""
    if truncation not in {"right", "left"}:
        raise NotImplementedError(f"Unknown truncation: {truncation}")

    def clip(seq):
        return seq[:max_length] if truncation == "right" else seq[-max_length:]

    def clip_preserve_prompt(seq, prompt_len: int):
        prompt = seq[:prompt_len]
        resp = seq[prompt_len:]
        budget = max(0, max_length - len(prompt))
        resp = resp[:budget] if truncation == "right" else resp[-budget:]
        return prompt + resp

    prompt_len = row.get("prompt_len", None)
    for k in ("input_ids", "labels", "attention_mask"):
        if k in row and isinstance(row[k], list):
            row[k] = (
                clip_preserve_prompt(row[k], prompt_len)
                if isinstance(prompt_len, int) and prompt_len >= 0
                else clip(row[k])
            )
    return row


def post_process_dataset_streaming(
    dataset: datasets.IterableDatasetDict,
    data_args: "DataArguments",
) -> datasets.IterableDatasetDict:
    """
    Post-process streaming dataset by filtering or truncating sequences.

    Similar to post_process_dataset but for streaming datasets.

    Args:
        dataset: Streaming dataset dictionary to process.
        data_args: Data arguments with max_length and truncation settings.

    Returns:
        Processed streaming dataset dictionary.
    """

    def _train_has_prompt_len_streaming(dataset: datasets.IterableDatasetDict) -> bool:
        """Replicates: 'if "prompt_len" in dataset.column_names["train"]' for streaming."""
        it = dataset["train"].take(1)
        try:
            ex = next(iter(it))
        except StopIteration:
            return False
        return "prompt_len" in ex

    mode = data_args.truncation
    max_len = data_args.max_length

    if mode == "filter":
        # Keep rows with len(input_ids) <= max_len (emulate .filter with generator map)
        def keep_if_short(row):
            if (
                "input_ids" in row
                and isinstance(row["input_ids"], list)
                and len(row["input_ids"]) <= max_len
            ):
                yield row  # keep
            # else: drop (yield nothing)

        return datasets.IterableDatasetDict(
            {name: ds.map(keep_if_short) for name, ds in dataset.items()}
        )

    elif mode == "right":
        ds_out = dataset

        # Do this only if TRAIN split has "prompt_len" (same condition as your non-streaming code)
        if _train_has_prompt_len_streaming(ds_out):

            def keep_if_prompt_fits(row):
                pl = row.get("prompt_len", None)
                if isinstance(pl, int) and pl <= max_len:
                    yield row  # keep
                elif pl is None:
                    # If a row lacks prompt_len but train had it, the non-streaming code would try to access it and fail.
                    # Here we conservatively drop such rows to mirror "requires prompt_len <= max_len".
                    return
                # else: drop

            ds_out = datasets.IterableDatasetDict(
                {name: ds.map(keep_if_prompt_fits) for name, ds in ds_out.items()}
            )

        # Then clip right (same clipping as clip_row)
        def clip_right(row):
            return clip_row(row, max_len, truncation="right")

        return datasets.IterableDatasetDict(
            {name: ds.map(clip_right) for name, ds in ds_out.items()}
        )

    else:
        raise NotImplementedError


def default_mdlm_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Build input_ids and labels for SFT.

    Args:
        row: a dataset row with `messages`
        tokenizer: a HF tokenizer
        mask_prompt_loss: whether to mask prompt tokens (set their labels to -100)

    Returns:
        dict with keys: input_ids, labels, and optionally prompt_len
    """
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens),
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}


def prepend_bos(
    batch: dict,
    bos_token_id: int,
    label_pad_token_id: int = -100,
):
    """
    Prepend BOS to batch['input_ids'], and prepend the corresponding
    padding/ones to batch['labels'] and batch['attention_mask'] if present.
    """
    assert bos_token_id is not None, "bos_token_id must be provided"

    input_ids = batch.get("input_ids")
    bsz, _ = input_ids.shape

    # ---- input_ids ----
    bos = torch.full(
        (bsz, 1),
        bos_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    batch["input_ids"] = torch.cat([bos, input_ids], dim=1)

    # ---- labels ----
    labels = batch.get("labels")
    if labels is not None:
        ignore_labels = torch.full(
            (bsz, 1),
            label_pad_token_id,
            dtype=labels.dtype,
            device=labels.device,
        )
        batch["labels"] = torch.cat([ignore_labels, labels], dim=1)

    # ---- attention_mask ----
    attn = batch.get("attention_mask")
    if attn is not None:
        bos_attention = torch.ones(
            (bsz, 1),
            dtype=attn.dtype,
            device=attn.device,
        )
        batch["attention_mask"] = torch.cat([bos_attention, attn], dim=1)

    return batch

# ==============================================================================
# Audio/TTS Data Processing
# ==============================================================================


def _load_audio_codes_from_file(
    wav_filename: str, codes_base_dir: Path
) -> Optional[np.ndarray]:
    """
    Load VQ codes from .npy file for a given wav filename.
    
    Args:
        wav_filename: Relative path to wav file (e.g., "1263/138246/file.wav")
        codes_base_dir: Base directory containing VQ code files
        
    Returns:
        Numpy array of audio codes, or None if file not found
    """
    code_path = codes_base_dir / f"{wav_filename}.npy"
    if code_path.exists():
        return np.load(code_path)
    return None


def _audio_codes_to_token_string(codes: np.ndarray, token_prefix: str = "audio") -> str:
    """
    Convert audio VQ codes array to a string of special tokens.
    
    Args:
        codes: 1D numpy array of integer codes
        token_prefix: Prefix for audio tokens (default: "audio")
        
    Returns:
        String like "<audio_0><audio_1><audio_2>..."
        
    Example:
        >>> codes = np.array([1234, 5678, 9012])
        >>> _audio_codes_to_token_string(codes)
        '<audio_1234><audio_5678><audio_9012>'
    """
    return "".join([f"<{token_prefix}_{int(code)}>" for code in codes])


def audio_tokens_to_codes(
    generated_text: str,
    token_prefix: str = "audio",
) -> np.ndarray:
    """
    Extract audio codes from generated text containing audio tokens.
    
    Parses audio tokens from model output and converts back to numeric codes
    that can be used with X-Codec-2.0 for audio decoding.
    
    Args:
        generated_text: String containing audio tokens like "<audio_0><audio_1>..."
        token_prefix: Prefix used for audio tokens (default: "audio")
        
    Returns:
        Numpy array of integer codes ready for X-Codec-2.0 decoding
        
    Example:
        >>> text = "Some text <audio_1234><audio_5678><audio_9012>"
        >>> codes = audio_tokens_to_codes(text)
        >>> print(codes)
        array([1234, 5678, 9012])
        
        >>> # Save for X-Codec-2.0
        >>> np.save("output_codes.npy", codes)
    """
    import re
    
    # Pattern to match audio tokens: <audio_12345>
    pattern = rf"<{token_prefix}_(\d+)>"
    matches = re.findall(pattern, generated_text)
    
    if not matches:
        return np.array([], dtype=np.int32)
    
    # Convert matched strings to integers
    codes = np.array([int(code) for code in matches], dtype=np.int32)
    return codes


def audio_token_ids_to_codes(
    token_ids: List[int],
    tokenizer,
    token_prefix: str = "audio",
) -> np.ndarray:
    """
    Convert token IDs back to audio codes.
    
    Takes model output token IDs, decodes them, and extracts audio codes.
    
    Args:
        token_ids: List of token IDs from model generation
        tokenizer: HuggingFace tokenizer used for encoding
        token_prefix: Prefix used for audio tokens (default: "audio")
        
    Returns:
        Numpy array of audio codes
        
    Example:
        >>> # After model generation
        >>> output = model.generate(input_ids, max_new_tokens=200)
        >>> codes = audio_token_ids_to_codes(output[0].tolist(), tokenizer)
        >>> np.save("generated_audio.npy", codes)
    """
    # Decode token IDs to text
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # Extract audio codes
    return audio_tokens_to_codes(generated_text, token_prefix)


def save_audio_codes_for_codec(
    generated_text: str,
    output_path: str,
    token_prefix: str = "audio",
) -> int:
    """
    Extract audio codes from generated text and save for X-Codec-2.0.
    
    Convenience function that extracts codes and saves them in the format
    expected by X-Codec-2.0 decoder.
    
    Args:
        generated_text: String containing audio tokens
        output_path: Path to save .npy file (e.g., "output/generated.npy")
        token_prefix: Prefix used for audio tokens (default: "audio")
        
    Returns:
        Number of audio codes extracted and saved
        
    Example:
        >>> text = model_generate("Hello world")
        >>> num_codes = save_audio_codes_for_codec(text, "output/hello.npy")
        >>> print(f"Saved {num_codes} audio codes")
    """
    codes = audio_tokens_to_codes(generated_text, token_prefix)
    
    if len(codes) == 0:
        warnings.warn(f"No audio tokens found in generated text")
        return 0
    
    # Save in X-Codec-2.0 format
    np.save(output_path, codes)
    return len(codes)


def _process_tts_csv_row(
    row: pd.Series,
    codes_base_dir: Path,
    token_prefix: str = "audio",
) -> Optional[dict]:
    """
    Process a single CSV row into message format.
    
    Args:
        row: Pandas Series with 'filename' and 'text' columns
        codes_base_dir: Base directory containing VQ code files
        token_prefix: Prefix for audio tokens
        
    Returns:
        Dict with 'messages' key in chat format, or None if codes not found
    """
    filename = row["filename"]
    text = row["text"].strip()
    
    # Load audio codes
    codes = _load_audio_codes_from_file(filename, codes_base_dir)
    if codes is None:
        return None
    
    # Convert to message format
    audio_token_string = _audio_codes_to_token_string(codes, token_prefix)
    return {
        "messages": [
            {"content": text, "role": "user"},
            {"content": audio_token_string, "role": "assistant"},
        ]
    }


def load_tts_dataset_from_csv(
    csv_path: str,
    codes_base_dir: str,
    token_prefix: str = "audio",
    output_path: Optional[str] = None,
    test_size: float = 0.05,
    random_state: int = 42,
    verbose: bool = True,
) -> datasets.DatasetDict:
    """
    Load Text-to-Speech dataset from CSV and audio VQ codes.
    
    Converts CSV data with audio codes into HuggingFace DatasetDict format
    suitable for TTS training with discrete audio tokens.
    
    Args:
        csv_path: Path to CSV file with columns: 'filename', 'text'
        codes_base_dir: Base directory containing VQ code .npy files
        token_prefix: Prefix for audio tokens (default: "audio" -> <audio_0>)
        output_path: Optional path to save dataset using save_to_disk()
        test_size: Fraction of data to use for test set (default: 0.05)
        random_state: Random seed for train/test split (default: 42)
        verbose: Whether to print progress information
        
    Returns:
        DatasetDict with 'train' and 'test' splits containing message format:
        {
            'messages': [
                {'content': 'text transcription', 'role': 'user'},
                {'content': '<audio_0><audio_1>...', 'role': 'assistant'}
            ]
        }
        
    Example:
        >>> # Load TTS dataset with X-Codec-2.0 codes
        >>> dataset = load_tts_dataset_from_csv(
        ...     csv_path="data/tts_data.csv",
        ...     codes_base_dir="/path/to/vq_codes",
        ... )
        >>> # Use with trainer
        >>> trainer = Trainer(
        ...     train_dataset=dataset["train"],
        ...     eval_dataset=dataset["test"],
        ...     ...
        ... )
    """
    codes_base_dir = Path(codes_base_dir)
    
    # Load CSV
    if verbose:
        print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, sep="\t")
    df = df.dropna(subset=["filename", "text", "duration"])

    # Remove audios which are more than 13 seconds which is 13 * 25 tokens in the audio
    # more than 1.5 seconds for more than one block
    df = df[(df["duration"] <= 13) & (df["duration"] >= 1.5)]
    df = df.reset_index(drop=True)

    if verbose:
        print(f"Found {len(df)} entries")
    
    # Process dataset
    processed_data: List[dict] = []
    skipped: List[str] = []
    
    if verbose:
        print("Processing dataset...")
    
    for idx, row in df.iterrows():
        if verbose and idx > 0 and idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)}...")
        
        message = _process_tts_csv_row(row, codes_base_dir, token_prefix)
        
        if message is None:
            skipped.append(row["filename"])
        else:
            processed_data.append(message)
    
    if verbose:
        print(f"\nProcessed: {len(processed_data)} samples")
        print(f"Skipped: {len(skipped)} samples (codes not found)")
        if skipped and len(skipped) <= 5:
            print(f"Skipped files: {skipped}")
    
    # Split train/test
    train_data, test_data = train_test_split(
        processed_data, test_size=test_size, random_state=random_state
    )
    
    # Convert to HuggingFace Dataset objects
    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(test_data)
    
    # Create DatasetDict
    dataset_dict = datasets.DatasetDict(
        {"train": train_dataset, "test": test_dataset}
    )
    
    if verbose:
        print(f"\nCreated DatasetDict:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
    
    # Optionally save to disk (HuggingFace format)
    if output_path:
        dataset_dict.save_to_disk(output_path)
        if verbose:
            print(f"\nSaved dataset to: {output_path}")
            print(f"Load later with: datasets.load_from_disk('{output_path}')")
    
    return dataset_dict