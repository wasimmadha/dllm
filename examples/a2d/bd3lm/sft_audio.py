"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/a2d/bd3lm/sft.py

- 8 GPUs (ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/a2d/bd3lm/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/sft.py"

- 2 Nodes, 16 GPUs (ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/sft.py"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers

import dllm
from dllm.utils.data import load_tts_dataset_from_csv 

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "models/a2d/Qwen3-0.6B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/a2d/Qwen3-0.6B/mdlm/alpaca"
    group_by_length: bool = True
    learning_rate: float = 1e-4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    eval_steps: float = 0.1
    save_steps: float = 0.1
    # a2d-specific
    block_size: int = 32
    right_shift_logits: bool = False


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Get original vocab size
    original_vocab_size = len(tokenizer)
    # print(f"Original vocab size: {original_vocab_size}")

    # Add audio tokens
    num_audio_tokens = 65536
    audio_tokens = [f"<audio_{i}>" for i in range(num_audio_tokens)]
    tokenizer.add_tokens(audio_tokens)

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # # Show before/after
    # print(f"New vocab size: {len(tokenizer)}")
    # print(f"Added: {len(tokenizer) - original_vocab_size} tokens")
    # print(f"Model embedding shape: {model.get_input_embeddings().weight.shape}")

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = load_tts_dataset_from_csv(
            csv_path="/root/research/X-Codec-2.0/sample-train-clean-100.csv",
            codes_base_dir="/root/research/X-Codec-2.0/codes_output/vq_codes",
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_mdlm_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )

        print("Load preprocessed data: ", data_args.load_preprocessed_data)
        print("--------------------------------")

        print("Dataset Info: ")
        print(dataset)
        print(dataset["train"].column_names)
        print(dataset["test"].column_names)
        print("--------------------------------")
        print(dataset["train"][0])
        print(dataset["test"][0])
        print("--------------------------------")

        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    trainer = dllm.core.trainers.BD3LMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        block_size=training_args.block_size,
        right_shift_logits=training_args.right_shift_logits,
        data_collator=(
            dllm.core.trainers.bd3lm.AppendEOSBlockWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                ),
                block_size=training_args.block_size,
            )
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
