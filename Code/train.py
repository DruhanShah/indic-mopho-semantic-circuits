from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM as AutoModel,
    DataCollatorForLanguageModeling as Collator,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils import init_environ, set_seed


def clean_text(examples):
    return {"text": [text for text in examples["text"]
                     if text and not text.isspace()]}


def group_texts(examples, block_size):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig) -> None:

    init_environ(cfg.paths.assets_dir)
    set_seed(cfg.seed)

    print(f"Starting training for language: {cfg.language}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path.cwd()
    lang = cfg.language
    num_workers = cfg.processing.num_workers
    block_size = cfg.model.n_ctx

    print(f"Loading Wikipedia dataset for '{lang}'")
    raw_dataset = load_dataset(
        "wikimedia/wikipedia",
        f"{cfg.data.date}.{lang}",
        split=cfg.data.split,
    )
    dataset = raw_dataset.map(
        clean_text,
        batched=True,
        num_proc=num_workers,
        remove_columns=raw_dataset.column_names,
    )

    tokenizer_path = output_dir / "tokenizer"
    tokenizer_path.mkdir(parents=True, exist_ok=True)

    print(f"Training tokenizer for '{lang}'...")

    def text_iterator():
        for i in range(0, len(dataset), 1000):
            yield dataset[i:i + 1000]["text"]

    bpe_tokenizer = ByteLevelBPETokenizer()
    bpe_tokenizer.train_from_iterator(
        text_iterator(),
        vocab_size=cfg.tokenizer.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    bpe_tokenizer.save_model(str(tokenizer_path))

    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_path))
    tokenizer.add_special_tokens(
        {
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
        }
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
    )
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        fn_kwargs={"block_size": block_size},
    )
    print(f"Dataset prepared. Total samples: {len(lm_dataset)}")

    print("Configuring GPT-2 model...")
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    model_kwargs["n_layer"] = model_kwargs.pop("n_l")
    model_kwargs["n_head"] = model_kwargs.pop("n_h")
    model_kwargs["n_positions"] = model_kwargs.pop("n_ctx")

    model_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **model_kwargs,
    )
    model = AutoModel.from_config(model_config)

    data_collator = Collator(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **OmegaConf.to_container(cfg.training, resolve=True)
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Training finished. Saved model and tokenizer to {output_dir}")


if __name__ == "__main__":
    train()
