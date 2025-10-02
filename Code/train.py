from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


def train_tokenizer(lang: str, output_dir: str, cfg: DictConfig):
    tokenizer_path = Path(output_dir) / f"{lang}-tokenizer"
    if not tokenizer_path.exists():
        print(f"Training tokenizer for '{lang}'...")
        tokenizer_path.mkdir(parents=True, exist_ok=True)

        tokenizer_data_path = Path(output_dir) / f"{lang}-tokenizer-data.txt"
        dataset_name = f"{cfg.data.date}.{lang}"

        if not tokenizer_data_path.exists():
            print(f"Streaming dataset '{cfg.data.name}' ({dataset_name})"
                  f"to {tokenizer_data_path} for tokenizer training.")
            dataset_stream = load_dataset(cfg.data.name, dataset_name,
                                          split='train', streaming=True)
            with open(tokenizer_data_path, "w", encoding="utf-8") as f:
                for example in dataset_stream:
                    text = example['text'].strip()
                    if text:
                        f.write(text + "\n")

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[str(tokenizer_data_path)],
            vocab_size=cfg.tokenizer.vocab_size,
            min_frequency=cfg.tokenizer.min_frequency,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

        tokenizer.save_model(str(tokenizer_path))
        print(f"Tokenizer for '{lang}' trained and saved to {tokenizer_path}")
    else:
        print(f"Tokenizer for '{lang}' already exists at {tokenizer_path}")

    return str(tokenizer_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Pre-prep
    model_dir = Path(hydra.utils.to_absolute_path(cfg.paths.model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = train_tokenizer(cfg.language, str(model_dir), cfg)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    model = GPT2LMHeadModel(config=GPT2Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        **cfg.model
    ))
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model configured with {model.num_parameters():,} params")

    dataset_config_name = f"{cfg.data.date}.{cfg.language}"
    dataset = load_dataset(cfg.data.name, dataset_config_name, split='train')

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True,
                         max_length=cfg.model.n_positions)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.processing.num_workers,
        remove_columns=dataset.column_names
    )
    print(f"Dataset processed. Number of examples: {len(tokenized_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)

    # Actual training
    output_dir = f"./gpt2-{cfg.language}-checkpoints"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        **OmegaConf.to_container(cfg.training, resolve=True)
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    trainer.train()

    final_model_path = model_dir / f"gpt2-{cfg.language}-final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print(f"Final model and tokenizer saved to {final_model_path}")


if __name__ == "__main__":
    main()
