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

def train_tokenizer(data_path: str, lang: str, output_dir: str, vocab_size: int, min_frequency: int):
    tokenizer_path = Path(output_dir) / f"{lang}-tokenizer"
    if not tokenizer_path.exists():
        print(f"Training tokenizer for '{lang}'...")
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        
        tokenizer = ByteLevelBPETokenizer()

        tokenizer.train(
            files=[data_path],
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

        tokenizer.save_model(str(tokenizer_path))
        print(f"Tokenizer for '{lang}' trained and saved to {tokenizer_path}")
    else:
        print(f"Tokenizer for '{lang}' already exists at {tokenizer_path}")
    
    return str(tokenizer_path)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Pre-preparation
    model_dir = Path(hydra.utils.to_absolute_path(cfg.paths.model_dir))
    data_dir = Path(hydra.utils.to_absolute_path(cfg.paths.data_dir))
    
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = data_dir / f"{cfg.language}_wiki.txt"
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_file}. "
            f"Please run the data preparation steps outlined in the README."
        )

    tokenizer_path = train_tokenizer(
        str(data_file), 
        cfg.language, 
        str(model_dir),
        cfg.tokenizer.vocab_size,
        cfg.tokenizer.min_frequency
    )
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    model = GPT2LMHeadModel(config=GPT2Config(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        **cfg.model
    ))
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model configured. Number of parameters: {model.num_parameters():,}")

    dataset = load_dataset("text", data_files={"train": str(data_file)}, split="train")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.model.n_positions)
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=cfg.processing.num_workers, 
        remove_columns=["text"]
    )
    print(f"Dataset processed. Number of examples: {len(tokenized_dataset)}")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Actual training
    output_dir = f"./gpt2-{cfg.language}-checkpoints"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        **OmegaConf.to_container(cfg.training, resolve=True) # Unpack training args
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
