import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

MODEL_NAME = "gpt2"
LANGUAGES = ["en", "hi", "te"]
BLOCK_SIZE = 128
VOCAB_SIZE = 30_000

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= BLOCK_SIZE:
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

for lang in LANGUAGES:
    print(f"Loading Wikipedia dataset for {lang}")
    raw_dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split='train[:1%]')

    def clean_text(examples):
        return {'text': [text for text in examples['text'] if text and not text.isspace()]}

    dataset = raw_dataset.map(clean_text, batched=True,
                              num_proc=4, remove_columns=raw_dataset.column_names)
    output_dir = Path(f"./gpt2-wikipedia-{lang}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer"

    if not tokenizer_path.exists() or not any(tokenizer_path.iterdir()):
        print(f"Training tokenizer for '{lang}'...")
        tokenizer_path.mkdir(exist_ok=True)

        def text_iterator():
            for i in range(0, len(dataset), 1000):
                yield dataset[i : i + 1000]["text"]

        bpe_tokenizer = ByteLevelBPETokenizer()
        bpe_tokenizer.train_from_iterator(
            text_iterator(),
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )
        bpe_tokenizer.save_model(str(tokenizer_path))
    else:
        print(f"Tokenizer exists for {lang}")

    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_path))
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
    })

    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=True, num_proc=4, remove_columns=["text"])
    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)

    print(f"Dataset prepared for '{lang}'. Total samples: {len(lm_dataset)}")

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        vocab_size=len(tokenizer),
        n_ctx=BLOCK_SIZE,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AutoModelForCausalLM.from_config(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=500,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    print(f"Training finished. Saving model for '{lang}' to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
