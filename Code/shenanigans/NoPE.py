import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def compare_gpt2_generations():

    model_name = 'openai-community/gpt2'
    print(f"Loading model '{model_name}' and tokenizer...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
    except OSError:
        print(f"Error: Model '{model_name}' not found.")
        print("Please ensure you have an internet connection to download the model.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = "Jack and Jill"
    max_length = 70
    generation_params = {
        "max_length": max_length,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 2,
        "early_stopping": True
    }

    print("\n" + "="*50)
    print("Case 1: Generating text with standard positional embeddings.")
    print("="*50)

    inputs_normal = tokenizer(prompt, return_tensors="pt").to(device)
    outputs_normal = model.generate(**inputs_normal, **generation_params)
    text_normal = tokenizer.decode(outputs_normal[0], skip_special_tokens=True)

    print(f"\nPrompt: '{prompt}'")
    print("Generated Text (Normal):")
    print(text_normal)

    print("\n" + "="*50)
    print("Case 2: Generating text WITHOUT positional embeddings.")
    print("="*50)

    pos_embedding_layer = model.transformer.wpe
    original_forward = pos_embedding_layer.forward

    try:
        def new_forward(position_ids):
            return torch.zeros(
                position_ids.shape[0],
                position_ids.shape[1],
                pos_embedding_layer.embedding_dim,
                device=device
            )

        pos_embedding_layer.forward = new_forward

        inputs_no_pos = tokenizer(prompt, return_tensors="pt").to(device)
        outputs_no_pos = model.generate(**inputs_no_pos, **generation_params)
        text_no_pos = tokenizer.decode(outputs_no_pos[0], skip_special_tokens=True)

        print(f"\nPrompt: '{prompt}'")
        print("Generated Text (Without Positional Embeddings):")
        print(text_no_pos)

    finally:
        # IMPORTANT: Restore the original forward method to avoid side-effects.
        pos_embedding_layer.forward = original_forward


if __name__ == "__main__":
    compare_gpt2_generations()
