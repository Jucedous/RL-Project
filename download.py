from transformers import AutoTokenizer, AutoModelForCausalLM

def load_local_model(model_path):
    try:
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token...")
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_path)

        print("Model and tokenizer loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

local_model_path = r"C:\Users\Owner\Downloads\meta-llamaLlama-3.2-1B"
tokenizer, model = load_local_model(local_model_path)

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50):
    """
    Generate text using the Llama model with attention mask.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during text generation: {e}")
        return ""

if __name__ == "__main__":
    prompt = "Q: What is 1 plus 1?\nA:"
    print("Generating text...")
    generated_text = generate_text(prompt)
    print("\nGenerated Text:")
    print(generated_text)