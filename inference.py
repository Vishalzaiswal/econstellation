import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/llama3b_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
text = "Analyze the following econometric data..."
print(generate_response(text))
