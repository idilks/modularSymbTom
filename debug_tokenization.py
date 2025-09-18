"""
Debug how specific location words get tokenized
"""
from transformers import AutoTokenizer

# Test with the model from your error
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

test_words = ["radiator", "fireplace", "kitchen", "garden", "pantry", "nightstand"]

for word in test_words:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"{word}: {len(tokens)} tokens -> {decoded}")