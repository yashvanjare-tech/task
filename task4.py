from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
prompt = input("Enter a topic or sentence to start with: ")
inputs = tokenizer.encode(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        num_return_sequences=1
    )
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)
