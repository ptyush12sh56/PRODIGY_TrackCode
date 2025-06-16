import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2_finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_finetuned")

prompt = "In the age of artificial intelligence,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    repetition_penalty=1.5,
    top_p=0.92,
    temperature=0.8,
    do_sample=True,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
