from transformers import GPT2LMHeadModel, GPT2Tokenizer
model="gpt2"
tokenizer=GPT2Tokenizer.from_pretrained(model)
model=GPT2LMHeadModel.from_pretrained(model)
text="Quick fox jumps over a wall"
input_ids=tokenizer.encode(text,return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
print(output)
