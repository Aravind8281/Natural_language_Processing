from transformers import BertTokenizer,BertModel
model_name="bert-base-uncased"
tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertModel.from_pretrained(model_name)
text = "Hugging Face Transformers is awesome!"
inputs=tokenizer(text,return_tensors="pt")
outputs=model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
