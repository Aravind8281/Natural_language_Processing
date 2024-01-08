from transformers import RobertaTokenizer, RobertaModel
import torch
tokenizer=RobertaTokenizer.from_pretrained("roberta-base")
model=RobertaModel.from_pretrained("roberta-base")
text="Natural language processing helps to make machines to understand human language"
input_ids=tokenizer.encode(text,return_tensors="pt")
output=model(input_ids)
print(output.last_hidden_state)
