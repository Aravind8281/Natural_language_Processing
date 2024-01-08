import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) 
texts = ["I love Hugging Face Transformers!", "I dislike fine-tuning."]
labels = [1, 0]  
tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
dataset = TensorDataset(tokenized_texts["input_ids"], tokenized_texts["attention_mask"], torch.tensor(labels))
loader = DataLoader(dataset, batch_size=2, shuffle=True)
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3  

# Fine-tuning loop
for epoch in range(num_epochs):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        inputs, attention_mask, labels = batch
        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
model.save_pretrained("fine_tuned_model")
