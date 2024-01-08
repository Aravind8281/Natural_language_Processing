from transformers import pipeline
qa_model = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer='bert-large-uncased-whole-word-masking-finetuned-squad')
context = "Hugging Face is a great platform for natural language processing."
question = "What is Hugging Face?"
answer=qa_model(question=question,context=context)
print(answer)
