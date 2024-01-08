from transformers import GPT2LMHeadModel, GPT2Tokenizer
def generate_response(prompt, model, tokenizer, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
print("Chatbot: Hello! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    chatbot_response = generate_response(user_input, model, tokenizer)
    print("Chatbot:", chatbot_response)
