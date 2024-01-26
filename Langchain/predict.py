from langchain import HuggingFaceHub
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
llm_huggingface=HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0,"max_length":64})
output=llm_huggingface.predict("Can you tell me the capital of Russia")
print(output)
