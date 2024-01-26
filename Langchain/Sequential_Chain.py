import langchain 
llm_huggingface=HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0,"max_length":64})
capital_prompt = PromptTemplate(input_variables=["country"], template="Tell me the capital of {country}")
famous_prompt = PromptTemplate(input_variables=["famous"], template="Tell me the capital of {famous}")
capital_chain = langchain.LLMChain(llm=llm_huggingface, prompt=capital_prompt)
famous_chain = langchain.LLMChain(llm=llm_huggingface, prompt=famous_prompt)
chain = langchain.chains.SimpleSequentialChain(chains=[capital_chain, famous_chain])
result = chain.run("India")
print(result)
