from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
prompt_template=PromptTemplate(input_variables=["country"],template="Tell me the capital of this {country}")
prompt_template.format(country="India")
chain=LLMChain(llm=llm,prompt=prompt_template)
print(chain.run("India")
