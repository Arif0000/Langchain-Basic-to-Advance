from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

prompt1 = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['top[ic]']
)
model = ChatGoogleGenerativeAI()
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}', 
    input_variables=['text']
)
joke_gen_chain = RunnableSequence(
    {'joke': RunnablePassthrough(),
     'explainations': RunnableSequence(prompt2, model, parser)}
    
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({'topic':'cricket'}))