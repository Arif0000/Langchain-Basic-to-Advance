from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts PromptTemplate
from langchain_core.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linked post about {topic}',
    input_variables=['topic']
)
model = ChatGoogleGenerativeAI()

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableParallel(prompt1,model,pars),
        'linkedin': RunnableParallel(prompt2,model,parser)
    }
)

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linked'])