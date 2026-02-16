from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

def word_count(text):
    return len(text.split())
prompt = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI()
parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
    
})

final_chain = final_chain.invoke({'topic':'AI'})
fianl_result = """{} \n word count -{}""".format(result['joke'],result['word_count'])

print(fianl_result)

