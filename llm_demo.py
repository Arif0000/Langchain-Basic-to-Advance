from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

llm =  ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3, max_completion_token=10)

result = llm.invoke(" what is the capptal of india")

print(result)