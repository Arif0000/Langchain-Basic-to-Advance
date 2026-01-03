from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
    temperature=0.3)

result = model.invoke("who is Mohd Ari Ansari")

print(result)