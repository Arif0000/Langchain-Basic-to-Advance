from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
actors_info = [
    "Shah Rukh Khan (SRK) is one of the biggest superstars of Indian cinema, known as the King of Bollywood. He is famous for romantic, dramatic, and intense roles and has a massive global fan following.",

    "Salman Khan is known for his larger-than-life screen presence and action-packed films. He has delivered many blockbuster movies and is admired for his mass appeal.",

    "Aamir Khan is often called Mr. Perfectionist for his dedication to realism and detail in acting. He is known for socially relevant and critically acclaimed films.",

    "Ranveer Singh is known for his energetic performances and versatility. He effortlessly adapts to diverse roles ranging from historical characters to modern youth icons.",

    "Ranbir Kapoor is admired for his natural acting style and emotional depth. He is especially popular for coming-of-age and romantic drama films.",

    "Hrithik Roshan is known for his exceptional dancing skills, action roles, and charismatic screen presence. He is often regarded as one of the most stylish actors in Bollywood."
]

query = " tell me about Tarik"

doc_em = embeddings.embed_documents(actors_info)
query_emb = embeddings.embed_query(query)
print(cosine_similarity([query_emb], doc_em)[0])

score = cosine_similarity([query_emb], doc_em)[0]
index, score = sorted(list(enumerate(score)), key=lambda x:x[1])[-1]

print(query)
print(actors_info[index])
print("similarity score is:", score)

