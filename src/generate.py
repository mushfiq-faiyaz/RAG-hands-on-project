import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])



from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("ranfy")

question = "Can I get my money back?"

q_vec = model.encode(question).tolist()
results = collection.query(query_embeddings=[q_vec], n_results=3)

chunks = results["documents"][0]
metadatas = results["metadatas"][0]

context_parts = []
for i, chunk in enumerate(chunks):
    meta = metadatas[i]
    context_parts.append(f"[Source: {meta['source']} page {meta['page']}]\n{chunk}")

context = "\n\n".join(context_parts)




#print(context)



system_prompt = (
    "You are a helpful assistant for RanFy Inc. "
    "Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't know. "
    "Be concise."
)

user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0,
)

print(response.choices[0].message.content)

