import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
from dotenv import load_dotenv
from groq import Groq

from rerank import hybrid_search, rerank, meta_by_id, text_by_id

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY is not set. Check your .env file.")
    sys.exit(1)

groq_client = Groq(api_key=api_key)



# import os
# os.environ["HF_HUB_OFFLINE"] = "1"

# import sys
# from dotenv import load_dotenv
# from groq import Groq
# from sentence_transformers import SentenceTransformer
# import chromadb

# # load_dotenv()
# # groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# load_dotenv()

# api_key = os.environ.get("GROQ_API_KEY")
# if not api_key:
#     print("ERROR: GROQ_API_KEY is not set. Check your .env file.")
#     sys.exit(1)




groq_client = Groq(api_key=api_key)



# model = SentenceTransformer("all-MiniLM-L6-v2")
# client = chromadb.PersistentClient(path="chroma_db")
# collection = client.get_collection("ranfy")



# def retrieve(question, k=3):
#     q_vec = model.encode(question).tolist()
#     results = collection.query(query_embeddings=[q_vec], n_results=k)
#     chunks = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     return chunks, metadatas









# def build_context(chunks, metadatas):
#     parts = []
#     for i, chunk in enumerate(chunks):
#         meta = metadatas[i]
#         parts.append(f"[Source: {meta['source']} page {meta['page']}]\n{chunk}")
#     return "\n\n".join(parts)



def build_context(doc_ids):
    parts = []
    for doc_id in doc_ids:
        meta = meta_by_id[doc_id]
        text = text_by_id[doc_id]
        parts.append(f"[Source: {meta['source']} page {meta['page']}]\n{text}")
    return "\n\n".join(parts)


# def ask(question, history, k=3):
#     chunks, metadatas = retrieve(question, k=k)
#     context = build_context(chunks, metadatas)

#     system_prompt = (
#         "You are a helpful assistant for RanFy Inc. "
#         "Answer the user's question using ONLY the context below. "
#         "If the answer is not in the context, say you don't know. "
#         "Be concise."
#     )

#     messages = [{"role": "system", "content": system_prompt}]
#     for turn in history:
#         messages.append(turn)
#     messages.append({
#         "role": "user",
#         "content": f"Context:\n{context}\n\nQuestion: {question}",
#     })

#     response = groq_client.chat.completions.create(
#         model="openai/gpt-oss-120b",
#         messages=messages,
#         temperature=0,
#     )
#     return response.choices[0].message.content, metadatas



def ask(question, history, k_retrieve=20, k_final=3):
    candidates = hybrid_search(question, k=k_retrieve)
    top_ids = rerank(question, candidates, top_n=k_final)
    context = build_context(top_ids)

    system_prompt = (
        "You are a helpful assistant for RanFy Inc. "
        "Answer the user's question using ONLY the context below. "
        "If the answer is not in the context, say you don't know. "
        "Be concise."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append(turn)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content, top_ids







# history = []

# print("RanFy RAG chat. Type 'exit' to quit.")
# while True:
#     try:
#         question = input("\nYou: ").strip()
#     except (KeyboardInterrupt, EOFError):
#         print()
#         break

#     if not question:
#         continue
#     if question.lower() in ("exit", "quit"):
#         break

#     answer, metadatas = ask(question, history)
#     print("\nBot:", answer)

#     seen = set()
#     print("Sources:")
#     for meta in metadatas:
#         key = (meta["source"], meta["page"])
#         if key not in seen:
#             print(f"  - {meta['source']}, page {meta['page']}")
#             seen.add(key)

#     history.append({"role": "user", "content": question})
#     history.append({"role": "assistant", "content": answer})




def main():
    history = []
    print("RanFy RAG chat. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        # answer, metadatas = ask(question, history)
        # print("\nBot:", answer)

        # seen = set()
        # print("Sources:")
        # for meta in metadatas:
        #     key = (meta["source"], meta["page"])
        #     if key not in seen:
        #         print(f"  - {meta['source']}, page {meta['page']}")
        #         seen.add(key)

        # history.append({"role": "user", "content": question})
        # history.append({"role": "assistant", "content": answer})
        answer, top_ids = ask(question, history)
        print("\nBot:", answer)

        seen = set()
        print("Sources:")
        for doc_id in top_ids:
            meta = meta_by_id[doc_id]
            key = (meta["source"], meta["page"])
            if key not in seen:
                print(f"  - {meta['source']}, page {meta['page']}")
                seen.add(key)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()