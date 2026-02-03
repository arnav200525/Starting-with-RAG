from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

chathistory = []


def ask_question(question):
    global chathistory
    print("-"*90)
    print("User Asked:", question)
    
    if chathistory:
        rewrite_prompt = f"""
Given the chat history below, rewrite the new question so that it is standalone and searchable.

Chat History:
{chr(10).join([f"{m.type}: {m.content}" for m in chathistory])}

New Question:
{question}

Return ONLY the rewritten question.
"""

        rewrite_response = client.models.generate_content(
            model="gemini-flash-latest", contents=rewrite_prompt
        )

        standalone_question = rewrite_response.text.strip()
    else:
        standalone_question = question

    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(standalone_question)

    context = "\n".join([f"- {doc.page_content}" for doc in relevant_docs])

    rag_prompt = f"""
Answer the question using only the information from the documents below.
If the answer is not present, say:
"I don't have enough information to answer that question."

Question:
{standalone_question}

Documents:
{context}
"""

    response = client.models.generate_content(
        model="gemini-flash-latest", contents=rag_prompt
    )

    answer = response.text.strip()

    chathistory.append(HumanMessage(content=question))
    chathistory.append(AIMessage(content=answer))

    print(f"Bot: {answer}")
    print("-"*90)
    return answer


def start():
    print("Ask me question! Type 'quit' to exit")
    while True:
        que = input("\n\n\nYour Query:")
        print("\n\n\n")

        if que == "quit":
            print("Exited!")
            break
        else:
            ask_question(que)


if __name__ == "__main__":
    start()
