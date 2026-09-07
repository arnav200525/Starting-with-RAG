from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from groq import Groq

import os
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


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
    
    followup_words = [
    "it", "they", "them", "that",
    "those", "he", "she", "his", "her"
]

    question_lower = question.lower()

    needs_rewrite = any(
        word in question_lower.split()
        for word in followup_words
    )

    if chathistory and needs_rewrite:
        
        rewrite_prompt = f"""
    Given the chat history below, rewrite the new question so that it is standalone and searchable.

    Chat History:
    {chr(10).join([f"{m.type}: {m.content}" for m in chathistory])}

    New Question:
    {question}

    Return ONLY the rewritten question.
    """

        rewrite_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )

        standalone_question = (
            rewrite_response.choices[0]
            .message.content
            .strip()
        )

    else:
        standalone_question = question

    retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.5
    }
)
    relevant_docs = retriever.invoke(standalone_question)
    print("Retrieved Docs:", len(relevant_docs))

    context = "\n\n".join([
    f"Document {i+1}:\n{doc.page_content[:600]}"
    for i, doc in enumerate(relevant_docs)
])


    rag_prompt = f"""
Answer the question using only the information from the documents below.
If the answer is not present, say:
"I don't have enough information to answer that question."

Question:
{standalone_question}

Documents:
{context}
"""

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": rag_prompt
        }
    ]
)

    answer = (
        response.choices[0]
        .message.content
        .strip()
    )

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
