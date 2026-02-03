from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = OllamaLLM(
    model="phi3", 
    # model = "llama3:8b", 
    temperature=0.2,
    num_predict=256,
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

query = input("Enter your Question>>")

retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)


def response_generation(user_query, loaded_documents):
    input = f"""Based on the following documents, please answer this question: {user_query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in loaded_documents])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    response = llm.invoke(input)
    return response


def main():
    '''
    print(f"User Query: {query}")
    print("--- Context ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n") '''

    final_output = response_generation(query, relevant_docs)
    print("-" * 50)
    print(final_output)


if __name__ == "__main__":
    main()
