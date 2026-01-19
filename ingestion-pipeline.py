import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def create_vector_store():
    print("Loading documents...", flush=True)

    loader = DirectoryLoader(
    "./docs", 
    glob="*.txt", 
    loader_cls=TextLoader, 
    loader_kwargs={"encoding": "utf-8"} # Removed 'errors'
    )

    documents = loader.load()
    if not documents:
        raise ValueError("No documents found in ./docs")

    print(f"Splitting {len(documents)} documents...", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.", flush=True)
    print("Creating embeddings...", flush=True)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    batch_size = 50
    vectorstore = None
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        progress = min(((i + batch_size) / total_chunks) * 100, 100)
        print(f"[{progress:.1f}%] Processing batch {i // batch_size + 1}", flush=True)

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedding_model)

                # 👇 ADD THIS LINE (ONLY ONCE IS ENOUGH)
            print(type(vectorstore))

        else:
            vectorstore.add_documents(batch)

        time.sleep(2)

    print("Saving vector store to local file...", flush=True)
    vectorstore.save_local("faiss_index")
    print("--- SUCCESS: Saved to 'faiss_index' folder ---")

if __name__ == "__main__":
    create_vector_store()
