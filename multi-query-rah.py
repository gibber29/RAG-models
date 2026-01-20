# This version combines:

# 1.Multi-Query Expansion (3 versions of the question).

# 2.Deduplicated Retrieval (Parallel local FAISS search).

# 3.Chat Memory (Remembers previous context).

# 4.Clean Output (Strips the technical metadata/signatures).


import os
from operator import itemgetter # Add this import
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

# 1. Models & Vectorstore
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 2. Multi-Query Logic (Remains the same)
def get_expanded_docs(inputs):
    question = inputs["question"]
    # Single call to expand queries
    variations_raw = llm.invoke(f"Generate 3 search queries for: {question}").content
    queries = [q.strip() for q in variations_raw.split("\n") if q.strip()] + [question]
    
    all_docs = []
    for q in queries:
        all_docs.extend(vectorstore.similarity_search(q, k=2))
            
    unique_docs = {doc.page_content for doc in all_docs}
    return "\n\n".join(unique_docs)

# --------------------------------------------------
# 3. FIXED Conversational Chain
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based ONLY on the context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# The "Base" chain must take a dict and return the prompt's requirements
rag_chain = (
    {
        "context": RunnableLambda(get_expanded_docs),
        "question": itemgetter("question"),
        "history": itemgetter("history") # This extracts the list correctly
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# 4. Persistence Wrapper
# --------------------------------------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# --------------------------------------------------
# 5. The User Loop
# --------------------------------------------------
def start_chat():
    print("--- CONVERSATIONAL MULTI-QUERY RAG ---")
    session_id = "user_001"
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # We pass just the question; the wrapper adds 'history' automatically
        response = conversational_rag.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\nAI: {response}")

if __name__ == "__main__":

    start_chat()
