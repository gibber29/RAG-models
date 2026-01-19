import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

# --------------------------------------------------
# 1. Models & Automatic Rate Limiter
# --------------------------------------------------
# This replaces the manual 'delay'. 
# It allows 10 requests per minute (0.16 per second) without pausing your script.

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.15, 
    check_every_n_seconds=0.1, 
    max_bucket_size=2 
)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    max_retries=3,          
    rate_limiter=rate_limiter 
)

# --------------------------------------------------
# 2. Database Loading
# --------------------------------------------------

print("Loading FAISS index...", flush=True)
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# --------------------------------------------------
# 3. Streamlined Chain (No Delays)
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use ONLY the context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["question"]) | vectorstore.as_retriever(search_kwargs={"k": 3}),
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: x["chat_history"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# 4. Memory Management
# --------------------------------------------------

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    # Prune history to last 4 messages to keep token count low
    
    if len(store[session_id].messages) > 4:
        store[session_id].messages = store[session_id].messages[-4:]
    return store[session_id]

chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# --------------------------------------------------
# 5. Execution Loop
# --------------------------------------------------

def ask_question(query: str):
    try:
        # Directly invoke without manual countdowns
        answer = chat_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "fast_session"}}
        )
        print(f"\nAI: {answer}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    print("\nREADY (No delays). Type your question.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in {"exit", "quit"}: break
        if user_input.strip():

            ask_question(user_input)
