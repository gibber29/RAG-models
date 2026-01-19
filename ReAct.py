import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# 1. Models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. Tool (FAISS)
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

@tool
def search_docs(query: str):
    """Search internal docs for technical facts."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

# 3. Middleware (Correct Class Implementation)
# This will watch your tokens and summarize old messages to save space.

memory_manager = SummarizationMiddleware(
    model=llm,
    trigger=("tokens", 3000),  # Summarize when history hits 3000 tokens
    keep=("messages", 5)       # Always keep the 5 most recent messages
)

# 4. Agent Setup
checkpointer = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[search_docs],
    system_prompt="You are a robust assistant. Always use search_docs before answering.",
    middleware=[memory_manager],
    checkpointer=checkpointer
)

# 5. Persistent Chat Loop
def chat(query, session_id="user_1"):
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"messages": [("user", query)]}, config)
    
    # The 'result' contains the full message history of the state
    last_message = result["messages"][-1]
    
    print(f"\nAgent: {last_message.content if isinstance(last_message.content, str) else last_message.text}")

if __name__ == "__main__":
    print("--- Robust Conversational RAG Active ---")
    while True:
        u_input = input("\nUser: ")
        if u_input.lower() in ["exit", "quit"]: break

        chat(u_input)
