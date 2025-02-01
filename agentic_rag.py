import os
from typing import Dict, List, Any

# Dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from enum import Enum

class LLMProvider(str, Enum):
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"

# Configure environment variables
def configure_environment(provider: LLMProvider):
    required_keys = ['TAVILY_API_KEY', 'QDRANT_API_KEY']
    
    if provider == LLMProvider.DEEPSEEK:
        required_keys.append('DEEPSEEK_API_KEY')
    
    for key in required_keys:
        if key not in os.environ:
            raise ValueError(f'{key} environment variable is not set')

def load_and_process_documents(pdf_path: str) -> List[Any]:
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return text_splitter.split_documents(documents)

def setup_vectorstore(documents: List[Any], qdrant_url: str) -> Qdrant:
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Create vectorstore
    return Qdrant.from_documents(
        documents,
        embeddings,
        url=qdrant_url,
        prefer_grpc=True,
        collection_name="documents",
        api_key=os.environ["QDRANT_API_KEY"],
    )

def setup_agent(vectorstore: Qdrant, provider: LLMProvider = LLMProvider.OLLAMA) -> AgentExecutor:
    
    # Create tools with descriptions
    retriever = Tool(
        name="retriever",
        description="Search in the vector database for relevant information. Input should be a search query.",
        func=vectorstore.as_retriever().get_relevant_documents
    )
    
    search = TavilySearchResults()
    search.name = "tavily_search"
    search.description = "Search the web for real-time information. Input should be a search query."
    
    tools = [retriever, search]
    
    # Create LLM based on provider
    if provider == LLMProvider.DEEPSEEK:
        llm = ChatDeepSeek(
            model_name="deepseek-chat",
            temperature=0,
            api_key=os.environ["DEEPSEEK_API_KEY"]
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            model="deepseek-r1:1.5b",  # You can change this to any Ollama supported model
            temperature=0
        )
    
    # Create prompt for ReAct agent
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    # Create prompt for ReAct agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", template)
    ])
    
    # Create the agent using ReAct approach
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    return AgentExecutor(agent=agent, tools=tools)

def main():
    try:
        provider = LLMProvider.OLLAMA  # Change this to LLMProvider.DEEPSEEK when you want to use DeepSeek
        
        # Configure environment
        configure_environment(provider)
        
        # Load and process documents
        documents = load_and_process_documents("Kafka-e-book.pdf")
        
        # Setup vectorstore
        vectorstore = setup_vectorstore(documents, "https://813babd4-914d-49af-bed1-a70cf66d21b2.us-east4-0.gcp.cloud.qdrant.io:6333")
        
        # Setup agent
        agent_executor = setup_agent(vectorstore, provider)
        
        # Example query
        result = agent_executor.invoke(
            {
                "input": "What is Kafka?",
            }
        )
        print(result)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()