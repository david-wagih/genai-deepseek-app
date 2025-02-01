import os
from typing import Dict, List, Any

# Dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_deepseek import ChatDeepSeek
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_tool

# Configure environment variables
def configure_environment():
    required_keys = ['ATHINA_API_KEY', 'TAVILY_API_KEY', 'DEEPSEEK_API_KEY', 'QDRANT_API_KEY']
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

def setup_agent(vectorstore: Qdrant) -> AgentExecutor:
    # Create tools
    retriever = vectorstore.as_retriever()
    search = TavilySearchResults()
    tools = [
        {
            "type": "retrieval",
            "function": retriever,
            "description": "Use this tool to retrieve information from the vector database."
        },
        {
            "type": "search",
            "function": search,
            "description": "Use this tool to search for real-time information from the web."
        }
    ]
    
    # Create LLM
    llm = ChatDeepSeek(
        model_name="deepseek-chat",
        temperature=0,
        api_key=os.environ["DEEPSEEK_API_KEY"]
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that can use tools to get information and answer questions accurately."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    llm_with_tools = llm.bind(
        tools=[format_tool_to_openai_tool(t) for t in tools]
    )
    
    agent = create_openai_tools_agent(llm_with_tools, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

def main():
    try:
        # Configure environment
        configure_environment()
        
        # Load and process documents
        documents = load_and_process_documents("Kafka-e-book.pdf")
        
        # Setup vectorstore
        vectorstore = setup_vectorstore(documents, "https://813babd4-914d-49af-bed1-a70cf66d21b2.us-east4-0.gcp.cloud.qdrant.io:6333")
        
        # Setup agent
        agent_executor = setup_agent(vectorstore)
        
        # Example query
        result = agent_executor.invoke({"input": "What is Kafka?"})
        print(result)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()