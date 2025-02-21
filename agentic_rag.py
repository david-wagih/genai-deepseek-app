import os
from typing import Dict, List, Any, Optional
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
import sys
from functools import lru_cache
import time

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
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Optimize chunking with better overlap and size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,  # Add overlap for better context
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # More intelligent splitting
    )
    return text_splitter.split_documents(documents)


class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        
    @lru_cache(maxsize=100)
    def get_relevant_documents(self, query: str) -> List[Any]:
        """Cache similar queries to avoid redundant vector searches"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        
        results = self.vectorstore.as_retriever().get_relevant_documents(query)
        
        if not results:
            # If no results found, try a broader search
            results = self.vectorstore.as_retriever(
                search_kwargs={"k": 5, "score_threshold": 0.3}
            ).get_relevant_documents(query)
        
        return results
    
    def setup_vectorstore(self, documents: List[Any], qdrant_url: str, collection_name: str = "documents") -> None:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )
        
        # Create vectorstore with unique collection name
        self.vectorstore = Qdrant.from_documents(
            documents,
            embeddings,
            url=qdrant_url,
            prefer_grpc=True,
            collection_name=collection_name,  # Use unique collection name
            api_key=os.environ["QDRANT_API_KEY"]
        )
        
        # Configure the retriever
        self.vectorstore._retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "score_threshold": 0.5,
                "fetch_k": 20
            }
        )
    
    def setup_agent(self, provider: LLMProvider = LLMProvider.OLLAMA) -> AgentExecutor:
        retriever = Tool(
            name="retriever",
            description="Search the document database for relevant information. Use specific keywords from the question.",
            func=self.get_relevant_documents
        )
        
        search = TavilySearchResults()
        search.name = "tavily_search"
        search.description = "Search the web for current information not found in the documents."
        
        tools = [retriever, search]
        
        system_message = """You are a knowledgeable AI assistant analyzing documents. NEVER use XML tags.

When responding, STRICTLY follow this format:

Thought: Brief analysis of what to do next
Action: retriever
Action Input: "specific search query"
Observation: (system provides search results)
Thought: Analysis of the results
Final Answer: Clear, concise conclusion

Guidelines:
1. Start EVERY response with "Thought:"
2. Follow EACH thought with "Action:" and "Action Input:"
3. Wait for "Observation:" from the system
4. End with a "Final Answer:"
5. NEVER use XML tags or <think> tags
6. NEVER skip steps in the format"""

        template = """Available Tools:
{tool_names}

Tool Details:
{tools}

Example of CORRECT format:
Thought: I need to search the document for main topics
Action: retriever
Action Input: "document main topics and key points"
Observation: (system provides results)
Thought: The document appears to be about...
Final Answer: This document contains...

Your task: {input}
{agent_scratchpad}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", template),
        ])

        # Configure LLM with stricter settings
        llm = (ChatDeepSeek(
            model_name="deepseek-chat",
            temperature=0.3,  # Lower temperature for more consistent formatting
            max_tokens=2000,
            top_p=0.1,  # More focused responses
            api_key=os.environ["DEEPSEEK_API_KEY"]
        ) if provider == LLMProvider.DEEPSEEK 
        else ChatOllama(model="mistral:latest", temperature=0.3))
        
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=3,
            max_execution_time=30,
            return_intermediate_steps=True
        )

def should_use_rag(query: str) -> bool:
    """Determine if query needs RAG or can be answered directly"""
    simple_queries = ["hello", "hi", "how are you"]
    if query.lower() in simple_queries:
        return False
    return True

def chat_loop(agent_executor: AgentExecutor):
    """Interactive chat loop to converse with the agent."""
    print("\nWelcome to the Document Chat Assistant!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'help' to see available commands.")
    print("-" * 50)

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye! Thanks for chatting.")
                break

            # Handle help command    
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- 'exit' or 'quit': End the conversation")
                print("- 'help': Show this help message")
                print("- Any other input will be treated as a question about the document")
                continue

            if not should_use_rag(user_input):
                print("\nAssistant: ", end='', flush=True)
                print("Simple response without RAG")
                continue

            if not user_input:
                continue

            # Get response from agent
            print("\nAssistant: Searching documents...", end='', flush=True)
            try:
                result = agent_executor.invoke(
                    {
                        "input": user_input,
                    }
                )
                print("\rAssistant: ", end='')  # Clear the "Searching" message
                
                if isinstance(result, dict):
                    if "output" in result:
                        print(result["output"])
                    elif "intermediate_steps" in result:
                        steps = result["intermediate_steps"]
                        if steps:
                            # Show progress through steps
                            print(f"\nFound {len(steps)} relevant pieces of information.")
                            print(steps[-1][1])  # Print final answer
            except Exception as e:
                print("\rAssistant: I encountered an error. Let me try a different approach.")
                # Add fallback logic here

        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for chatting.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try asking your question again.")

class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        
    def start_operation(self, operation_name: str):
        self.metrics[operation_name] = {"start": time.time()}
        
    def end_operation(self, operation_name: str):
        if operation_name in self.metrics:
            self.metrics[operation_name]["duration"] = (
                time.time() - self.metrics[operation_name]["start"]
            )
            
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

def main():
    try:
        monitor = PerformanceMonitor()
        rag_system = RAGSystem()
        
        monitor.start_operation("document_processing")
        documents = load_and_process_documents("Kafka-e-book.pdf")
        monitor.end_operation("document_processing")
        
        monitor.start_operation("vectorstore_setup")
        rag_system.setup_vectorstore(
            documents, 
            "https://813babd4-914d-49af-bed1-a70cf66d21b2.us-east4-0.gcp.cloud.qdrant.io:6333"
        )
        monitor.end_operation("vectorstore_setup")
        
        provider = LLMProvider.OLLAMA
        
        print("Initializing the chat system...")
        print("Setting up AI agent...")
        
        # Configure environment
        configure_environment(provider)
        
        # Setup agent
        agent_executor = rag_system.setup_agent(provider)
        
        # Start the chat loop
        chat_loop(agent_executor)
        
        # Log metrics
        print("Performance metrics:", monitor.get_metrics())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()