import streamlit as st
import os
from agentic_rag import RAGSystem, LLMProvider, configure_environment, PerformanceMonitor
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
from datetime import datetime
from qdrant_client import QdrantClient

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if 'monitor' not in st.session_state:
    st.session_state.monitor = PerformanceMonitor()
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = None
if 'collections' not in st.session_state:
    st.session_state.collections = []

def process_document(uploaded_file) -> List[Any]:
    """Process uploaded document and return chunks"""
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    
    # Remove temporary file
    os.remove("temp.pdf")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def get_unique_collection_name(file_name: str) -> str:
    """Generate a unique collection name based on file name and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
    return f"doc_{file_hash}_{timestamp}"

def initialize_rag_system(documents, file_name: str):
    """Initialize the RAG system with documents"""
    try:
        st.session_state.monitor.start_operation("vectorstore_setup")
        
        # Create new RAG system instance for clean state
        st.session_state.rag_system = RAGSystem()
        
        # Generate unique collection name
        collection_name = get_unique_collection_name(file_name)
        
        # Store collection name in session state
        if 'collections' not in st.session_state:
            st.session_state.collections = []
        st.session_state.collections.append(collection_name)
        
        # Setup vectorstore with new collection
        st.session_state.rag_system.setup_vectorstore(
            documents,
            "https://813babd4-914d-49af-bed1-a70cf66d21b2.us-east4-0.gcp.cloud.qdrant.io:6333",
            collection_name=collection_name
        )
        
        provider = LLMProvider.OLLAMA
        configure_environment(provider)
        
        # Setup agent
        st.session_state.agent_executor = st.session_state.rag_system.setup_agent(provider)
        st.session_state.monitor.end_operation("vectorstore_setup")
        st.session_state.documents_processed = True
        st.session_state.current_collection = collection_name
        st.session_state.messages = []  # Clear chat history
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        raise e

def delete_collection(collection_name: str):
    """Delete a collection from Qdrant"""
    try:
        client = QdrantClient(
            url="https://813babd4-914d-49af-bed1-a70cf66d21b2.us-east4-0.gcp.cloud.qdrant.io:6333",
            api_key=os.environ["QDRANT_API_KEY"]
        )
        client.delete_collection(collection_name)
        st.session_state.collections.remove(collection_name)
        if st.session_state.current_collection == collection_name:
            st.session_state.current_collection = None
            st.session_state.documents_processed = False
            st.session_state.messages = []
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")

def switch_collection(collection_name: str):
    """Switch to a different document collection"""
    try:
        # Create new RAG system instance for clean state
        st.session_state.rag_system = RAGSystem()
        st.session_state.current_collection = collection_name
        st.session_state.messages = []  # Clear chat history
        
        # Initialize vectorstore with selected collection
        st.session_state.rag_system.setup_vectorstore(
            [],  # No need to reprocess documents
            "https://813babd4-914d-49af-bed1-a70cf66d21b2.us-east4-0.gcp.cloud.qdrant.io:6333",
            collection_name=collection_name
        )
        
        # Setup agent
        provider = LLMProvider.OLLAMA
        configure_environment(provider)
        st.session_state.agent_executor = st.session_state.rag_system.setup_agent(provider)
        st.session_state.documents_processed = True
        
    except Exception as e:
        st.error(f"Error switching collection: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="📚 Document Q&A System",
    page_icon="📚",
    layout="wide"
)

# Main title
st.title("📚 Interactive Document Q&A System")
st.markdown("""
This system allows you to:
- Upload PDF documents
- Ask questions about the documents
- Get AI-powered answers with citations
""")

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Document Processing")
    
    # Document upload section
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                st.session_state.monitor.start_operation("document_processing")
                documents = process_document(uploaded_file)
                st.session_state.monitor.end_operation("document_processing")
                
                with st.spinner("Initializing Q&A system..."):
                    initialize_rag_system(documents, uploaded_file.name)
                    st.session_state.current_collection = st.session_state.collections[-1]
                
                st.success(f"Document processed successfully! Collection: {st.session_state.collections[-1]}")
    
    # Collection management section
    if st.session_state.collections:
        st.header("Document Collections")
        
        # Collection selector
        selected_collection = st.selectbox(
            "Select Document",
            st.session_state.collections,
            index=st.session_state.collections.index(st.session_state.current_collection) if st.session_state.current_collection else 0
        )
        
        col1, col2 = st.columns(2)
        
        # Switch button
        if col1.button("Switch to Document"):
            if selected_collection != st.session_state.current_collection:
                with st.spinner("Switching document..."):
                    switch_collection(selected_collection)
                st.success(f"Switched to {selected_collection}")
        
        # Delete button
        if col2.button("Delete Document", type="secondary"):
            if st.session_state.collections:
                with st.spinner("Deleting document..."):
                    delete_collection(selected_collection)
                st.success(f"Deleted {selected_collection}")
        
        # Show current collection
        if st.session_state.current_collection:
            st.info(f"Current Document: {st.session_state.current_collection}")

    # Clear all button
    if st.session_state.collections:
        if st.button("Clear All Documents", type="secondary"):
            if st.session_state.collections:
                with st.spinner("Clearing all documents..."):
                    for collection in st.session_state.collections.copy():
                        delete_collection(collection)
                st.success("All documents cleared")

    # Show processing metrics
    if 'monitor' in st.session_state:
        st.header("Processing Metrics")
        metrics = st.session_state.monitor.get_metrics()
        for operation, data in metrics.items():
            if 'duration' in data:
                st.metric(
                    label=operation.replace('_', ' ').title(),
                    value=f"{data['duration']:.2f}s"
                )

# Main chat interface
st.header("Ask Questions")

if not st.session_state.documents_processed or not st.session_state.current_collection:
    st.info("Please upload and process a document first, or select an existing document.")
else:
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent_executor.invoke(
                        {"input": prompt}
                    )
                    if isinstance(response, dict) and "output" in response:
                        answer = response["output"]
                    else:
                        answer = str(response)
                    
                    st.markdown(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and ❤️") 