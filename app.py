import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
import os
from utils.file_handling import save_uploaded_file

# Initialize components
@st.cache_resource
def init_components():
    processor = DocumentProcessor()
    vector_store = VectorStore()
    # Using small fast model (phi-2 or gemma-2b recommended)
    rag_chain = RAGChain(model_name="microsoft/phi-2")  # 2.7B params, fast
    return processor, vector_store, rag_chain

def main():
    st.title("Local RAG Agent")
    st.write("Private document analysis with Hugging Face models and ChromaDB")
    
    # Initialize components
    processor, vector_store, rag_chain = init_components()
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                documents = processor.process_document(file_path)
                vector_store.add_documents(documents)
                st.success(f"Processed {uploaded_file.name}")
    
    # Main chat interface
    st.header("Chat with your documents")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Search for relevant documents
                    context = vector_store.search(prompt, n_results=3)
                    
                    # Generate response
                    response = rag_chain.generate_response(prompt, context)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()