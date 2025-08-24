import streamlit as st
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
import os
import logging
from utils.file_handling import save_uploaded_file

# --- Constants and Configuration ---
MODEL_NAME = "microsoft/phi-2"  # 2.7B params, fast

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Component Initialization ---
@st.cache_resource
def init_components():
    """Initialize all main components of the RAG application."""
    try:
        processor = DocumentProcessor()
        vector_store = VectorStore()
        rag_chain = RAGChain(model_name=MODEL_NAME)
        logger.info("Components initialized successfully.")
        return processor, vector_store, rag_chain
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        st.error(f"Application failed to initialize. Please check logs. Error: {e}")
        return None, None, None

# --- Main Application Logic ---
def main():
    st.title("Local RAG Agent")
    st.write("Private document analysis with Hugging Face models and ChromaDB")
    
    processor, vector_store, rag_chain = init_components()
    
    if not all([processor, vector_store, rag_chain]):
        st.warning("Components could not be loaded. The application is not operational.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Sidebar for Document and Database Management ---
    with st.sidebar:
        st.header("Document Management")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    try:
                        file_path = save_uploaded_file(uploaded_file)
                        if file_path:
                            documents = processor.process_document(file_path)
                            vector_store.add_documents(documents)
                            st.success(f"Processed and indexed {uploaded_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to process {uploaded_file.name}: {e}", exc_info=True)
                        st.error(f"Could not process {uploaded_file.name}.")

        st.header("Database Management")
        if st.button("Clear Document Database"):
            try:
                vector_store.clear_collection()
                st.session_state.messages = []  # Clear chat history
                st.success("All documents and chat history have been cleared.")
                st.rerun()
            except Exception as e:
                logger.error(f"Failed to clear database: {e}", exc_info=True)
                st.error("Failed to clear database.")

    # --- Main Chat Interface ---
    st.header("Chat with your documents")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # --- Chat Input and Response Generation ---
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if the database has any documents
                    if vector_store.get_collection_stats().get("count", 0) == 0:
                        st.warning("The document database is empty. Please upload documents first.")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "The document database is empty. Please upload documents first."
                        })
                        return

                    context = vector_store.search(prompt, n_results=3)
                    response = rag_chain.generate_response(prompt, context)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    logger.error(f"Error generating response for prompt '{prompt}': {e}", exc_info=True)
                    error_msg = "Sorry, I encountered an error while generating a response. Please check the logs for details."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()