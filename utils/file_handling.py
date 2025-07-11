import os
from typing import Optional
import streamlit as st

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to data directory and return path"""
    try:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None