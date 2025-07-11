import os
from PyPDF2 import PdfReader
from docx import Document
import markdown
from bs4 import BeautifulSoup
from typing import List, Dict, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    # In document_processor.py, update the __init__ method:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):  # Reduced chunk size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def load_document(self, file_path: str) -> Union[str, None]:
        """Load document based on file extension"""
        if not os.path.exists(file_path):
            return None
            
        if file_path.endswith('.pdf'):
            return self._load_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self._load_docx(file_path)
        elif file_path.endswith('.md'):
            return self._load_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def _load_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def _load_markdown(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            html = markdown.markdown(f.read())
            return BeautifulSoup(html, "html.parser").get_text()

    def process_document(self, file_path: str, metadata: Dict = {}) -> List[Dict]:
        """Process document into chunks with metadata"""
        text = self.load_document(file_path)
        if not text:
            return []
            
        chunks = self.text_splitter.create_documents([text])
        return [{
            "text": chunk.page_content,
            "metadata": {
                **metadata,
                "source": file_path,
                "chunk_id": f"{file_path}-{i}"
            }
        } for i, chunk in enumerate(chunks)]