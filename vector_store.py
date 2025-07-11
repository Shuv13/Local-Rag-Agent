import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import logging
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, collection_name: str = "documents", persist_dir: str = "chroma-data"):
        """
        Initialize the vector store with ChromaDB backend and SentenceTransformer embeddings.
        
        Args:
            collection_name: Name of the collection to use
            persist_dir: Directory to store the ChromaDB data
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Try to get existing collection first
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.logger.info(f"Loaded existing collection '{self.collection_name}'")
            except ValueError:
                # Create new collection if it doesn't exist
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                self.logger.info(f"Created new collection '{self.collection_name}'")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize VectorStore: {str(e)}")
            raise

    def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text or list of texts.
        
        Args:
            text: Input text or list of texts to embed
            
        Returns:
            List of embeddings (single list for single text, list of lists for multiple texts)
        """
        if isinstance(text, str):
            return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
        """
        if not documents:
            self.logger.warning("No documents provided to add")
            return
            
        try:
            # Prepare document data
            ids = [str(doc["metadata"].get("chunk_id", idx)) for idx, doc in enumerate(documents)]
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            embeddings = self.embed_text(texts)
            
            # Add to collection
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            self.logger.info(f"Added {len(ids)} documents to collection")
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise

    def search(self, query: str, n_results: int = 5, filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of result dictionaries with 'text', 'metadata', and 'score'
        """
        try:
            query_embedding = self.embed_text(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                try:
                    formatted_results.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    })
                except (IndexError, KeyError) as e:
                    self.logger.warning(f"Error formatting result {i}: {str(e)}")
                    continue
                    
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []

    def delete_documents(self, filter: Dict):
        """
        Delete documents matching the filter criteria.
        
        Args:
            filter: Dictionary of metadata fields to match for deletion
        """
        try:
            self.collection.delete(where=filter)
            self.logger.info(f"Deleted documents matching filter: {filter}")
        except Exception as e:
            self.logger.error(f"Error deleting documents: {str(e)}")
            raise

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.collection.delete()
            self.logger.info("Collection cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict:
        """
        Get basic statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            return {
                "count": self.collection.count(),
                "name": self.collection.name,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {}