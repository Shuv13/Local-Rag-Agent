from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Optional
import torch
import logging

class RAGChain:
    def __init__(self, model_name: str = "google/gemma-3-1b-it"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model_and_tokenizer()
        self.pipeline = self._create_pipeline()
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        self.prompt_template = self._create_prompt_template()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_model_and_tokenizer(self):
        """Initialize model with proper device handling and optimized dtype."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Set torch_dtype for performance; bfloat16 is good for CPUs
            dtype = torch.float16 if self.device == "cuda" else torch.bfloat16

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _create_pipeline(self):
        """Create an optimized pipeline for faster inference."""
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,  # Increased for potentially longer answers
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )

    def _create_prompt_template(self):
        """Create efficient prompt template."""
        return ChatPromptTemplate.from_template(
            """Answer the question based on the context. Be concise.
            Context: {context}
            Question: {question}
            Answer:"""
        )

    def _truncate_context(self, context_text: str, max_tokens: int = 8192) -> str:
        """
        Truncate context to fit the model's window.
        Gemma has a large context window, so we can use a larger value.
        """
        try:
            inputs = self.tokenizer(
                context_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_tokens
            )
            return self.tokenizer.decode(
                inputs["input_ids"][0], 
                skip_special_tokens=True
            )
        except Exception as e:
            self.logger.error(f"Error truncating context: {str(e)}")
            return ""

    def generate_response(self, question: str, context: List[Dict]) -> str:
        """Generate RAG response"""
        if not context:
            return "No relevant context found."
            
        try:
            # Join document texts without premature truncation
            context_text = "\n\n".join([doc.get("text", "") for doc in context])
            truncated_context = self._truncate_context(context_text)
            
            if not truncated_context.strip():
                return "No usable context."
            
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "question": question,
                "context": truncated_context
            })
            
            return str(response).strip() if response else "No response generated."
            
        except Exception as e:
            self.logger.error(f"Generation error: {str(e)}")
            return "Error generating response."