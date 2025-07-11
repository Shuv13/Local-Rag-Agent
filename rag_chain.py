from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Optional
import torch
import logging

class RAGChain:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model_and_tokenizer()
        self.pipeline = self._create_pipeline()
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        self.prompt_template = self._create_prompt_template()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _initialize_model_and_tokenizer(self):
        """Initialize model with proper device handling"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with device_map="auto" but don't specify device here
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _create_pipeline(self):
        """Create pipeline without explicit device argument"""
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # Remove device argument since we're using device_map="auto"
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )

    def _create_prompt_template(self):
        """Create efficient prompt template"""
        return ChatPromptTemplate.from_template(
            """Answer the question based on the context. Be concise.
            Context: {context}
            Question: {question}
            Answer:"""
        )

    def _truncate_context(self, context_text: str, max_tokens: int = 1024) -> str:
        """Truncate context to fit model's window"""
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
            context_text = "\n".join([doc.get("text", "")[:500] for doc in context])
            context_text = self._truncate_context(context_text)
            
            if not context_text.strip():
                return "No usable context."
            
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "question": question,
                "context": context_text
            })
            
            return str(response).strip() if response else "No response generated."
            
        except Exception as e:
            self.logger.error(f"Generation error: {str(e)}")
            return "Error generating response."