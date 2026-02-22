"""
Adapter Pattern implementation for LLM providers.

Provides a unified interface for different LLM backends:
- OllamaProvider: Local Ollama server
- OpenAIProvider: OpenAI API (simulated)
- GeminiProvider: Google Gemini API (simulated)
"""

from abc import ABC, abstractmethod

from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_answer_generator import OllamaAnswerGenerator


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_answer(self, question: str, context) -> str:
        """Generate an answer given a question and context."""
        pass


class OllamaProvider(LLMProvider):
    """Adapter for local Ollama LLM."""
    
    def __init__(self, model: str = "phi3", host: str = "localhost", port: int = 11434):
        self.access = AccessOllama(host=host, port=port)
        self.generator = OllamaAnswerGenerator(access_ollama=self.access, model=model)
    
    def generate_answer(self, question: str, context) -> str:
        if isinstance(context, list):
            context_text = "\n".join([chunk.chunk_text for chunk in context])
        else:
            context_text = str(context)
        
        answer = self.generator.generate_answer(question, context_text)
        return f"[Ollama] {answer}"


class OpenAIProvider(LLMProvider):
    """Adapter for OpenAI API (simulated for demonstration)."""
    
    def generate_answer(self, question: str, context) -> str:
        context_preview = str(context)[:100] if context else "No context"
        return f"[OpenAI] Answer for '{question}' based on: {context_preview}..."


class GeminiProvider(LLMProvider):
    """Adapter for Google Gemini API (simulated for demonstration)."""
    
    def generate_answer(self, question: str, context) -> str:
        context_preview = str(context)[:100] if context else "No context"
        return f"[Gemini] Answer for '{question}' based on: {context_preview}..."
