"""
CSE Course Chatbot - Main Controller

A RAG-based chatbot for answering questions about CSE courses.
Implements Strategy, Chain of Responsibility, and Adapter patterns.
"""

import json
from typing import List

from models import Course
from strategies import (
    TopNStrategy, WindowStrategy, DocumentStrategy, HierarchicalStrategy
)
from handlers import CourseIdHandler, CourseTitleHandler, SemanticHandler
from providers import OllamaProvider, OpenAIProvider, GeminiProvider

from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_embedder import OllamaEmbedder
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.rag.model.chunk import Chunk


class CSEChatbot:
    """Main chatbot controller orchestrating all components."""
    
    def __init__(self, data_path: str = "courses.jsonl"):
        self._courses = self._load_courses(data_path)
        
        self._init_rag_components()
        self._index_courses()
        
        self.strategy = TopNStrategy()
        self.provider = OllamaProvider()
        
        self._build_handler_chain()
    
    def _load_courses(self, path: str) -> List[Course]:
        """Load course data from JSONL file."""
        courses = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    courses.append(Course(
                        id=data['id'],
                        title=data['title'],
                        description=data['description']
                    ))
        except FileNotFoundError:
            print(f"Warning: {path} not found.")
        return courses
    
    def _init_rag_components(self):
        """Initialize RAG components (embedder, content store)."""
        print("Initializing RAG components...")
        self._ollama = AccessOllama(host="localhost", port=11434)
        self._embedder = OllamaEmbedder(
            access_ollama=self._ollama, 
            model="nomic-embed-text"
        )
        self._store = InternalContentStore(embedder=self._embedder)
    
    def _index_courses(self):
        """Index course documents for semantic search."""
        print("Indexing courses...")
        chunks = []
        for course in self._courses:
            chunk = Chunk(
                document_id=course.id,
                chunk_id="0",
                total_chunks=1,
                chunk_text=f"{course.title}: {course.description}",
                properties={"title": course.title, "id": course.id}
            )
            chunks.append(chunk)
        self._store.store(chunks)
        print("Indexing complete.")
    
    def _build_handler_chain(self):
        """Build the Chain of Responsibility."""
        self._semantic_handler = SemanticHandler(
            courses=self._courses,
            strategy=self.strategy,
            provider=self.provider
        )
        self._title_handler = CourseTitleHandler(
            successor=self._semantic_handler,
            courses=self._courses
        )
        self._handler_chain = CourseIdHandler(
            successor=self._title_handler,
            courses=self._courses
        )
    
    def set_strategy(self, strategy_name: str):
        """Change retrieval strategy at runtime (Strategy Pattern)."""
        strategies = {
            "top_n": TopNStrategy,
            "window": WindowStrategy,
            "document": DocumentStrategy,
            "hierarchical": HierarchicalStrategy
        }
        if strategy_name in strategies:
            self.strategy = strategies[strategy_name]()
            self._semantic_handler.strategy = self.strategy
            print(f"Strategy changed to: {strategy_name}")
        else:
            print(f"Unknown strategy: {strategy_name}")
    
    def set_provider(self, provider_name: str):
        """Change LLM provider at runtime (Adapter Pattern)."""
        providers = {
            "ollama": OllamaProvider,
            "openai": OpenAIProvider,
            "gemini": GeminiProvider
        }
        if provider_name in providers:
            self.provider = providers[provider_name]()
            self._semantic_handler.provider = self.provider
            print(f"Provider changed to: {provider_name}")
        else:
            print(f"Unknown provider: {provider_name}")
    
    def ask(self, question: str) -> str:
        """Process a user question through the handler chain."""
        return self._handler_chain.handle(question, self._store)


def main():
    """Run the interactive chatbot."""
    chatbot = CSEChatbot("courses.jsonl")
    
    print("\nCSE Course Chatbot")
    print("Commands: 'exit', 'set strategy <name>', 'set provider <name>'")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            if user_input.lower().startswith("set strategy "):
                strategy = user_input.split()[-1]
                chatbot.set_strategy(strategy)
                continue
            
            if user_input.lower().startswith("set provider "):
                provider = user_input.split()[-1]
                chatbot.set_provider(provider)
                continue
            
            response = chatbot.ask(user_input)
            print(f"Bot: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
