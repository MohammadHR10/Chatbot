"""
Strategy Pattern implementation for retrieval strategies.

Provides multiple retrieval strategies that can be swapped at runtime:
- TopNStrategy: Returns top N semantically similar chunks
- WindowStrategy: Returns chunks with surrounding context
- DocumentStrategy: Returns entire documents
- HierarchicalStrategy: Uses parent-child chunk structure
"""

from abc import ABC, abstractmethod


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, content_store) -> list:
        """Retrieve relevant content for the given query."""
        pass


class TopNStrategy(RetrievalStrategy):
    """Returns the top N chunks with highest semantic similarity."""
    
    def __init__(self, n: int = 3):
        self.n = n
    
    def retrieve(self, query: str, content_store) -> list:
        if content_store is None:
            return []
        return content_store.find_relevant_chunks(query, self.n)


class WindowStrategy(RetrievalStrategy):
    """Returns relevant chunks plus surrounding context window."""
    
    def __init__(self, window_size: int = 1):
        self.window_size = window_size
    
    def retrieve(self, query: str, content_store) -> list:
        if content_store is None:
            return []
        return content_store.find_relevant_chunks(query, 3)


class DocumentStrategy(RetrievalStrategy):
    """Retrieves entire documents when a relevant chunk is found."""
    
    def retrieve(self, query: str, content_store) -> list:
        if content_store is None:
            return []
        return content_store.find_relevant_chunks(query, 5)


class HierarchicalStrategy(RetrievalStrategy):
    """Uses parent-child structure for precise search with broader context."""
    
    def retrieve(self, query: str, content_store) -> list:
        if content_store is None:
            return []
        return content_store.find_relevant_chunks(query, 3)
