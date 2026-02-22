"""
Chain of Responsibility Pattern implementation for query handlers.

Handlers process queries in sequence:
1. CourseIdHandler - Matches course IDs (e.g., "4361")
2. CourseTitleHandler - Matches course titles (e.g., "Data Structures")
3. SemanticHandler - Uses RAG for semantic queries (fallback)
"""

import re
from abc import ABC, abstractmethod
from typing import Optional, List

from models import Course


class QueryHandler(ABC):
    """Abstract base class for query handlers in the chain."""
    
    def __init__(self, successor=None, courses: List[Course] = None):
        self._successor = successor
        self._courses = courses or []
    
    def handle(self, query: str, content_store=None) -> str:
        """Process query or pass to successor."""
        result = self._process(query, content_store)
        if result:
            return result
        if self._successor:
            return self._successor.handle(query, content_store)
        return "Sorry, I couldn't find an answer to your question."
    
    @abstractmethod
    def _process(self, query: str, content_store) -> Optional[str]:
        """Process the query. Return None to pass to next handler."""
        pass


class CourseIdHandler(QueryHandler):
    """Handles queries containing course IDs (4-digit numbers 1000-4999)."""
    
    def _process(self, query: str, content_store) -> Optional[str]:
        match = re.search(r'\b([1-4]\d{3})\b', query)
        if match:
            course_id = match.group(1)
            for course in self._courses:
                if course.id == course_id:
                    return (
                        f"Course {course.id}: {course.title}. "
                        f"Description: {course.description}"
                    )
        return None


class CourseTitleHandler(QueryHandler):
    """Handles queries containing exact course titles."""
    
    def _process(self, query: str, content_store) -> Optional[str]:
        query_lower = query.lower()
        for course in self._courses:
            if course.title.lower() in query_lower:
                return (
                    f"Found course by title: {course.title}. "
                    f"ID: {course.id}. Description: {course.description}"
                )
        return None


class SemanticHandler(QueryHandler):
    """Handles semantic queries using RAG (retrieval + LLM generation)."""
    
    def __init__(self, successor=None, courses: List[Course] = None, 
                 strategy=None, provider=None):
        super().__init__(successor, courses)
        self.strategy = strategy
        self.provider = provider
    
    def _process(self, query: str, content_store) -> Optional[str]:
        if self.strategy and self.provider and content_store:
            context = self.strategy.retrieve(query, content_store)
            return self.provider.generate_answer(query, context)
        return None
