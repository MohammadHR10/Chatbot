"""Course data model for the CSE Chatbot."""

from dataclasses import dataclass


@dataclass
class Course:
    """Represents a course in the CSE catalog."""
    
    id: str
    title: str
    description: str
