# CSE Course Information Chatbot

A RAG-based chatbot for answering questions about CSE undergraduate courses.

## Design Patterns Implemented

1. **Strategy Pattern** - Swappable retrieval strategies (TopN, Window, Document, Hierarchical)
2. **Chain of Responsibility** - Query routing (Course ID → Title → Semantic)
3. **Adapter Pattern** - Pluggable LLM providers (Ollama, OpenAI, Gemini)

## Requirements

- Python 3.10+
- Ollama with `phi3` and `nomic-embed-text` models

## Setup

```bash
# Install Ollama and pull models
ollama pull phi3
ollama pull nomic-embed-text

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Tests
```bash
python test_chatbot.py
```

### Interactive Chatbot
```bash
python chatbot.py
```

### Runtime Commands
```
set strategy top_n       # Switch to TopN retrieval
set strategy window      # Switch to Window retrieval
set provider ollama      # Use Ollama LLM
set provider openai      # Use OpenAI adapter
exit                     # Quit
```

## Project Structure

```
cse_course_chatbot/
├── chatbot.py        # Main controller
├── handlers.py       # Chain of Responsibility
├── strategies.py     # Strategy Pattern
├── providers.py      # Adapter Pattern
├── models.py         # Data model
├── courses.jsonl     # Course data
├── test_chatbot.py   # Automated tests
└── requirements.txt  # Dependencies
```
