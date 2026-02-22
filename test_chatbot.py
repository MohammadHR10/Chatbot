"""
Automated tests for the CSE Course Chatbot.

Tests the three design patterns:
1. Chain of Responsibility - Query routing
2. Strategy Pattern - Retrieval strategy swapping
3. Adapter Pattern - LLM provider swapping
"""

from chatbot import CSEChatbot
from strategies import TopNStrategy, WindowStrategy
from providers import OllamaProvider, OpenAIProvider


def test_chain_of_responsibility():
    """Test that queries are routed to the correct handler."""
    print("Testing Chain of Responsibility...")
    
    bot = CSEChatbot("courses.jsonl")
    
    response1 = bot.ask("Tell me about course 4361")
    assert "Software Design Patterns" in response1
    print("  [PASS] CourseIdHandler handled '4361'")
    
    response2 = bot.ask("What is Data Structures?")
    assert "2320" in response2
    print("  [PASS] CourseTitleHandler handled 'Data Structures'")
    
    response3 = bot.ask("Which course covers design patterns?")
    assert "[Ollama]" in response3
    print("  [PASS] SemanticHandler handled semantic query")


def test_strategy_pattern():
    """Test that retrieval strategies can be swapped at runtime."""
    print("\nTesting Strategy Pattern...")
    
    bot = CSEChatbot("courses.jsonl")
    
    assert isinstance(bot.strategy, TopNStrategy)
    print("  [PASS] Default strategy is TopNStrategy")
    
    bot.set_strategy("window")
    assert isinstance(bot.strategy, WindowStrategy)
    assert isinstance(bot._semantic_handler.strategy, WindowStrategy)
    print("  [PASS] Strategy swapped to WindowStrategy")


def test_adapter_pattern():
    """Test that LLM providers can be swapped at runtime."""
    print("\nTesting Adapter Pattern...")
    
    bot = CSEChatbot("courses.jsonl")
    
    assert isinstance(bot.provider, OllamaProvider)
    print("  [PASS] Default provider is OllamaProvider")
    
    bot.set_provider("openai")
    assert isinstance(bot.provider, OpenAIProvider)
    assert isinstance(bot._semantic_handler.provider, OpenAIProvider)
    print("  [PASS] Provider swapped to OpenAIProvider")


if __name__ == "__main__":
    try:
        test_chain_of_responsibility()
        test_strategy_pattern()
        test_adapter_pattern()
        print("\n" + "=" * 40)
        print("ALL TESTS PASSED!")
        print("=" * 40)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nERROR: {e}")
