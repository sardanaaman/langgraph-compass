"""Quickstart example: Add intelligent follow-ups to any LangGraph agent.

This example shows the minimal setup to get Compass working with your agent.
Run with: python examples/quickstart.py
"""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from compass import CompassNode


def simple_agent(state: MessagesState) -> dict:
    """A simple agent that echoes back a response."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # In a real agent, you'd call an LLM here
    response = f"I received your message about: {last_message}. Here's some helpful information about that topic that spans multiple sentences to ensure we have enough content for Compass to work with effectively."

    return {"messages": [AIMessage(content=response)]}


def main():
    # 1. Create the Compass node
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    compass = CompassNode(
        model=model,
        strategy="adaptive",
        max_suggestions=2,
    )

    # 2. Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("agent", simple_agent)
    builder.add_node("compass", compass)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "compass")
    builder.add_edge("compass", END)

    graph = builder.compile()

    # 3. Run it
    result = graph.invoke({"messages": [HumanMessage(content="What is machine learning?")]})

    # 4. Access suggestions
    print("Agent Response:")
    print(result["messages"][-1].content)
    print("\nCompass Suggestions:")
    for suggestion in result.get("compass_suggestions", []):
        print(f"  - {suggestion}")


if __name__ == "__main__":
    main()
