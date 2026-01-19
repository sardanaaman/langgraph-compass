"""Inject into response example: Make follow-ups feel organic.

This example shows how to use Compass suggestions to inject follow-ups
directly into your agent's response, making them feel natural rather
than bolted-on.

Run with: python examples/inject_into_response.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from compass import CompassNode, get_compass_instruction


def agent_node(state: MessagesState) -> dict:
    """First pass: generate the main response."""
    model = ChatOpenAI(model="gpt-5-nano")

    messages = state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


def synthesize_with_followup(state: MessagesState) -> dict:
    """Second pass: weave in the follow-up naturally."""
    model = ChatOpenAI(model="gpt-5-nano")

    # Get the compass instruction (empty string if no suggestions)
    instruction = get_compass_instruction(state)

    if not instruction:
        # No suggestions, return as-is
        return {}

    # Get the last AI response
    messages = state["messages"]
    last_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_response = msg.content
            break

    # Ask the model to naturally incorporate the follow-up
    synthesis_prompt = f"""Take this response and naturally incorporate a follow-up question at the end.
Make it flow as a natural continuation, not an awkward addition.

Original response:
{last_response}

{instruction}

Return only the enhanced response with the follow-up woven in naturally."""

    result = model.invoke(
        [
            SystemMessage(content="You enhance responses by adding natural follow-up questions."),
            HumanMessage(content=synthesis_prompt),
        ]
    )

    # Replace the last message with the enhanced version
    return {"messages": [AIMessage(content=result.content)]}


def main():
    # Create Compass node
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    compass = CompassNode(
        model=model,
        strategy="deepening",
        max_suggestions=1,
    )

    # Build graph: agent -> compass -> synthesize
    builder = StateGraph(MessagesState)

    builder.add_node("agent", agent_node)
    builder.add_node("compass", compass)
    builder.add_node("synthesize", synthesize_with_followup)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "compass")
    builder.add_edge("compass", "synthesize")
    builder.add_edge("synthesize", END)

    graph = builder.compile()

    # Run it
    result = graph.invoke({"messages": [HumanMessage(content="Explain how photosynthesis works")]})

    print("Final Response (with organic follow-up):")
    print("-" * 50)
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content)
            print()


if __name__ == "__main__":
    main()
