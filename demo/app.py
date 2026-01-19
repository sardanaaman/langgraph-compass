"""
Compass Demo - Hugging Face Spaces

A simple interactive demo showing Compass follow-up question generation.

To deploy:
1. Create a new Space on huggingface.co (Gradio SDK)
2. Copy this file and requirements.txt to the Space repo
3. Add your OPENAI_API_KEY as a Space secret
"""

import os

import gradio as gr
from langchain_openai import ChatOpenAI

from compass import CompassNode, DefaultTriggerPolicy

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize Compass with different strategies
STRATEGIES = ["adaptive", "clarifying", "exploratory", "deepening"]


def generate_response_and_followups(
    user_query: str,
    strategy: str,
    temperature: float,
) -> tuple[str, str, str]:
    """Generate an agent response and Compass follow-up suggestions."""
    if not user_query.strip():
        return "", "", "Please enter a question."

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    # Generate a simple response (simulating an agent)
    response = llm.invoke(f"Answer this question concisely: {user_query}")
    response_text = response.content

    # Create Compass node with selected strategy
    compass = CompassNode(
        model=llm,
        strategy=strategy,
        trigger=DefaultTriggerPolicy(min_response_length=20),
        max_suggestions=3,
        generate_candidates=5,
    )

    # Build state as if from a LangGraph
    state = {
        "query": user_query,
        "response": response_text,
        "messages": [],
    }

    # Generate follow-ups
    result = compass(state, config={})
    suggestions = result.get("compass_suggestions", [])

    # Format suggestions
    if suggestions:
        formatted = "\n".join(f"• {s}" for s in suggestions)
    else:
        formatted = "No follow-up suggestions generated."

    return response_text, formatted, ""


def compare_strategies(user_query: str, temperature: float) -> dict:
    """Compare all strategies side-by-side."""
    if not user_query.strip():
        return dict.fromkeys(STRATEGIES, "Please enter a question.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    # Generate response once
    response = llm.invoke(f"Answer this question concisely: {user_query}")
    response_text = response.content

    results = {"response": response_text}

    for strategy in STRATEGIES:
        compass = CompassNode(
            model=llm,
            strategy=strategy,
            trigger=DefaultTriggerPolicy(min_response_length=20),
            max_suggestions=2,
            generate_candidates=4,
        )

        state = {
            "query": user_query,
            "response": response_text,
            "messages": [],
        }

        result = compass(state, config={})
        suggestions = result.get("compass_suggestions", [])

        if suggestions:
            results[strategy] = "\n".join(f"• {s}" for s in suggestions)
        else:
            results[strategy] = "No suggestions"

    return results


HEADER_MD = """# 🧭 Compass Demo

**Intelligent follow-up question generation for LangGraph agents.**

Compass transforms AI agents from reactive responders into proactive
conversational partners. Try it out below!

[GitHub](https://github.com/sardanaaman/langgraph-compass) |
[PyPI](https://pypi.org/project/langgraph-compass/)
"""

COMPARE_MD = """See how different strategies generate different follow-ups for the same query.

| Strategy | Best For |
|----------|----------|
| **Adaptive** | Auto-selects based on context |
| **Clarifying** | Resolving ambiguities |
| **Exploratory** | Opening new directions |
| **Deepening** | Adding more detail |
"""

FOOTER_MD = """---

### How It Works

1. **You ask a question** → A simulated agent responds
2. **Compass analyzes** the query and response
3. **Follow-ups generated** using your chosen strategy
4. **Novelty filtering** ensures suggestions aren't repetitive

In a real LangGraph agent, Compass runs as a node in your graph,
optionally in parallel for zero latency impact.

---

*Built and battle-tested at Cisco*
"""

# Build Gradio interface
with gr.Blocks(
    title="Compass Demo",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(HEADER_MD)

    with gr.Tab("Try It"):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., How do I deploy a Python app to AWS?",
                    lines=2,
                )
                strategy_select = gr.Dropdown(
                    choices=STRATEGIES,
                    value="adaptive",
                    label="Strategy",
                    info="How should Compass generate follow-ups?",
                )
                temp_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                )
                submit_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                response_output = gr.Textbox(
                    label="Agent Response",
                    lines=4,
                    interactive=False,
                )
                followups_output = gr.Textbox(
                    label="Compass Suggestions",
                    lines=4,
                    interactive=False,
                )
                error_output = gr.Textbox(
                    label="",
                    visible=False,
                )

        submit_btn.click(
            fn=generate_response_and_followups,
            inputs=[query_input, strategy_select, temp_slider],
            outputs=[response_output, followups_output, error_output],
        )

    with gr.Tab("Compare Strategies"):
        gr.Markdown(COMPARE_MD)

        compare_query = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What's the best way to learn machine learning?",
            lines=2,
        )
        compare_temp = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            label="Temperature",
        )
        compare_btn = gr.Button("Compare All Strategies", variant="primary")

        with gr.Row():
            response_box = gr.Textbox(label="Agent Response", lines=3, interactive=False)

        with gr.Row():
            adaptive_box = gr.Textbox(label="Adaptive", lines=3, interactive=False)
            clarifying_box = gr.Textbox(label="Clarifying", lines=3, interactive=False)

        with gr.Row():
            exploratory_box = gr.Textbox(label="Exploratory", lines=3, interactive=False)
            deepening_box = gr.Textbox(label="Deepening", lines=3, interactive=False)

        def run_comparison(query, temp):
            results = compare_strategies(query, temp)
            return (
                results.get("response", ""),
                results.get("adaptive", ""),
                results.get("clarifying", ""),
                results.get("exploratory", ""),
                results.get("deepening", ""),
            )

        compare_btn.click(
            fn=run_comparison,
            inputs=[compare_query, compare_temp],
            outputs=[response_box, adaptive_box, clarifying_box, exploratory_box, deepening_box],
        )

    gr.Markdown(FOOTER_MD)

if __name__ == "__main__":
    demo.launch()
