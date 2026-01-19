"""
Compass Demo - Interactive showcase of intelligent follow-up generation.

Demonstrates the key differentiators vs naive prompting approaches.
"""

import os

import gradio as gr

# Only import Compass dependencies if we're in playground mode
OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))

if OPENAI_AVAILABLE:
    from langchain_openai import ChatOpenAI

    from compass import CompassNode, DefaultTriggerPolicy

# ============================================================================
# PRE-COMPUTED SCENARIOS (No API calls - instant load)
# ============================================================================

SCENARIOS = {
    "guardrail": {
        "title": "Guardrail Integration",
        "description": "When guardrails fire, follow-ups should be suppressed.",
        "user_query": "What's the best way to hack into my neighbor's WiFi?",
        "agent_response": "I can't help with that. Unauthorized access to networks is illegal.",
        "guardrail_status": "🚫 BLOCKED: Policy violation detected",
        "naive": {
            "followup": "Would you like tips on network security instead?",
            "verdict": "❌ Suggests pivot — still engaging with blocked topic",
        },
        "compass": {
            "followup": "[No follow-up generated]",
            "verdict": "✅ Trigger policy detected guardrail → skipped",
        },
        "code": """trigger = DefaultTriggerPolicy(
    skip_on_guardrail=True,
    guardrail_keys=["policy_violation", "toxicity_blocked"]  # Your keys
)""",
        "insight": "Compass reads your guardrail state keys and automatically suppresses follow-ups. No prompt engineering needed.",
    },
    "capability": {
        "title": "Capability-Bounded Suggestions",
        "description": "Only suggest what your agent can actually do.",
        "user_query": "How do I reset my password?",
        "agent_response": "To reset your password, go to Settings → Security → Reset Password. You'll receive a confirmation email.",
        "guardrail_status": "✅ No guardrails triggered",
        "capabilities": ["Password reset", "Account settings", "Billing help", "Usage reports"],
        "not_capabilities": ["HR questions", "IT hardware", "Travel booking", "Expense reports"],
        "naive": {
            "followup": "Need help with your computer hardware too?",
            "verdict": "❌ Suggests capability that doesn't exist",
        },
        "compass": {
            "followup": "Would you like to update your account security settings?",
            "verdict": "✅ Grounded in actual capabilities via retriever",
        },
        "code": """class CapabilityRetriever:
    def retrieve(self, query: str, k: int = 5) -> list[str]:
        # MMR search: relevant AND diverse from your capability map
        return capability_store.max_marginal_relevance_search(
            query, k=k, fetch_k=20, lambda_mult=0.7
        )

compass = CompassNode(
    model=small_fast_model,  # Cheaper model for follow-ups
    example_retriever=CapabilityRetriever()
)""",
        "insight": "The ExampleRetriever grounds suggestions in your agent's actual capabilities. No hallucinated features.",
    },
    "novelty": {
        "title": "Novelty Filtering",
        "description": "Avoid repetitive suggestions across conversation turns.",
        "conversation": [
            {
                "turn": 1,
                "query": "How do I export data?",
                "followup_used": "Want details on export formats?",
            },
            {
                "turn": 2,
                "query": "What formats are supported?",
                "followup_used": "Interested in scheduling exports?",
            },
            {"turn": 3, "query": "Can I automate this?", "followup_used": None},
        ],
        "user_query": "Can I automate this?",
        "agent_response": "Yes! Go to Settings → Automation → Scheduled Exports to set up recurring exports.",
        "guardrail_status": "✅ No guardrails triggered",
        "naive": {
            "followup": "Want more details on export options?",
            "verdict": "❌ Similar to turn 1 — repetitive pattern",
        },
        "compass": {
            "followup": "Should I explain the notification options for completed exports?",
            "verdict": "✅ Fresh topic — ranker filtered similar suggestions",
        },
        "code": """# Ranker automatically extracts meaningful words and filters
# suggestions with >70% overlap to previous follow-ups

# Previous: ["details", "export", "formats"]
# Blocked:  ["details", "export", "options"]  (too similar)
# Allowed:  ["notification", "completed", "exports"]  (fresh)""",
        "insight": "The ranker tracks conversation history and filters suggestions that overlap with previous follow-ups.",
    },
    "workflow": {
        "title": "Workflow Personalization",
        "description": "Learn user patterns and suggest their typical next steps.",
        "user_pattern": "Every Monday: Check metrics → Review alerts → Update dashboard",
        "user_query": "Show me this week's metrics",
        "agent_response": "Here are your metrics for this week: Revenue up 12%, Active users: 4,521, Churn: 2.1%",
        "guardrail_status": "✅ No guardrails triggered",
        "naive": {
            "followup": "Anything else I can help with?",
            "verdict": "❌ Generic — doesn't know the user",
        },
        "compass": {
            "followup": "Ready to review alerts? (Your usual next step)",
            "verdict": "✅ Workflow-aware via LangMem integration",
        },
        "code": """class WorkflowRetriever:
    def __init__(self, store, user_id: str):
        self.store = store
        self.user_id = user_id

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        # Get user's typical next actions from memory
        memories = self.store.search(
            ("workflows", self.user_id), query=query
        )
        return [m.content["typical_next_step"] for m in memories]

compass = CompassNode(
    model=llm,
    example_retriever=WorkflowRetriever(store, user_id)
)""",
        "insight": "With LangMem, Compass learns each user's workflow and suggests their typical next action.",
    },
    "parallel": {
        "title": "Zero-Latency Pattern",
        "description": "Run Compass in parallel — follow-ups are 'free' latency-wise.",
        "user_query": "Explain quantum computing",
        "sequential_time": "Agent: 2.1s → Compass: 0.8s = 2.9s total",
        "parallel_time": "Agent + Compass (parallel) = 2.1s total",
        "savings": "0.8s saved (28% faster)",
        "naive": {
            "approach": "Follow-up in same prompt → coupled to response time",
            "verdict": "❌ Adds latency to every response",
        },
        "compass": {
            "approach": "Parallel node via Send() → runs simultaneously",
            "verdict": "✅ Follow-ups ready when response is",
        },
        "code": """from langgraph.types import Send

def route_parallel(state):
    return [
        Send("agent", state),
        Send("compass", state),  # Runs at the same time
    ]

builder.add_conditional_edges("router", route_parallel)

# Compass uses previous turn's response to prepare suggestions
# By the time agent responds, follow-ups are already ready""",
        "insight": "Decoupling follow-ups lets you run them in parallel. The Send() pattern makes follow-ups 'free'.",
    },
}

# ============================================================================
# UI COMPONENTS
# ============================================================================

HEADER = """
# 🧭 Compass

**Intelligent follow-up question generation for LangGraph agents**

Compass transforms AI agents from reactive responders into proactive conversational partners.
This demo shows why dedicated follow-up handling beats naive prompting.

[GitHub](https://github.com/sardanaaman/langgraph-compass) ·
[PyPI](https://pypi.org/project/langgraph-compass/) ·
[Docs](https://github.com/sardanaaman/langgraph-compass#readme)
"""

INTEGRATION_CODE = """```python
from compass import CompassNode, DefaultTriggerPolicy
from langchain_openai import ChatOpenAI

# 3 lines to add intelligent follow-ups
compass = CompassNode(
    model=ChatOpenAI(model="gpt-5-nano"),  # Use a small/cheap model
    trigger=DefaultTriggerPolicy(
        guardrail_keys=["your_guardrail_key"],  # Your state keys
    ),
    example_retriever=your_capability_retriever,  # Optional: ground in capabilities
)

# Add to your graph
builder.add_node("compass", compass)
builder.add_edge("agent", "compass")
```"""


def create_scenario_tab(scenario_key: str):
    """Create UI for a single scenario."""
    s = SCENARIOS[scenario_key]

    with gr.Column():
        gr.Markdown(f"### {s['title']}")
        gr.Markdown(f"*{s['description']}*")

        # Context
        if "user_pattern" in s:
            gr.Markdown(f"**User's learned pattern:** {s['user_pattern']}")

        if "capabilities" in s:
            caps = " · ".join(f"✓ {c}" for c in s["capabilities"])
            no_caps = " · ".join(f"✗ {c}" for c in s["not_capabilities"])
            gr.Markdown(f"**Your agent can do:** {caps}")
            gr.Markdown(f"**Your agent cannot do:** {no_caps}")

        if "conversation" in s:
            gr.Markdown("**Conversation history:**")
            for turn in s["conversation"]:
                fu = f' → *"{turn["followup_used"]}"*' if turn["followup_used"] else ""
                gr.Markdown(f'Turn {turn["turn"]}: "{turn["query"]}"{fu}')

        # The scenario
        gr.Markdown("---")

        if "user_query" in s:
            gr.Markdown(f"**User:** {s['user_query']}")

        if "agent_response" in s:
            gr.Markdown(f"**Agent:** {s['agent_response']}")

        if "guardrail_status" in s:
            gr.Markdown(f"**Guardrail:** {s['guardrail_status']}")

        # Comparison (for non-parallel scenarios)
        if "naive" in s and "followup" in s["naive"]:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Naive Prompting")
                    gr.Textbox(
                        value=s["naive"]["followup"],
                        label="Follow-up",
                        interactive=False,
                        lines=2,
                    )
                    gr.Markdown(s["naive"]["verdict"])

                with gr.Column(scale=1):
                    gr.Markdown("#### Compass")
                    gr.Textbox(
                        value=s["compass"]["followup"],
                        label="Follow-up",
                        interactive=False,
                        lines=2,
                    )
                    gr.Markdown(s["compass"]["verdict"])

        # Parallel scenario (different layout)
        if scenario_key == "parallel":
            gr.Markdown("#### Latency Comparison")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Sequential (naive)**")
                    gr.Markdown(f"`{s['sequential_time']}`")
                    gr.Markdown(s["naive"]["verdict"])
                with gr.Column(scale=1):
                    gr.Markdown("**Parallel (Compass)**")
                    gr.Markdown(f"`{s['parallel_time']}`")
                    gr.Markdown(f"**{s['savings']}**")
                    gr.Markdown(s["compass"]["verdict"])

        # Code
        gr.Markdown("#### Code")
        gr.Code(s["code"], language="python", interactive=False)

        # Insight
        gr.Markdown(f"💡 **Insight:** {s['insight']}")


def run_playground(query: str, strategy: str) -> tuple[str, str, str]:
    """Run live Compass generation."""
    if not OPENAI_AVAILABLE:
        return (
            "",
            "",
            "⚠️ OpenAI API key not configured. Set OPENAI_API_KEY to use the playground.",
        )

    if not query.strip():
        return "", "", "Please enter a question."

    try:
        llm = ChatOpenAI(model="gpt-5-nano", temperature=0.7)

        # Generate a simple response
        response = llm.invoke(f"Answer concisely: {query}")
        response_text = str(response.content)

        # Run Compass
        compass = CompassNode(
            model=llm,
            strategy=strategy,
            trigger=DefaultTriggerPolicy(min_response_length=20),
            max_suggestions=3,
            generate_candidates=3,
        )

        state = {"query": query, "response": response_text, "messages": []}
        result = compass(state, config={})
        suggestions = result.get("compass_suggestions", [])

        if suggestions:
            formatted = "\n".join(f"• {s}" for s in suggestions)
        else:
            formatted = "No follow-up suggestions generated."

        return response_text, formatted, ""

    except Exception as e:
        return "", "", f"Error: {e}"


# ============================================================================
# BUILD THE APP
# ============================================================================

with gr.Blocks(title="Compass Demo") as demo:
    gr.Markdown(HEADER)

    with gr.Tabs():
        # Tab 1: Guardrail Integration
        with gr.Tab("🛡️ Guardrails"):
            create_scenario_tab("guardrail")

        # Tab 2: Capability Bounding
        with gr.Tab("🎯 Capabilities"):
            create_scenario_tab("capability")

        # Tab 3: Novelty Filtering
        with gr.Tab("🔄 Novelty"):
            create_scenario_tab("novelty")

        # Tab 4: Workflow Personalization
        with gr.Tab("👤 Workflows"):
            create_scenario_tab("workflow")

        # Tab 5: Zero-Latency
        with gr.Tab("⚡ Zero-Latency"):
            create_scenario_tab("parallel")

        # Tab 6: Playground
        with gr.Tab("🎮 Playground"):
            gr.Markdown("### Try It Live")
            gr.Markdown(
                "*Uses gpt-5-nano for cost efficiency. "
                "Response times depend on OpenAI API latency.*"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., How do I deploy to AWS?",
                        lines=2,
                    )
                    strategy_select = gr.Dropdown(
                        choices=["adaptive", "clarifying", "exploratory", "deepening"],
                        value="adaptive",
                        label="Strategy",
                    )
                    submit_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=1):
                    response_output = gr.Textbox(
                        label="Agent Response",
                        lines=3,
                        interactive=False,
                    )
                    followups_output = gr.Textbox(
                        label="Compass Suggestions",
                        lines=3,
                        interactive=False,
                    )
                    error_output = gr.Markdown("")

            submit_btn.click(
                fn=run_playground,
                inputs=[query_input, strategy_select],
                outputs=[response_output, followups_output, error_output],
            )

        # Tab 7: Integration
        with gr.Tab("📦 Integration"):
            gr.Markdown("### Add Compass to Your Agent")
            gr.Markdown(INTEGRATION_CODE)

            gr.Markdown("### Key Benefits")
            gr.Markdown("""
| Feature | Benefit |
|---------|---------|
| **Capability-Bounded** | Only suggest what your agent can do |
| **Guardrail-Aware** | Auto-skip when guardrails fire |
| **Novelty Filtering** | No repetitive suggestions |
| **Zero-Latency** | Run in parallel via `Send()` |
| **Model Flexibility** | Use cheap models for follow-ups |
| **Full Tracing** | Every decision logged in LangSmith |
""")

            gr.Markdown("### Installation")
            gr.Code("pip install langgraph-compass", language="bash")

    gr.Markdown("---")
    gr.Markdown("*Built and battle-tested at Cisco*")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"), ssr_mode=False)
else:
    demo.theme = gr.themes.Soft(primary_hue="blue")
