"""
Compass Demo - Interactive showcase of intelligent follow-up generation.
"""

import os

import gradio as gr

# Only import Compass dependencies for playground
OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))

if OPENAI_AVAILABLE:
    from langchain_openai import ChatOpenAI

    from compass import CompassNode, DefaultTriggerPolicy

# ============================================================================
# THEME & STYLING
# ============================================================================

CUSTOM_CSS = """
/* Chatbot - remove fixed height, auto-size to content */
.chatbot-container {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
}
.chatbot-container > div {
    height: auto !important;
    max-height: none !important;
}

.scenario-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.insight-box {
    background: #1a1a2e;
    border-left: 3px solid #4f8cff;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
    color: #e0e0e0;
}
.insight-box strong {
    color: #4f8cff;
}
.verdict-good {
    color: #1e8e3e;
    font-weight: 500;
}
.verdict-bad {
    color: #d93025;
    font-weight: 500;
}

/* Timeline visualization */
.timeline-container {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin: 8px 0;
}
.timeline-row {
    display: flex;
    align-items: center;
    margin: 8px 0;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
}
.timeline-label {
    width: 140px;
    font-weight: 500;
    color: #333;
}
.timeline-bar-container {
    flex: 1;
    height: 28px;
    background: #e9ecef;
    border-radius: 4px;
    position: relative;
    overflow: visible;
}
.timeline-bar {
    height: 100%;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    color: white;
    font-weight: 500;
    font-size: 13px;
}
.bar-agent {
    background: linear-gradient(90deg, #4285f4 0%, #5a9cf4 100%);
}
.bar-compass {
    background: linear-gradient(90deg, #34a853 0%, #4db66a 100%);
}
.bar-compass-waiting {
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 4px,
        #e9ecef 4px,
        #e9ecef 8px
    );
    border: 1px dashed #aaa;
}
.timeline-total {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #dee2e6;
    font-weight: 600;
    font-size: 15px;
}
.timeline-total-line {
    height: 3px;
    border-radius: 2px;
    margin-top: 4px;
}
.total-sequential {
    background: #6c757d;
}
.total-parallel {
    background: #1e8e3e;
}
.time-saved {
    color: #1e8e3e;
    font-weight: 600;
    font-size: 1.1rem;
    margin-top: 16px;
}
"""

# ============================================================================
# SCENARIO DATA
# ============================================================================

# Tab 1: Grounded Suggestions (was "Capabilities")
GROUNDED_SCENARIO = {
    "title": "Grounded Suggestions",
    "subtitle": "Only suggest follow-ups your agent can actually handle",
    "context": (
        "Your IT support agent handles: password resets, account settings, "
        "software installation, and VPN setup. It does NOT handle: hardware issues, "
        "procurement, or HR questions."
    ),
    "few_shot_examples": [
        "Would you like help with your VPN connection?",
        "Should I walk you through the software installation steps?",
        "Do you need to update your account security settings?",
        "Want me to explain the password requirements?",
    ],
    "conversation": [
        ("user", "How do I reset my password?"),
        (
            "assistant",
            "To reset your password:\n1. Go to Settings → Security\n2. Click 'Reset Password'\n3. Check your email for the confirmation link\n\nThe new password must be at least 12 characters.",
        ),
    ],
    "naive": {
        "suggestion": "Having other IT issues I can help with?",
        "problem": "Too vague — could lead user to ask about hardware (which agent can't handle)",
    },
    "compass": {
        "suggestions": [
            "Would you like to set up two-factor authentication?",
            "Need help configuring your VPN connection?",
        ],
        "benefit": "Grounded in capability map — only suggests supported paths",
    },
    "code": '''# Pass capability examples to Compass via ExampleRetriever
class CapabilityRetriever:
    """Returns example questions from your agent's capability map."""

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        # MMR search: relevant AND diverse
        return self.capability_store.max_marginal_relevance_search(
            query, k=k, fetch_k=20, lambda_mult=0.7
        )

compass = CompassNode(
    model=llm,
    example_retriever=CapabilityRetriever(store),
)''',
    "insight": "The ExampleRetriever grounds suggestions in your agent's actual capabilities. Pass your capability examples to Compass and it will only suggest follow-ups your agent can handle.",
}

# Tab 2: Workflow Personalization
WORKFLOW_SCENARIO = {
    "title": "Workflow Personalization",
    "subtitle": "Learn user patterns and suggest their typical next steps",
    "context": """This user's learned workflow pattern (via LangMem):
Every Monday morning: Check metrics → Review alerts → Update dashboard → Send report""",
    "conversation": [
        ("user", "Show me this week's metrics"),
        (
            "assistant",
            "Here are your metrics for this week:\n\n- Revenue: $142K (+12%)\n- Active users: 4,521\n- Churn rate: 2.1%\n- Support tickets: 89 (down 15%)",
        ),
    ],
    "naive": {
        "suggestion": "Is there anything else you'd like to know?",
        "problem": "Generic — doesn't leverage what we know about this user",
    },
    "compass": {
        "suggestion": "Ready to review this week's alerts?",
        "benefit": "Workflow-aware — suggests the user's typical next action",
    },
    "code": '''# Integrate with LangMem for personalized workflows
class WorkflowRetriever:
    """Retrieves user's typical next actions from LangMem."""

    def __init__(self, store, user_id: str):
        self.store = store
        self.user_id = user_id

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        memories = self.store.search(
            ("workflows", self.user_id), query=query
        )
        return [m.content["typical_next_step"] for m in memories]

compass = CompassNode(
    model=llm,
    example_retriever=WorkflowRetriever(store, user_id),
)''',
    "insight": "With LangMem integration, Compass learns each user's workflow and proactively suggests their typical next action — turning your assistant into a personalized co-pilot.",
}

# Tab 3: Conditional Triggers
CONDITIONAL_SCENARIO = {
    "title": "Conditional Triggers",
    "subtitle": "Control exactly when follow-ups are generated",
    "context": (
        "Compass can skip follow-up generation based on any state condition: "
        "guardrails fired, specific intents detected, short responses, or custom logic."
    ),
    "examples": [
        {
            "name": "Guardrail Blocked",
            "conversation": [
                ("user", "What's the best way to bypass the firewall?"),
                (
                    "assistant",
                    "I can't help with that request. If you're having legitimate network access issues, please contact IT security.",
                ),
            ],
            "state": "policy_violation: True",
            "naive": "Would you like tips on network troubleshooting?",
            "naive_problem": "Still engages with blocked topic",
            "compass": "[No follow-up generated]",
            "compass_benefit": "Trigger detected guardrail → skipped",
        },
        {
            "name": "Greeting / Small Talk",
            "conversation": [
                ("user", "Hey, good morning!"),
                ("assistant", "Good morning! How can I help you today?"),
            ],
            "state": "intent: greeting",
            "naive": "Would you like to explore our features?",
            "naive_problem": "Pushy — user hasn't asked for anything yet",
            "compass": "[No follow-up generated]",
            "compass_benefit": "Trigger detected greeting → skipped",
        },
        {
            "name": "Short Response",
            "conversation": [
                ("user", "What time is it?"),
                ("assistant", "It's 3:45 PM."),
            ],
            "state": "response_length: 12 chars",
            "naive": "Want to set a reminder?",
            "naive_problem": "Over-eager for a simple factual answer",
            "compass": "[No follow-up generated]",
            "compass_benefit": "Response too short → skipped",
        },
    ],
    "code": """trigger = DefaultTriggerPolicy(
    skip_on_guardrail=True,
    guardrail_keys=["policy_violation", "pii_detected"],  # Your keys
    skip_classifications=["greeting", "farewell", "thanks"],
    min_response_length=50,
    custom_skip_keys=["user_opted_out"],  # Any custom conditions
)

compass = CompassNode(model=llm, trigger=trigger)""",
    "insight": "Trigger policies give you fine-grained control. Skip on guardrails, specific intents, short responses, or any custom state condition — all configurable to your state keys.",
}

# Tab 4: Novelty Filter
NOVELTY_SCENARIO = {
    "title": "Novelty Filtering",
    "subtitle": "Avoid repetitive suggestions across conversation turns",
    "conversation_history": [
        {
            "turn": 1,
            "query": "How do I export my data?",
            "followup": "Want details on the export formats?",
        },
        {
            "turn": 2,
            "query": "What formats are available?",
            "followup": "Interested in scheduling automatic exports?",
        },
        {"turn": 3, "query": "Can I automate this?", "followup": None},
    ],
    "current_conversation": [
        ("user", "Can I automate this?"),
        (
            "assistant",
            "Yes! Go to Settings → Automation → Scheduled Exports.\n\nYou can set daily, weekly, or monthly schedules. Exports are sent to your configured destination automatically.",
        ),
    ],
    "naive": {
        "suggestion": "Would you like more details on export options?",
        "problem": "Similar to Turn 1 — repetitive pattern",
    },
    "compass": {
        "suggestion": "Should I explain the notification settings for completed exports?",
        "benefit": "Ranker filtered similar suggestions → fresh topics only",
    },
    "how_it_works": (
        "The ranker extracts meaningful words and filters suggestions "
        "with >70% overlap to previous follow-ups:\n\n"
        'Previous: ["details", "export", "formats"]\n'
        'Blocked:  ["details", "export", "options"] — too similar\n'
        'Allowed:  ["notification", "completed", "alerts"] — fresh topic'
    ),
    "code": """# Novelty filtering happens automatically
# The ranker checks against conversation history

compass = CompassNode(
    model=llm,
    max_suggestions=2,
    generate_candidates=5,  # Generate more, filter to best
)

# Ranker uses word overlap (not embeddings) for speed
# Threshold configurable via SuggestionRanker if needed""",
    "insight": "The ranker tracks conversation history and filters suggestions that overlap with previous follow-ups. No repeated patterns, every suggestion is fresh.",
}

# Tab 5: Zero Latency
PARALLEL_SCENARIO = {
    "title": "Zero-Latency Pattern",
    "subtitle": "Run follow-up generation in parallel — it's free",
    "explanation": (
        "Since Compass is a separate node, you can run it in parallel with your agent "
        "using LangGraph's Send() pattern. Follow-ups are ready the moment your response is."
    ),
    "sequential": {
        "agent_time": 2.1,
        "compass_time": 0.8,
        "total": 2.9,
    },
    "parallel": {
        "agent_time": 2.1,
        "compass_time": 0.8,
        "total": 2.1,
        "savings": 0.8,
    },
    "code": '''from langgraph.types import Send
from compass import get_compass_instruction

def route_parallel(state):
    """Run agent and compass simultaneously."""
    return [
        Send("agent", state),
        Send("compass", state),  # Uses previous turn's context
    ]

builder.add_conditional_edges("router", route_parallel)

# ─────────────────────────────────────────────────────────────
# Compass prepares suggestions speculatively while agent runs.
# In your response synthesis step, weave them in organically:
# ─────────────────────────────────────────────────────────────

def synthesize_response(state):
    """Final step: combine agent response with follow-ups."""
    response = state["agent_response"]
    suggestions = state.get("compass_suggestions", [])

    if suggestions:
        # Option 1: Inject as natural continuation
        instruction = get_compass_instruction(suggestions)
        final = f"{response}\\n\\n{instruction}"

        # Option 2: Return structured for UI rendering as chips/buttons
        # return {"response": response, "followup_chips": suggestions}

    return {"final_response": final}''',
    "insight": "Compass runs speculatively in parallel. By the time your agent finishes, follow-ups are ready to weave into the response — either as natural text or structured chips for your UI.",
}


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================


def create_chat_display(messages: list[tuple[str, str]]) -> list[dict]:
    """Convert message tuples to Gradio chatbot format."""
    result = []
    for role, content in messages:
        if role == "user":
            result.append({"role": "user", "content": content})
        else:
            result.append({"role": "assistant", "content": content})
    return result


def create_comparison_block(naive: dict, compass: dict, show_multiple: bool = False):
    """Create side-by-side comparison UI."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Naive Prompting**")
            suggestion = naive.get("suggestion", "")
            gr.Textbox(value=suggestion, label="Suggestion", interactive=False, lines=2)
            gr.Markdown(f"<span class='verdict-bad'>Problem: {naive['problem']}</span>")

        with gr.Column(scale=1):
            gr.Markdown("**Compass**")
            if show_multiple and isinstance(compass.get("suggestions"), list):
                suggestions = compass["suggestions"]
                for i, s in enumerate(suggestions):
                    gr.Textbox(value=s, label=f"Option {i + 1}", interactive=False, lines=1)
            else:
                # Single suggestion
                suggestion = compass.get("suggestion") or (
                    compass.get("suggestions", [""])[0] if compass.get("suggestions") else ""
                )
                gr.Textbox(value=suggestion, label="Suggestion", interactive=False, lines=2)
            gr.Markdown(f"<span class='verdict-good'>Benefit: {compass['benefit']}</span>")


# ============================================================================
# TAB BUILDERS
# ============================================================================


def build_grounded_tab():
    """Tab 1: Grounded Suggestions"""
    s = GROUNDED_SCENARIO

    gr.Markdown(f"### {s['title']}")
    gr.Markdown(f"*{s['subtitle']}*")

    gr.Markdown("**Agent Context**")
    gr.Markdown(s["context"])

    gr.Markdown("---")
    gr.Markdown("**Conversation**")
    gr.Chatbot(value=create_chat_display(s["conversation"]), elem_classes=["chatbot-container"])

    gr.Markdown("**Follow-up Comparison**")
    create_comparison_block(s["naive"], s["compass"], show_multiple=True)

    gr.Markdown("**How Compass knows what to suggest**")
    examples_text = "\n".join(f"- {ex}" for ex in s["few_shot_examples"])
    gr.Markdown(f"Pass capability examples via `ExampleRetriever`:\n{examples_text}")

    gr.Markdown("**Code**")
    gr.Code(s["code"], language="python", interactive=False)

    gr.Markdown(f"<div class='insight-box'><strong>Insight:</strong> {s['insight']}</div>")


def build_workflow_tab():
    """Tab 2: Workflow Personalization"""
    s = WORKFLOW_SCENARIO

    gr.Markdown(f"### {s['title']}")
    gr.Markdown(f"*{s['subtitle']}*")

    gr.Markdown("**User Context (from LangMem)**")
    gr.Markdown(s["context"])

    gr.Markdown("---")
    gr.Markdown("**Conversation**")
    gr.Chatbot(value=create_chat_display(s["conversation"]), elem_classes=["chatbot-container"])

    gr.Markdown("**Follow-up Comparison**")
    create_comparison_block(s["naive"], s["compass"])

    gr.Markdown("**Code**")
    gr.Code(s["code"], language="python", interactive=False)

    gr.Markdown(f"<div class='insight-box'><strong>Insight:</strong> {s['insight']}</div>")


def build_conditional_tab():
    """Tab 3: Conditional Triggers"""
    s = CONDITIONAL_SCENARIO

    gr.Markdown(f"### {s['title']}")
    gr.Markdown(f"*{s['subtitle']}*")
    gr.Markdown(s["context"])

    for example in s["examples"]:
        gr.Markdown(f"---\n**Scenario: {example['name']}**")
        gr.Markdown(f"`State: {example['state']}`")

        gr.Chatbot(
            value=create_chat_display(example["conversation"]), elem_classes=["chatbot-container"]
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**Naive**")
                gr.Textbox(value=example["naive"], interactive=False, lines=1, show_label=False)
                gr.Markdown(f"<span class='verdict-bad'>Problem: {example['naive_problem']}</span>")
            with gr.Column(scale=1):
                gr.Markdown("**Compass**")
                gr.Textbox(value=example["compass"], interactive=False, lines=1, show_label=False)
                gr.Markdown(
                    f"<span class='verdict-good'>Benefit: {example['compass_benefit']}</span>"
                )

    gr.Markdown("---\n**Code**")
    gr.Code(s["code"], language="python", interactive=False)

    gr.Markdown(f"<div class='insight-box'><strong>Insight:</strong> {s['insight']}</div>")


def build_novelty_tab():
    """Tab 4: Novelty Filtering"""
    s = NOVELTY_SCENARIO

    gr.Markdown(f"### {s['title']}")
    gr.Markdown(f"*{s['subtitle']}*")

    gr.Markdown("**Conversation History**")
    for turn in s["conversation_history"]:
        fu = f' → *"{turn["followup"]}"*' if turn["followup"] else " → *(current turn)*"
        gr.Markdown(f'Turn {turn["turn"]}: "{turn["query"]}"{fu}')

    gr.Markdown("---\n**Current Turn**")
    gr.Chatbot(
        value=create_chat_display(s["current_conversation"]), elem_classes=["chatbot-container"]
    )

    gr.Markdown("**Follow-up Comparison**")
    create_comparison_block(s["naive"], s["compass"])

    gr.Markdown("**How It Works**")
    gr.Code(s["how_it_works"], language=None, interactive=False)

    gr.Markdown("**Code**")
    gr.Code(s["code"], language="python", interactive=False)

    gr.Markdown(f"<div class='insight-box'><strong>Insight:</strong> {s['insight']}</div>")


def build_parallel_tab():
    """Tab 5: Zero-Latency Pattern"""
    s = PARALLEL_SCENARIO

    gr.Markdown(f"### {s['title']}")
    gr.Markdown(f"*{s['subtitle']}*")
    gr.Markdown(s["explanation"])

    gr.Markdown("---")
    gr.Markdown("**Latency Comparison**")

    # Calculate percentages for bar widths (based on total sequential time)
    total_seq = s["sequential"]["total"]
    agent_pct = int((s["sequential"]["agent_time"] / total_seq) * 100)
    compass_pct = int((s["sequential"]["compass_time"] / total_seq) * 100)

    # Sequential timeline - Compass AFTER agent (added latency)
    sequential_html = f"""
    <div class="timeline-container">
        <div class="timeline-row">
            <span class="timeline-label">Agent Response</span>
            <div class="timeline-bar-container">
                <div class="timeline-bar bar-agent" style="width: {agent_pct}%">{s["sequential"]["agent_time"]}s</div>
            </div>
        </div>
        <div class="timeline-row">
            <span class="timeline-label">Compass</span>
            <div class="timeline-bar-container">
                <div style="width: {agent_pct}%; display: inline-block;"></div>
                <div class="timeline-bar bar-compass" style="width: {compass_pct}%; display: inline-block; position: absolute; left: {agent_pct}%;">{s["sequential"]["compass_time"]}s</div>
            </div>
        </div>
        <div class="timeline-total">
            Total: {s["sequential"]["total"]}s
            <div class="timeline-total-line total-sequential" style="width: 100%"></div>
        </div>
    </div>
    """

    # Parallel timeline - Compass runs DURING agent (no added latency)
    parallel_html = f"""
    <div class="timeline-container" style="background: #e8f5e9;">
        <div class="timeline-row">
            <span class="timeline-label">Agent Response</span>
            <div class="timeline-bar-container">
                <div class="timeline-bar bar-agent" style="width: {agent_pct}%">{s["parallel"]["agent_time"]}s</div>
            </div>
        </div>
        <div class="timeline-row">
            <span class="timeline-label">Compass</span>
            <div class="timeline-bar-container">
                <div class="timeline-bar bar-compass" style="width: {compass_pct}%">{s["parallel"]["compass_time"]}s</div>
            </div>
        </div>
        <div class="timeline-total" style="color: #1e8e3e;">
            Total: {s["parallel"]["total"]}s (same as agent alone!)
            <div class="timeline-total-line total-parallel" style="width: {agent_pct}%"></div>
        </div>
    </div>
    """

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Sequential Execution**")
            gr.HTML(sequential_html)

        with gr.Column(scale=1):
            gr.Markdown("**Parallel Execution (with Send)**")
            gr.HTML(parallel_html)

    savings_pct = int(s["parallel"]["savings"] / s["sequential"]["total"] * 100)
    gr.Markdown(
        f"<p class='time-saved'>Time saved: {s['parallel']['savings']}s ({savings_pct}% faster)</p>"
    )

    gr.Markdown("**Code**")
    gr.Code(s["code"], language="python", interactive=False)

    gr.Markdown(f"<div class='insight-box'><strong>Insight:</strong> {s['insight']}</div>")


def build_playground_tab():
    """Tab 6: Live Playground"""
    gr.Markdown("### Playground")
    gr.Markdown("*Try Compass live with your own queries.*")

    if not OPENAI_AVAILABLE:
        gr.Markdown(
            "**Note:** Set `OPENAI_API_KEY` environment variable to enable live generation."
        )
        return

    gr.Markdown(
        "Uses `gpt-5-nano` for cost efficiency. Response time depends on OpenAI API latency."
    )

    def run_playground(query: str, strategy: str, num_suggestions: int) -> tuple[str, str, str]:
        if not query.strip():
            return "", "", "Please enter a question."
        try:
            llm = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
            response = llm.invoke(f"Answer concisely: {query}")
            response_text = str(response.content)

            compass = CompassNode(
                model=llm,
                strategy=strategy,
                trigger=DefaultTriggerPolicy(min_response_length=20),
                max_suggestions=num_suggestions,
                generate_candidates=num_suggestions + 2,
            )

            state = {"query": query, "response": response_text, "messages": []}
            result = compass(state, config={})
            suggestions = result.get("compass_suggestions", [])

            if suggestions:
                formatted = "\n".join(f"- {s}" for s in suggestions)
            else:
                formatted = "No follow-up suggestions generated."

            return response_text, formatted, ""
        except Exception as e:
            return "", "", f"Error: {e}"

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Your Question", placeholder="e.g., How do I deploy to AWS?", lines=2
            )
            strategy_select = gr.Dropdown(
                choices=["adaptive", "clarifying", "exploratory", "deepening"],
                value="adaptive",
                label="Strategy",
            )
            num_suggestions = gr.Slider(
                minimum=1, maximum=3, value=2, step=1, label="Number of Suggestions"
            )
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            response_output = gr.Textbox(label="Agent Response", lines=3, interactive=False)
            followups_output = gr.Textbox(label="Compass Suggestions", lines=4, interactive=False)
            error_output = gr.Markdown("")

    submit_btn.click(
        fn=run_playground,
        inputs=[query_input, strategy_select, num_suggestions],
        outputs=[response_output, followups_output, error_output],
    )


# ============================================================================
# MAIN APP
# ============================================================================

HEADER = """# Compass

**Intelligent follow-up question generation for LangGraph agents**

Compass gives you granular control over follow-up generation — grounding suggestions
in your agent's capabilities, learning user workflows, and skipping when appropriate.

[GitHub](https://github.com/sardanaaman/langgraph-compass) |
[PyPI](https://pypi.org/project/langgraph-compass/) |
[Documentation](https://github.com/sardanaaman/langgraph-compass#readme)
"""

FOOTER = """---
*Built and battle-tested at Cisco*
"""

with gr.Blocks(title="Compass Demo") as demo:
    gr.Markdown(HEADER)

    with gr.Tabs():
        with gr.Tab("Grounded Suggestions"):
            build_grounded_tab()

        with gr.Tab("Workflow Personalization"):
            build_workflow_tab()

        with gr.Tab("Conditional Triggers"):
            build_conditional_tab()

        with gr.Tab("Novelty Filtering"):
            build_novelty_tab()

        with gr.Tab("Zero-Latency"):
            build_parallel_tab()

        with gr.Tab("Playground"):
            build_playground_tab()

    gr.Markdown(FOOTER)

if __name__ == "__main__":
    # Try different themes: Default, Soft, Monochrome, Glass, Base, Ocean, Origin
    demo.launch(theme=gr.themes.Base(), css=CUSTOM_CSS, ssr_mode=False)
else:
    demo.css = CUSTOM_CSS
    demo.theme = gr.themes.Base()
