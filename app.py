"""
Gradio Web Interface for DeepInflation

Chat interface with streaming tool execution status using ChatMessage metadata.
Supports custom API configuration (base_url, api_key, model).
"""

import os
import time

import gradio as gr
from dotenv import load_dotenv
from gradio import ChatMessage

from agent import DeepInflation

load_dotenv()


def create_agent_interface():
    """Create Gradio Blocks interface with streaming and custom API settings."""

    def initialize_agent(agent_state: dict, api_key: str, base_url: str, model: str, embedding_model: str, image_model: str) -> tuple:
        """Initialize agent with custom settings."""
        try:
            if not api_key.strip():
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return agent_state, "❌ API Key required", [], None

            base_url = base_url.strip() or os.getenv("BASE_URL") or None
            model = model.strip() or "gpt-5.2"
            embedding_model = embedding_model.strip() or "text-embedding-3-small"
            image_model = image_model.strip() or "dall-e-3"

            agent_state = {
                "agent": DeepInflation(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    embedding_model=embedding_model,
                    image_model=image_model,
                    verbose=True,
                )
            }

            config_str = f"Model: {model}\nEmbed: {embedding_model}"
            if base_url:
                config_str += f"\nURL: {base_url}"

            return agent_state, f"✓ Ready\n{config_str}", [], None

        except Exception as e:
            return agent_state, f"❌ {str(e)}", [], None

    def format_tool_args(tool_name: str, args: dict) -> str:
        """Format tool arguments for expandable display."""
        if not args:
            return ""

        if tool_name == "delegate_task_to_member":
            return f"**Task**: {args.get('task', '')}"

        if tool_name in ("analyze_potential", "plot_potential", "generate_schematic"):
            if tool_name == "generate_schematic":
                text = f"**Prompt**: `{args.get('prompt', '')}`"
            else:
                text = f"**Expression**: `{args.get('expression', '')}`"
            
            if args.get("output_path") or args.get("plot_path"):
                # Handle inconsistent arg names if necessary, but usually args has input args.
                # Actually, args dict here comes from tool_args which is INPUT args.
                # Output path is usually returned in result, not args, for these tools (unless passed as arg).
                # But for display purposes, we just show input args.
                pass
            return text

        if tool_name == "search_potential":
            return ""  # Will be populated by sr_config event

        # Generic: show all args
        return "\n".join(f"**{k}**: `{str(v)[:200]}`" for k, v in args.items())

    def display_user_message(message: str, history: list):
        """Immediately display user message."""
        history.append({"role": "user", "content": message})
        return "", history, None

    async def respond(agent_state: dict, history: list):
        """Async streaming response handler using agent.stream()."""
        if agent_state.get("agent") is None:
            history.append(ChatMessage(role="assistant", content="⚠️ Please initialize the agent first."))
            yield history, None
            return

        # Gradio 6.0: content is [{"type": "text", "text": "..."}]
        content = history[-1]["content"]
        user_message = content[0]["text"] if isinstance(content, list) else content
        agent = agent_state["agent"]

        # Track tool status messages: call_id -> (history index, tool_name, args_text)
        tool_indices = {}
        # Track streaming text response
        response_index = None
        accumulated_text = ""
        # Time-based batching to reduce LaTeX flicker
        last_update_time = 0
        UPDATE_INTERVAL = 0.15  # 150ms between updates

        try:
            async for event in agent.stream(user_message):
                event_type = event["type"]

                if event_type == "tool_start":
                    call_id = event["call_id"]
                    info = event["info"]
                    args = event.get("args", {})

                    tool_name = info.get("tool_name", "")
                    args_text = format_tool_args(tool_name, args)

                    tool_msg = ChatMessage(
                        role="assistant",
                        content=args_text,
                        metadata={
                            "title": info.get("title", "🔧 Tool"),
                            "log": info.get("log", ""),
                            "status": "pending",
                        },
                    )
                    tool_indices[call_id] = (len(history), tool_name, args_text)
                    history.append(tool_msg)
                    yield history, None

                elif event_type == "sr_config":
                    # Update search_potential tool message with config details
                    config = event["config"]
                    for call_id, (idx, tool_name, _) in list(tool_indices.items()):
                        if tool_name == "search_potential":
                            config_text = (
                                f"**Targets**: ns={config['ns_target']}±{config['ns_sigma']}, r={config['r_target']}±{config['r_sigma']}, N={config['N_obs']}\n"
                                f"**Operators**: {config['operators']}\n"
                                f"**Evolution**: {config['populations']} pops × {config['iterations']} iters\n"
                                f"**Est. time**: {config['estimated_time']}"
                            )
                            old_meta = getattr(history[idx], "metadata", {}) or {}
                            history[idx] = ChatMessage(
                                role="assistant",
                                content=config_text,
                                metadata={**old_meta, "status": "pending"},
                            )
                            tool_indices[call_id] = (idx, tool_name, config_text)
                            yield history, None
                            break

                elif event_type == "tool_end":
                    call_id = event["call_id"]
                    if call_id in tool_indices:
                        idx, _, args_text = tool_indices[call_id]
                        old_meta = getattr(history[idx], "metadata", {}) or {}
                        # Gradio status only accepts 'pending' or 'done'
                        # Indicate failure via title suffix
                        success = event.get("success", True)
                        title = old_meta.get("title", "🔧 Tool")
                        if not success and "✗" not in title:
                            title = f"{title} ✗"
                        history[idx] = ChatMessage(
                            role="assistant",
                            content=args_text,
                            metadata={
                                **old_meta,
                                "title": title,
                                "status": "done",
                                "duration": round(event.get("duration", 0), 1),
                            },
                        )
                        yield history, None

                elif event_type == "text_delta":
                    accumulated_text += event["delta"]
                    # Time-based batching to reduce LaTeX flicker
                    if time.time() - last_update_time >= UPDATE_INTERVAL or response_index is None:
                        last_update_time = time.time()
                        if response_index is None:
                            history.append(ChatMessage(role="assistant", content=accumulated_text))
                            response_index = len(history) - 1
                        else:
                            history[response_index] = ChatMessage(role="assistant", content=accumulated_text)
                        yield history, None

                elif event_type == "response":
                    # Final response
                    if response_index is None and event["content"]:
                        history.append(ChatMessage(role="assistant", content=event["content"]))
                    elif response_index is not None:
                        history[response_index] = ChatMessage(role="assistant", content=event["content"])
                    yield history, event.get("plot_path")

        except Exception as e:
            history.append(ChatMessage(role="assistant", content=f"❌ Error: {str(e)}"))
            yield history, None

    def clear_conversation(agent_state: dict):
        """Clear chat history and agent state."""
        if agent_state.get("agent") is not None:
            agent_state["agent"].clear_history()
        return agent_state, [], None

    # Build interface
    with gr.Blocks(title="DeepInflation") as interface:
        agent_state = gr.State({"agent": None})

        gr.Markdown("# DeepInflation")
        gr.Markdown("AI assistant for analyzing inflation cosmology models.")

        welcome = [
            {
                "role": "assistant",
                "content": """Welcome! I can help you explore inflation cosmology.

Ask about specific potentials, observational constraints, or let me search for models.

*Configure API settings on the right to start.*""",
            }
        ]

        with gr.Row():
            # Chat column
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    show_label=False,
                    height=600,
                    buttons=["copy"],
                    value=welcome,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "\\[", "right": "\\]", "display": True},
                        {"left": "$", "right": "$", "display": False},
                        {"left": "\\(", "right": "\\)", "display": False},
                    ],
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about inflation models...",
                        show_label=False,
                        lines=1,
                        max_lines=3,
                        scale=5,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                gr.Examples(
                    examples=[
                        "What is ns for V = phi^2?",
                        "Plot the Starobinsky model: (1 - exp(-sqrt(2/3)*phi))^2",
                        "Find a plateau potential with r < 0.01",
                    ],
                    inputs=msg_input,
                    label="📝 Examples",
                )

            # Config column
            with gr.Column(scale=2, min_width=280):
                with gr.Accordion("⚙️ Configuration", open=True):
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="Leave empty for env var",
                        type="password",
                        value="",
                    )
                    model_input = gr.Textbox(label="Chat Model", placeholder="gpt-5.2", value="gpt-5.2")
                    image_model_input = gr.Textbox(label="Image Model", placeholder="gpt-image-1", value="gpt-image-1")
                    embedding_model_input = gr.Textbox(
                        label="Embedding Model",
                        placeholder="text-embedding-3-small",
                        value="text-embedding-3-small",
                    )
                    base_url_input = gr.Textbox(
                        label="Base URL (Optional)",
                        placeholder="Custom endpoint",
                        value="",
                    )
                    init_btn = gr.Button("Initialize", variant="primary", size="sm")
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=3,
                        show_label=False,
                        placeholder="Click Initialize",
                    )

                gr.Markdown("""
### 💡 Quick Setup

**OpenAI:** Leave URL empty

**Custom:**
- URL: `https://your-endpoint/v1`
                """)

        # Plot output
        plot_output = gr.Image(
            label="Visualization",
            type="filepath",
            interactive=False,
            height=380,
            buttons=["download"],
        )

        # Event handlers
        init_btn.click(
            initialize_agent,
            inputs=[
                agent_state,
                api_key_input,
                base_url_input,
                model_input,
                embedding_model_input,
                image_model_input,
            ],
            outputs=[agent_state, status_output, chatbot, plot_output],
        )

        clear_btn.click(
            clear_conversation,
            inputs=[agent_state],
            outputs=[agent_state, chatbot, plot_output],
        ).then(lambda: "", outputs=[msg_input])

        # Submit handlers - display user message, then stream response
        msg_input.submit(
            display_user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, plot_output],
        ).then(respond, inputs=[agent_state, chatbot], outputs=[chatbot, plot_output])

        submit_btn.click(
            display_user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, plot_output],
        ).then(respond, inputs=[agent_state, chatbot], outputs=[chatbot, plot_output])

    return interface


if __name__ == "__main__":
    demo = create_agent_interface()

    # Sequential processing to avoid Julia/PySR concurrency issues
    demo.queue(max_size=10, default_concurrency_limit=1)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
