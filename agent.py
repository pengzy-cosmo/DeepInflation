"""Inflation Research Agent - Agno Framework Implementation"""

import json
import os
import time
from uuid import uuid4

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai.like import OpenAILike
from agno.run.agent import RunEvent
from agno.run.team import TeamRunEvent
from agno.team import Team

from encyclopedia_rag import EncyclopediaRAG
from sr_search import search_potential
from tools import analyze_potential, plot_potential, generate_schematic

# ============================================================================
# System Prompts
# ============================================================================

MAIN_AGENT_PROMPT = r"""You are an expert inflation cosmology assistant specialized in analyzing inflation potentials and discovering models via symbolic regression. Use tools or sub-agents to gather data, then provide clear answers.

# WORKFLOW (ReAct Pattern)
1. **Thought**: What information is needed?
2. **Action**: Call appropriate tool(s) or delegate to sub-agent
3. **Observation**: Examine results
4. Repeat until ready to answer

# DELEGATION

## SR Agent
**PRIMARY METHOD** of this agent team. Used for discovering potentials matching target observables or physics constraints via symbolic regression. (slow, 1-5 min).
**Delegate when**: User asks to find/discover potential, or wants models compatible with observational data.
**Input**: the task based on user's request (target observables, potential characteristics, constraints and time budget).
**Output**: Config summary and ranked candidates with expressions and predictions.

## Plotting Agent
**Visualization Expert**. Delegate all image generation tasks here.
**Delegate when**: User asks for plots, graphs, illustrations, or schematic images.
**Input**: Precise description of what to visualize (either a physics potential for plotting, or a conceptual description).

# TOOLS

## analyze_potential(expression)
Compute inflation observables (ns, r, A_s) for all valid trajectories of V(φ).
**Expression format**: Only 'phi' variable + numeric values (e.g., 'phi^2', '(1-exp(-0.816*phi))^2')
**Invalid**: Symbolic parameters like 'M*phi^2', 'V0*phi^2'

## search_knowledge_base (built-in)
Search Encyclopædia Inflationaris using hybrid retrieval (semantic + keyword).
**Use for**: Model names, physics concepts, theory background, potential expressions
**NOT for**: Finding models from observables (delegate to SR Agent instead)
**NOTE**: Only quiry with English text (No LaTeX, code, or math symbols)

# OUTPUT PRINCIPLES
- Focus on answering the user's question.
- The final answer should be concise and relevant.
- Use proper Markdown commands with $...$ for math.
- Always base the final answer on tool results, not assumptions. Do not invent data.
- Plots and images will be handled by the Plotting Agent; you just confirm they were generated.
"""

SR_AGENT_PROMPT = r"""You are a symbolic regression expert for inflation cosmology.
As a sub-agent, you should run the `search_potential` tool based on the main agent's delegation.
Note that you should run the `search_potential` tool once at a time, and return the config summary and results imediately unless instructed otherwise.

# YOUR TASK
1. **Interpret** the user's intent → extract observable targets and constraints
2. **Configure** PySR search parameters based on the guide below
3. **Run** search_potential with your configuration
4. **Return** both the config summary and the search results

# SYMBOLIC REGRESSION (PYSR) CONFIG
(See tool documentation for details)
"""

PLOTTING_AGENT_PROMPT = r"""You are a Scientific Visualization Expert for Inflation Cosmology.
Your goal is to choose the best visualization method for the user's request: **Physics Plot** vs **Conceptual Schematic**.

# DECISION LOGIC

1. **Physics-based Plotting** (`plot_potential`)
   - **Use when**: User asks to plot a specific inflation potential V(phi) or wants quantitative data (slow-roll parameters, ns-r constraints).
   - **Requirement**: You must have a computable mathematical expression for V(phi) involving only `phi` and numbers (e.g., `phi^2`, `(1-exp(-sqrt(2/3)*phi))^2`).
   - **Configuration**:
     - Analyze if the user wants specific panels (e.g., "show me the potential", "show constraints").
     - If not specified, default to ALL panels.
     - Pass `panels` list to the tool: `['potential']`, `['slow_roll']`, `['constraints']`, or combination.

2. **Conceptual/Schematic Art** (`generate_schematic`)
   - **Use when**:
     - User asks for "illustration", "artistic view", "cartoon", "concept".
     - The request is abstract (e.g., "draw the multiverse", "show inflation stability").
     - The potential is too complex for the numerical solver or symbolic (e.g., has undefined constants `V0`, `lambda`).
   - **Action**: Create a detailed, high-quality prompt for DALL-E describing the physical concept in an artistic way.

# FAILURE RECOVERY
- If `plot_potential` fails (e.g., due to solver error), catch the error and FALLBACK to `generate_schematic` to provide at least a visual aid, explaining that the numeric plot failed.

# OUTPUT
- perform the tool call.
- Return the file path of the generated image.
"""


# ============================================================================
# UI Helpers
# ============================================================================

# Tool display config: tool_name -> (emoji, display_title, is_long_running)
TOOL_DISPLAY_CONFIG = {
    "analyze_potential": ("🔬", "Analyzing", False),
    "plot_potential": ("📊", "Plotting", False),
    "generate_schematic": ("🎨", "Generating Image", True),
    "search_knowledge_base": ("📚", "Encyclopedia", False),
    "search_potential": ("🧬", "Symbolic Regression", True),
}


def _format_tool_info(tool_name: str, args: dict) -> dict:
    """Format tool info for UI display."""
    if tool_name == "delegate_task_to_member":
        member_name = args.get("member_id", "member").replace("-", " ").title()
        return {"tool_name": tool_name, "title": f"🤝 Delegate to {member_name}", "log": ""}

    emoji, title, long_running = TOOL_DISPLAY_CONFIG.get(tool_name, ("🔧", tool_name, False))
    result = {"tool_name": tool_name, "title": f"{emoji} {title}", "log": ""}
    
    if tool_name == "generate_schematic":
         result["log"] = f"Prompt: {args.get('prompt', '')[:50]}..."
         
    if long_running:
        result["long_running"] = True
    return result


def _extract_sr_config(config_json_str: str) -> dict:
    """Extract key SR config info for display."""
    try:
        config = json.loads(config_json_str)
        iterations = config.get("niterations", 40)
        return {
            "ns_target": config.get("ns_target", 0.9649),
            "ns_sigma": config.get("ns_sigma", 0.0042),
            "r_target": config.get("r_target", 0.0),
            "r_sigma": config.get("r_sigma", 0.014),
            "N_obs": config.get("N_obs", 60.0),
            "operators": config.get("binary_operators", []) + config.get("unary_operators", []),
            "iterations": iterations,
            "populations": config.get("populations", 31),
            "estimated_time": f"~{iterations // 10} min",
        }
    except (json.JSONDecodeError, TypeError):
        return {}


# ============================================================================
# DeepInflation Agent
# ============================================================================


class DeepInflation:
    """Inflation cosmology research agent with conversation management.

    Manages Agno Team, RAG system, session state, and streaming interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-5.2",
        embedding_model: str = "text-embedding-3-small",
        image_model: str = "dall-e-3",
        temperature: float = 1.0,
        verbose: bool = True,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key.")
        self._base_url = base_url or os.getenv("BASE_URL")
        self.verbose = verbose

        self._model = OpenAILike(
            id=model,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=temperature,
        )
        self._rag = EncyclopediaRAG(
            api_key=self._api_key,
            base_url=self._base_url,
            embedding_model=embedding_model,
        )
        self._db = SqliteDb(
            db_file="tmp/agent_storage.db",
            session_table="inflation_agent_sessions",
        )

        # Set verbose flag for submodules
        import encyclopedia_rag
        import tools as tools_module

        tools_module.VERBOSE = encyclopedia_rag.VERBOSE = verbose
        tools_module.IMAGE_MODEL = image_model

        if verbose:
            print(f"[Agent] Initializing with model={model}, image_model={image_model}, base_url={self._base_url or 'default'}")

        # Session state
        self.session_id = str(uuid4())
        self.last_plot_path: str | None = None
        self.team = self._create_team()

    def _create_team(self) -> Team:
        """Create Agno Team with SR and Plotting sub-agents."""
        
        # 1. Symbolic Regression Sub-Agent
        sr_agent = Agent(
            name="SR Agent",
            model=self._model,
            role="Configure and run symbolic regression for inflation potentials",
            instructions=SR_AGENT_PROMPT,
            tools=[search_potential],
            add_history_to_context=True,
            num_history_runs=3,
            markdown=True,
        )
        
        # 2. Plotting Sub-Agent (New)
        plotting_agent = Agent(
            name="Plotting Agent",
            model=self._model,
            role="Visualize physics concepts via plots or schematic art",
            instructions=PLOTTING_AGENT_PROMPT,
            tools=[plot_potential, generate_schematic],
            add_history_to_context=True,
            num_history_runs=2,
            markdown=True,
        )

        return Team(
            name="Inflation Research Team",
            model=self._model,
            members=[sr_agent, plotting_agent],
            tools=[analyze_potential],  # Note: plot_potential moved to sub-agent
            instructions=MAIN_AGENT_PROMPT,
            show_members_responses=True,
            markdown=True,
            db=self._db,
            add_history_to_context=True,
            num_history_runs=5,
            knowledge=self._rag.knowledge,
            search_knowledge=True,
        )

    async def stream(self, question: str):
        """Async streaming interface for Gradio.

        Yields:
            {"type": "tool_start", "call_id": str, "info": dict, "args": dict}
            {"type": "tool_end", "call_id": str, "duration": float}
            {"type": "sr_config", "config": dict}
            {"type": "text_delta", "delta": str}
            {"type": "response", "content": str, "plot_path": str|None}
        """
        self.last_plot_path = None
        pending_tools = {}  # call_id -> (tool_name, info, args, start_time)
        accumulated_text = ""

        try:
            async for event in self.team.arun(
                input=question,
                stream=True,
                stream_events=True,
                session_id=self.session_id,
            ):
                # Tool call started (Team or Member)
                if event.event in (TeamRunEvent.tool_call_started, RunEvent.tool_call_started):
                    tool_name = event.tool.tool_name
                    tool_args = event.tool.tool_args or {}
                    is_member = event.event == RunEvent.tool_call_started

                    if self.verbose:
                        prefix = "[Member Tool]" if is_member else "[Tool]"
                        args_str = str(tool_args)[:500]
                        print(f"{prefix} {tool_name}\n  Input: {args_str}{'...' if len(str(tool_args)) > 500 else ''}")

                    info = _format_tool_info(tool_name, tool_args)
                    call_id = f"{'member' if is_member else 'team'}_{tool_name}_{time.time()}"
                    pending_tools[call_id] = (tool_name, info, tool_args, time.time())
                    yield {
                        "type": "tool_start",
                        "call_id": call_id,
                        "info": info,
                        "args": tool_args,
                    }

                    # SR config display (member only)
                    if is_member and tool_name == "search_potential":
                        sr_config = _extract_sr_config(tool_args.get("config_json", "{}"))
                        if sr_config:
                            yield {"type": "sr_config", "config": sr_config}

                # Tool call completed (Team or Member)
                elif event.event in (
                    TeamRunEvent.tool_call_completed,
                    RunEvent.tool_call_completed,
                ):
                    tool_name = event.tool.tool_name
                    result = event.tool.result
                    is_member = event.event == RunEvent.tool_call_completed

                    # Parse success from JSON result
                    success = True
                    try:
                        output = json.loads(result) if isinstance(result, str) else result
                        if isinstance(output, dict):
                            success = output.get("success", True)
                    except (json.JSONDecodeError, TypeError):
                        output = None

                    # Find and pop matching pending tool
                    call_id = next(
                        (cid for cid, (name, *_) in pending_tools.items() if name == tool_name),
                        None,
                    )
                    if call_id:
                        _, _, _, start_time = pending_tools.pop(call_id)
                        duration = time.time() - start_time

                        if self.verbose:
                            prefix = "[Member Tool]" if is_member else "[Tool]"
                            output_str = str(result)[:500] if result else ""
                            print(
                                f"{prefix} {tool_name} {'✓' if success else '✗'} ({duration:.1f}s)\n  Output: {output_str}{'...' if len(str(result)) > 500 else ''}"
                            )

                        yield {
                            "type": "tool_end",
                            "call_id": call_id,
                            "duration": duration,
                            "success": success,
                        }

                        # Extract plot path on success
                        if tool_name in ("plot_potential", "generate_schematic") and isinstance(output, dict) and output.get("success"):
                            path = output.get("plot_path")
                            if path and os.path.exists(path):
                                self.last_plot_path = path

                # Content streaming
                elif event.event == TeamRunEvent.run_content:
                    if event.content:
                        accumulated_text += event.content
                        yield {"type": "text_delta", "delta": event.content}

            yield {
                "type": "response",
                "content": accumulated_text,
                "plot_path": self.last_plot_path,
            }

        except Exception as e:
            if self.verbose:
                print(f"[Agent] Error: {e}")
                import traceback

                traceback.print_exc()
            yield {"type": "response", "content": f"Error: {e}", "plot_path": None}

    def run(self, question: str) -> str:
        """Synchronous interface - returns final response only."""
        import asyncio

        async def get_response():
            async for event in self.stream(question):
                if event["type"] == "response":
                    return event["content"]
            return ""

        return asyncio.run(get_response())

    def clear_history(self):
        """Clear conversation history by creating new session."""
        self.session_id = str(uuid4())
        self.last_plot_path = None
        if self.verbose:
            print("[Agent] History cleared (new session)")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("DeepInflation")
    print("=" * 60)
    print("\nExamples:")
    print("  • What is ns for V = phi^2?")
    print("  • Plot the Starobinsky model: (1 - exp(-sqrt(2/3)*phi))^2")
    print("  • Find a plateau potential with r < 0.01")
    print("\nType 'quit' to exit, 'clear' to reset conversation\n")

    async def main():
        agent = DeepInflation(verbose=False)
        pending_tools = {}  # call_id -> title

        while True:
            try:
                question = input("> ").strip()
                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                if question.lower() == "clear":
                    agent.clear_history()
                    pending_tools.clear()
                    continue

                print("\n⏳ Thinking...")

                async for event in agent.stream(question):
                    if event["type"] == "tool_start":
                        info = event["info"]
                        call_id = event["call_id"]
                        title = info.get("title", "Tool")
                        log = info.get("log", "")
                        pending_tools[call_id] = title
                        print(f"  {title} {log}..." if log else f"  {title}...")
                    elif event["type"] == "sr_config":
                        config = event["config"]
                        print(
                            f"    Config: ns={config.get('ns_target')}±{config.get('ns_sigma')}, "
                            f"r={config.get('r_target')}±{config.get('r_sigma')}"
                        )
                        print(f"    Ops: {config.get('operators')}")
                        print(f"    Est: {config.get('estimated_time')}")
                    elif event["type"] == "tool_end":
                        call_id = event["call_id"]
                        duration = event.get("duration", 0)
                        title = pending_tools.pop(call_id, "Tool")
                        print(f"  {title} ✓ ({duration:.1f}s)")
                    elif event["type"] == "response":
                        print(f"\n{event['content']}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import sys

        sys.exit(1)
