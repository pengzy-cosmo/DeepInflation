"""Inflation research agent with a small handwritten agent loop."""

import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI

from .encyclopedia_rag import init_rag, search_encyclopedia
from .sr_search import search_potential
from .tools import analyze_potential, plot_potential

# ============================================================================
# System Prompts
# ============================================================================

MAIN_AGENT_PROMPT = r"""You are an expert inflation cosmology assistant specialized in analyzing inflation potentials and discovering models via symbolic regression.

# WORKFLOW (ReAct)

1. **Thought**: What does the user need? Which tool or delegation is appropriate?
2. **Action**: Call tool(s) or delegate to SR Agent
3. **Observation**: Examine results - successful? sufficient?
4. **Repeat** or **Answer**: If more info needed, continue; otherwise synthesize response

# DECISION TREE

```
User Request
├─ "What is ns/r for V = ...?" → analyze_potential
├─ "Plot/show/visualize V = ..." → plot_potential
├─ "What is [model name]?" / "Explain [concept]" → search_encyclopedia
├─ "Find/discover potential with ns≈.../r<..." → delegate to SR Agent
└─ "Find models compatible with Planck data" → delegate to SR Agent
```

# DELEGATION (SR Agent)

SR Agent runs **symbolic regression** to discover V(φ) expressions matching target observables (ns, r) and physics constraints. This is slow (1-5 min) but powerful for finding new models.

**When to delegate**: User wants to find/discover/search for potentials, or wants models compatible with observational data.
**Your role**: Extract user's physics goals (target ns, r, constraints, potential characteristics) and pass them clearly.
**SR Agent returns**: Search config summary + ranked candidates with (expression, ns, r, loss)

# TOOLS

## analyze_potential(expression)
Compute ns, r, A_s for all valid inflation trajectories.
- **Input**: V(φ) with concrete numbers only (e.g., `phi^2`, `(1-exp(-0.816*phi))^2`)
- **Invalid**: Symbolic parameters (`M*phi^2`, `V0*exp(-phi)`)
- **Output**: JSON with trajectory list, each containing ns, r, A_s, phi_end, phi_N

## plot_potential(expression, output_path)
Generate 3-panel diagnostic plot.
- Panel 1: V(φ) with trajectory markers (φ_end, N=50, N=60)
- Panel 2: Slow-roll parameters ε, η vs φ
- Panel 3: Predicted (ns, r) overlaid on Planck+BK18 posterior
- **Returns**: Absolute path to saved PNG

## search_encyclopedia(query, top_k=3)
Query Encyclopædia Inflationaris (100+ inflation models).
- **Use for**: Inflation models and physics background
- **NOT for**: Finding models from observables → delegate to SR Agent
- **Query format**: Plain English (no LaTeX or math symbols)
- **Returns**: Full model documentation including potential forms and theoretical background
- **Citation required**: When using information from this tool, cite the source

## run_sr_agent(task)
Delegate the task to the dedicated SR Agent.
- **Use for**: Searching or discovering potentials from target observables or constraints
- **Input**: A concise task description in plain English
- **Returns**: Search config summary + ranked candidates

# OUTPUT PRINCIPLES

- Focus on answering the user's question; be concise and relevant.
- Use proper Markdown with $...$ for math.
- For plots, provide the saved file path.
- Always base the final answer on tool results, not assumptions. Do not invent data. If data is missing or inconclusive, state that clearly.
- **Citation**: When using information from `search_encyclopedia`, include a reference:
  > Source: Encyclopædia Inflationaris ([arXiv:1303.3787](https://arxiv.org/abs/1303.3787))

# ERROR HANDLING

If SR Agent returns no valid candidates:
1. **Analyze** the failure: constraints too tight? search space too narrow? targets unrealistic?
2. **Explain** clearly what was attempted and why it failed
3. **Suggest** a reasonable next step
"""

SR_AGENT_PROMPT = r"""You are a symbolic regression specialist. Your job is to configure and run PySR searches to discover inflation potentials V(φ) matching target observables.

# WORKFLOW

1. **Interpret** the delegated task → extract physics goals (target ns, r, constraints, time budget)
2. **Configure** PySR parameters following the guide below
3. **Run** `search_potential` with your config JSON
4. **Return** config summary + ranked results immediately

Note: Run `search_potential` ONLY ONCE per task.

# PYSR CONFIG REFERENCE

Construct `config_json` based on physics goals and time budget.

## Physics Targets

- **ns_target** (default 0.9649): Target scalar spectral index
- **ns_sigma** (default 0.0042): Tolerance for ns (widen for exploration, tighten for precision)
- **r_target** (default 0.0): Target tensor-to-scalar ratio
- **r_sigma** (default 0.014): Tolerance for r
- **N_obs** (default 60.0): Number of e-folds at horizon crossing

## Operator Selection

**binary_operators** (required): Available `["+", "-", "*", "/", "^"]`
**unary_operators** (optional): Available `["exp", "log", "sqrt", "sin", "cos", "square", "cube", "neg", "tanh"]`

Principles:
- Always include `["+", "*"]` as base
- Use either `^` OR `["square", "cube"]` for powers (not both)
- Start with common operators like `["+", "*", "^"]` or `["+", "*", "^", "exp"]`
- `tanh` and other exotic operators: include only when specifically needed, and assign higher complexity cost
- Each additional operator increases search space; balance expressiveness vs efficiency

## Complexity Control

**maxsize**: Expression tree size limit (typical: 12-30)
- Lower → simpler, faster; Higher → more expressive

**constraints**: Limit operator argument complexity
- Format: `{operator: [arg1_max, arg2_max]}` or `{operator: max_complexity}`
- Use JSON array `[a, b]` for tuple constraints
- Example: `{"^": [-1, 1]}` limits exponent complexity to 1
- Example: `{"/": [-1, 3]}` limits denominator complexity to 3

**nested_constraints**: Forbid operator nesting
- Format: `{outer_op: {inner_op: max_depth}}`
- Example: `{"exp": {"exp": 0}}` prevents `exp(exp(x))`

**complexity_of_operators**: Assign cost to operators (default: 1)
- Example: `{"exp": 2, "tanh": 4}` makes tanh expressions more costly

### Configuration Principles

1. Only reference operators included in binary_operators/unary_operators
2. **Always constrain `^`** when included: `{"^": [-1, 1]}`
3. **Always constrain `/`** when included: `{"/": [-1, 3]}`
4. **Always limit nesting for complex unary operators** unless necessary
5. **Assign higher complexity cost to exotic operators** like tanh when included

## Evolution Parameters

- **populations** (default 31): Parallel search populations (typical: 15-50)
- **niterations** (default 40): Evolution cycles (typical: 20-60)
- **population_size** (default 27): Individuals per population

## Example Config

```json
{
  "ns_target": 0.9649,
  "ns_sigma": 0.0042,
  "r_target": 0.0,
  "r_sigma": 0.014,
  "N_obs": 60.0,
  "binary_operators": ["+", "*", "^"],
  "unary_operators": ["exp"],
  "constraints": {"^": [-1, 1]},
  "nested_constraints": {"exp": {"exp": 0}},
  "maxsize": 15,
  "populations": 25,
  "niterations": 35
}
```
Adapt to actual requirements. Do not copy blindly.

# POST-PROCESSING

When `search_potential` returns candidates:
- Select top 3-5 by: lowest loss, interpretability, structural diversity
- Report each candidate with: expression, ns, r, loss

If `search_potential` returns no valid candidates or all have high loss:
- Report failure clearly with the config used
- Do not invent or fabricate results

# OUTPUT FORMAT

Return in this format:
```
**Search Config**: ns={ns_target}±{ns_sigma}, r={r_target}±{r_sigma}, N={N_obs}
- Operators: {binary_operators} + {unary_operators}
- Constraints: {summary}

**Results**:
1. V(φ) = {expression} → ns={ns}, r={r}, loss={loss}
2. ...
```
"""


# ============================================================================
# Tool Schemas and UI Helpers
# ============================================================================

MAIN_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_potential",
            "description": "Compute ns, r, A_s for a concrete inflation potential.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Potential V(phi) using only phi and concrete numbers.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_potential",
            "description": "Generate a 3-panel diagnostic plot for a concrete inflation potential.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Potential V(phi) using only phi and concrete numbers.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output path for the PNG file.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_encyclopedia",
            "description": "Look up inflation models and cosmology concepts in the encyclopedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Plain English search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of parent documents to return.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sr_agent",
            "description": "Delegate model discovery to the dedicated symbolic regression agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A concise description of the target observables and constraints.",
                    }
                },
                "required": ["task"],
            },
        },
    },
]

SR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_potential",
            "description": "Run PySR symbolic regression to discover inflation potentials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "config_json": {
                        "type": "string",
                        "description": "A complete JSON config for search_potential.",
                    }
                },
                "required": ["config_json"],
            },
        },
    }
]

TOOL_DISPLAY_CONFIG = {
    "analyze_potential": ("🔬", "Analyzing", False),
    "plot_potential": ("📊", "Plotting", False),
    "search_encyclopedia": ("📚", "Encyclopedia", False),
    "run_sr_agent": ("🤝", "SR Agent", False),
    "search_potential": ("🧬", "Symbolic Regression", True),
}

DEFAULT_MAIN_AGENT_MAX_STEPS = 8
DEFAULT_SR_AGENT_MAX_STEPS = 4


# ============================================================================
# Basic Supporting Classes
# ============================================================================


class SessionStore:
    """Store only the main user/assistant conversation in SQLite."""

    def __init__(self, db_path: str = "tmp/agent_storage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def append(self, session_id: str, role: str, content: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversation_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )

    def load(self, session_id: str, limit_turns: int = 5) -> list[dict]:
        limit = max(1, limit_turns) * 2
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]


class Agent:
    """A minimal chat-completions wrapper for one agent prompt."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        tools: list[dict] | None = None,
        temperature: float = 1.0,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.temperature = temperature

    async def complete(self, messages: list[dict]):
        request_messages = [{"role": "system", "content": self.system_prompt}, *messages]
        kwargs = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.temperature,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    @staticmethod
    def text_from_message(message) -> str:
        """Normalize SDK message content to plain text."""
        content = message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif getattr(item, "type", None) == "text":
                    parts.append(getattr(item, "text", ""))
            return "".join(parts)
        return str(content)

    @staticmethod
    def assistant_message(message) -> dict:
        """Convert an SDK assistant message back into chat-completions input format."""
        result = {"role": "assistant", "content": Agent.text_from_message(message)}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        return result


# ============================================================================
# DeepInflation Agent
# ============================================================================


class DeepInflation:
    """Inflation cosmology assistant with a dedicated SR sub-agent."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-5.2",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 1.0,
        main_agent_max_steps: int = DEFAULT_MAIN_AGENT_MAX_STEPS,
        sr_agent_max_steps: int = DEFAULT_SR_AGENT_MAX_STEPS,
        verbose: bool = True,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key.")

        self._base_url = base_url or os.getenv("BASE_URL") or None
        self.verbose = verbose
        self.last_plot_path: str | None = None
        self.session_id = str(uuid4())
        self.main_agent_max_steps = max(1, main_agent_max_steps)
        self.sr_agent_max_steps = max(1, sr_agent_max_steps)
        self._store = SessionStore()
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

        # Keep the existing behavior: one main agent and one SR sub-agent.
        from . import encyclopedia_rag as rag_module
        from . import sr_search as sr_module
        from . import tools as tools_module

        tools_module.VERBOSE = rag_module.VERBOSE = sr_module.VERBOSE = verbose

        # Initialize the encyclopedia once at startup.
        init_rag(
            api_key=self._api_key,
            base_url=self._base_url,
            embedding_model=embedding_model,
        )

        self.main_agent = Agent(
            client=self._client,
            model=model,
            system_prompt=MAIN_AGENT_PROMPT,
            tools=MAIN_TOOL_SCHEMAS,
            temperature=temperature,
        )
        self.sr_agent = Agent(
            client=self._client,
            model=model,
            system_prompt=SR_AGENT_PROMPT,
            tools=SR_TOOL_SCHEMAS,
            temperature=temperature,
        )

        self._log(f"[Agent] Initializing with model={model}, base_url={self._base_url or 'default'}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _format_tool_info(self, tool_name: str) -> dict:
        """Format tool metadata for the UI."""
        emoji, title, long_running = TOOL_DISPLAY_CONFIG.get(tool_name, ("🔧", tool_name, False))
        info = {"tool_name": tool_name, "title": f"{emoji} {title}", "log": ""}
        if long_running:
            info["long_running"] = True
        return info

    def _extract_sr_config(self, config_json_str: str) -> dict:
        """Extract a short SR config summary for the UI."""
        try:
            config = json.loads(config_json_str)
        except (json.JSONDecodeError, TypeError):
            return {}

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
            "estimated_time": f"~{max(1, iterations // 10)} min",
        }

    async def _run_tool(self, tool_name: str, tool_args: dict) -> str:
        """Run one local tool and keep small UI state in sync."""
        if tool_name == "analyze_potential":
            return analyze_potential(tool_args["expression"])

        if tool_name == "plot_potential":
            output_path = tool_args.get("output_path", "./potential_plot.png")
            result = plot_potential(tool_args["expression"], output_path)
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                parsed = None

            if isinstance(parsed, dict) and parsed.get("success", True):
                path = parsed.get("plot_path")
                if path and os.path.exists(path):
                    self.last_plot_path = path
            return result

        if tool_name == "search_encyclopedia":
            return search_encyclopedia(tool_args["query"], tool_args.get("top_k", 3))

        raise ValueError(f"Unknown tool: {tool_name}")

    async def _run_sr_agent_stream(self, task: str):
        """Run the dedicated SR sub-agent in its own short context."""
        messages = [{"role": "user", "content": task}]

        # The SR sub-agent gets its own short loop and context. It is responsible
        # for configuring PySR once, running the search once, then summarizing the
        # result back to the main agent.
        for _ in range(self.sr_agent_max_steps):
            message = await self.sr_agent.complete(messages)
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                yield {"type": "sr_result", "content": self.sr_agent.text_from_message(message).strip()}
                return

            messages.append(self.sr_agent.assistant_message(message))

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid tool arguments: {exc}") from exc
                call_id = f"sr_{tool_name}_{time.time()}"
                start_time = time.time()

                self._log(f"[SR Tool] {tool_name} input={str(tool_args)[:500]}")
                yield {
                    "type": "tool_start",
                    "call_id": call_id,
                    "info": self._format_tool_info(tool_name),
                    "args": tool_args,
                }

                if tool_name == "search_potential":
                    sr_config = self._extract_sr_config(tool_args.get("config_json", "{}"))
                    if sr_config:
                        yield {"type": "sr_config", "config": sr_config}
                    result = search_potential(tool_args.get("config_json", "{}"))
                else:
                    result = json.dumps({"success": False, "error": f"Unknown SR tool: {tool_name}"})

                duration = time.time() - start_time
                try:
                    parsed = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                success = parsed.get("success", True) if isinstance(parsed, dict) else True
                self._log(f"[SR Tool] {tool_name} {'ok' if success else 'failed'} ({duration:.1f}s)")

                yield {
                    "type": "tool_end",
                    "call_id": call_id,
                    "duration": duration,
                    "success": success,
                }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        yield {
            "type": "sr_result",
            "content": "Symbolic regression agent reached the step limit before finishing.",
        }

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

        try:
            # Only keep recent main-dialogue turns in the main agent context.
            messages = [*self._store.load(self.session_id, limit_turns=5), {"role": "user", "content": question}]
            self._store.append(self.session_id, "user", question)

            # Main agent loop: ask the model what to do next, run any requested
            # tools, feed tool results back, and stop as soon as a final answer is
            # produced. The loop limit stays configurable to avoid a hard-coded
            # retry count while still preventing runaway tool recursion.
            for _ in range(self.main_agent_max_steps):
                message = await self.main_agent.complete(messages)
                tool_calls = list(message.tool_calls or [])

                if not tool_calls:
                    answer = self.main_agent.text_from_message(message).strip()
                    if answer:
                        yield {"type": "text_delta", "delta": answer}
                    self._store.append(self.session_id, "assistant", answer)
                    yield {"type": "response", "content": answer, "plot_path": self.last_plot_path}
                    return

                messages.append(self.main_agent.assistant_message(message))

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid tool arguments: {exc}") from exc
                    call_id = f"main_{tool_name}_{time.time()}"
                    start_time = time.time()

                    self._log(f"[Tool] {tool_name} input={str(tool_args)[:500]}")
                    yield {
                        "type": "tool_start",
                        "call_id": call_id,
                        "info": self._format_tool_info(tool_name),
                        "args": tool_args,
                    }

                    if tool_name == "run_sr_agent":
                        result = ""
                        async for event in self._run_sr_agent_stream(tool_args["task"]):
                            if event["type"] == "sr_result":
                                result = event["content"]
                            else:
                                yield event
                    else:
                        result = await self._run_tool(tool_name, tool_args)

                    duration = time.time() - start_time
                    try:
                        parsed = json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        parsed = None
                    success = parsed.get("success", True) if isinstance(parsed, dict) else True
                    self._log(f"[Tool] {tool_name} {'ok' if success else 'failed'} ({duration:.1f}s)")

                    yield {
                        "type": "tool_end",
                        "call_id": call_id,
                        "duration": duration,
                        "success": success,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

            fallback = "The agent reached the tool loop limit before producing a final answer."
            self._store.append(self.session_id, "assistant", fallback)
            yield {"type": "response", "content": fallback, "plot_path": self.last_plot_path}

        except Exception as exc:
            self._log(f"[Agent] Error: {exc}")
            yield {"type": "response", "content": f"Error: {exc}", "plot_path": None}

    def run(self, question: str) -> str:
        """Synchronous helper that returns only the final answer."""
        import asyncio

        async def get_response() -> str:
            async for event in self.stream(question):
                if event["type"] == "response":
                    return event["content"]
            return ""

        return asyncio.run(get_response())

    def clear_history(self):
        """Clear conversation history by creating a new session."""
        self.session_id = str(uuid4())
        self.last_plot_path = None
        self._log("[Agent] History cleared (new session)")


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
    print("  - What is ns for V = phi^2?")
    print("  - Plot the Starobinsky model: (1 - exp(-sqrt(2/3)*phi))^2")
    print("  - Find a plateau potential with r < 0.01")
    print("\nType 'quit' to exit, 'clear' to reset conversation\n")

    async def main():
        agent = DeepInflation(verbose=False)
        pending_tools = {}

        while True:
            try:
                question = input("> ").strip()
                if not question:
                    continue
                if question.lower() in {"quit", "exit", "q"}:
                    print("Goodbye!")
                    break
                if question.lower() == "clear":
                    agent.clear_history()
                    pending_tools.clear()
                    continue

                print("\nThinking...")

                async for event in agent.stream(question):
                    if event["type"] == "tool_start":
                        title = event["info"].get("title", "Tool")
                        pending_tools[event["call_id"]] = title
                        print(f"  {title}...")
                    elif event["type"] == "sr_config":
                        config = event["config"]
                        print(
                            f"    Config: ns={config['ns_target']}+-{config['ns_sigma']}, "
                            f"r={config['r_target']}+-{config['r_sigma']}"
                        )
                        print(f"    Ops: {config['operators']}")
                        print(f"    Est: {config['estimated_time']}")
                    elif event["type"] == "tool_end":
                        title = pending_tools.pop(event["call_id"], "Tool")
                        print(f"  {title} done ({event.get('duration', 0):.1f}s)")
                    elif event["type"] == "response":
                        print(f"\n{event['content']}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as exc:
                print(f"Error: {exc}\n")

    asyncio.run(main())
