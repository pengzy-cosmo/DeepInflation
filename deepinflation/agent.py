"""Inflation research agent with a small handwritten agent loop."""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI

from .encyclopedia_rag import init_rag, search_encyclopedia
from .prompts import (
    MAIN_AGENT_MAX_STEPS,
    MAIN_AGENT_PROMPT,
    MAIN_TOOL_SCHEMAS,
    SR_AGENT_MAX_STEPS,
    SR_AGENT_PROMPT,
    SR_TOOL_SCHEMAS,
    TOOL_DISPLAY_CONFIG,
)
from .sr_search import search_potential
from .tools import analyze_potential, plot_potential

logger = logging.getLogger(__name__)


def _parse_success(result: str) -> bool:
    """Return the 'success' field from a JSON tool result, defaulting to True."""
    try:
        parsed = json.loads(result)
        return parsed.get("success", True) if isinstance(parsed, dict) else True
    except (json.JSONDecodeError, TypeError):
        return True


# ============================================================================
# Session Storage
# ============================================================================


class SessionStore:
    """Persist the main user/assistant conversation in SQLite (tool calls excluded)."""

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
        """Return the last `limit_turns` turns (1 turn = 1 user + 1 assistant message)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit_turns * 2),
            ).fetchall()

        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]


# ============================================================================
# Agent
# ============================================================================


class Agent:
    """Thin wrapper around one OpenAI chat-completions call."""

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
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "temperature": self.temperature,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    @staticmethod
    def text_from_message(message) -> str:
        """Extract plain text from an SDK message (handles str, list, and None content)."""
        content = message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                for item in content
                if (isinstance(item, dict) and item.get("type") == "text") or getattr(item, "type", None) == "text"
            )
        return str(content)

    @staticmethod
    def assistant_message(message) -> dict:
        """Serialize an SDK assistant message into the chat-completions history format."""
        msg: dict = {"role": "assistant", "content": Agent.text_from_message(message)}
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                }
                for tool_call in message.tool_calls
            ]
        return msg


# ============================================================================
# DeepInflation
# ============================================================================


class DeepInflation:
    """Inflation cosmology assistant with a dedicated SR sub-agent.

    Streams structured events from `stream()`:
        tool_start  {"call_id": str, "info": dict, "args": dict}
        tool_end    {"call_id": str, "duration": float, "success": bool}
        sr_config   {"config": dict}
        text_delta  {"delta": str}
        response    {"content": str, "plot_path": str | None}
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-5.2",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 1.0,
        main_agent_max_steps: int = MAIN_AGENT_MAX_STEPS,
        sr_agent_max_steps: int = SR_AGENT_MAX_STEPS,
        verbose: bool = True,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key.")

        self._base_url = base_url or os.getenv("BASE_URL") or None
        self.last_plot_path: str | None = None
        self.session_id = str(uuid4())
        self.main_agent_max_steps = max(1, main_agent_max_steps)
        self.sr_agent_max_steps = max(1, sr_agent_max_steps)
        self._store = SessionStore()
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

        # Set log level for the entire deepinflation package.
        logging.getLogger("deepinflation").setLevel(logging.DEBUG if verbose else logging.WARNING)

        init_rag(api_key=self._api_key, base_url=self._base_url, embedding_model=embedding_model)

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

        logger.debug("Initialized: model=%s base_url=%s", model, self._base_url or "default")

    def _format_tool_info(self, tool_name: str) -> dict:
        """Build the tool info dict consumed by the UI."""
        emoji, title, long_running = TOOL_DISPLAY_CONFIG.get(tool_name, ("🔧", tool_name, False))
        info = {"tool_name": tool_name, "title": f"{emoji} {title}", "log": ""}
        if long_running:
            info["long_running"] = True
        return info

    def _sr_config_summary(self, config_json_str: str) -> dict:
        """Parse a PySR config JSON string into a UI-friendly summary dict."""
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
        if tool_name == "analyze_potential":
            return analyze_potential(tool_args["expression"])

        if tool_name == "plot_potential":
            result = plot_potential(tool_args["expression"], tool_args.get("output_path", "./potential_plot.png"))
            try:
                parsed = json.loads(result)
                path = parsed.get("plot_path") if isinstance(parsed, dict) else None
                if path and os.path.exists(path):
                    self.last_plot_path = path
            except (json.JSONDecodeError, TypeError):
                pass
            return result

        if tool_name == "search_encyclopedia":
            return search_encyclopedia(tool_args["query"], tool_args.get("top_k", 3))

        raise ValueError(f"Unknown tool: {tool_name}")

    async def _run_sr_agent(self, task: str):
        """Run the SR sub-agent loop in its own isolated context."""
        messages = [{"role": "user", "content": task}]

        # SR sub-agent: configure PySR once, run once, summarize results.
        for _ in range(self.sr_agent_max_steps):
            message = await self.sr_agent.complete(messages)
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                yield {"type": "sr_result", "content": Agent.text_from_message(message).strip()}
                return

            messages.append(Agent.assistant_message(message))

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid tool arguments: {exc}") from exc

                call_id = f"sr_{tool_name}_{time.time()}"
                start_time = time.time()

                logger.debug("[SR] %s args=%s", tool_name, str(tool_args)[:500])
                yield {
                    "type": "tool_start",
                    "call_id": call_id,
                    "info": self._format_tool_info(tool_name),
                    "args": tool_args,
                }

                if tool_name == "search_potential":
                    config_summary = self._sr_config_summary(tool_args.get("config_json", "{}"))
                    if config_summary:
                        yield {"type": "sr_config", "config": config_summary}
                    result = search_potential(tool_args.get("config_json", "{}"))
                else:
                    result = json.dumps({"success": False, "error": f"Unknown SR tool: {tool_name}"})

                duration = time.time() - start_time
                success = _parse_success(result)
                logger.debug("[SR] %s %s (%.1fs)", tool_name, "ok" if success else "failed", duration)
                yield {"type": "tool_end", "call_id": call_id, "duration": duration, "success": success}
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        yield {"type": "sr_result", "content": "Symbolic regression agent reached the step limit."}

    async def stream(self, question: str):
        """Async generator yielding structured events for Gradio."""
        self.last_plot_path = None

        try:
            messages = [*self._store.load(self.session_id), {"role": "user", "content": question}]
            self._store.append(self.session_id, "user", question)

            # Main loop: call model → run tools → repeat until final answer or step limit.
            for _ in range(self.main_agent_max_steps):
                message = await self.main_agent.complete(messages)
                tool_calls = list(message.tool_calls or [])

                if not tool_calls:
                    answer = Agent.text_from_message(message).strip()
                    if answer:
                        yield {"type": "text_delta", "delta": answer}
                    self._store.append(self.session_id, "assistant", answer)
                    yield {"type": "response", "content": answer, "plot_path": self.last_plot_path}
                    return

                messages.append(Agent.assistant_message(message))

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid tool arguments: {exc}") from exc

                    call_id = f"main_{tool_name}_{time.time()}"
                    start_time = time.time()

                    logger.debug("[Tool] %s args=%s", tool_name, str(tool_args)[:500])
                    yield {
                        "type": "tool_start",
                        "call_id": call_id,
                        "info": self._format_tool_info(tool_name),
                        "args": tool_args,
                    }

                    if tool_name == "run_sr_agent":
                        result = ""
                        async for event in self._run_sr_agent(tool_args["task"]):
                            if event["type"] == "sr_result":
                                result = event["content"]
                            else:
                                yield event
                    else:
                        result = await self._run_tool(tool_name, tool_args)

                    duration = time.time() - start_time
                    success = _parse_success(result)
                    logger.debug("[Tool] %s %s (%.1fs)", tool_name, "ok" if success else "failed", duration)
                    yield {"type": "tool_end", "call_id": call_id, "duration": duration, "success": success}
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

            fallback = "The agent reached the tool loop limit before producing a final answer."
            self._store.append(self.session_id, "assistant", fallback)
            yield {"type": "response", "content": fallback, "plot_path": self.last_plot_path}

        except Exception as exc:
            logger.error("Agent error: %s", exc)
            yield {"type": "response", "content": f"Error: {exc}", "plot_path": None}

    def run(self, question: str) -> str:
        """Synchronous helper that drives `stream()` and returns only the final answer."""
        import asyncio

        async def _get() -> str:
            async for event in self.stream(question):
                if event["type"] == "response":
                    return event["content"]
            return ""

        return asyncio.run(_get())

    def clear_history(self):
        """Reset conversation by starting a new session."""
        self.session_id = str(uuid4())
        self.last_plot_path = None
        logger.debug("History cleared (new session)")


# ============================================================================
# CLI
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
        pending_tools: dict[str, str] = {}

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
