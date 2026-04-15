"""Interactive CLI for the AI Research Assistant."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .client import AssistantClient, ConversationSession
from .tools import build_default_registry

console = Console()

SYSTEM_PROMPT = """\
You are an expert research assistant. You help users explore topics deeply, \
summarise information clearly, and produce well-structured reports. \
When asked to research a topic, break your answer into clear sections with headings. \
You have access to tools: use web_search for current information, calculator for maths, \
and get_current_datetime for the current date/time. \
Always cite limitations in your knowledge where relevant.\
"""

HELP_TEXT = """
[bold]Commands[/bold]
  /help     Show this message
  /clear    Clear conversation history
  /tools    List available tools
  /usage    Show token usage for this session
  /quit     Exit
"""


def _print_welcome(tool_names: list[str]) -> None:
    tools_str = "  ".join(f"[cyan]{t}[/cyan]" for t in tool_names)
    console.print(
        Panel(
            Text("AI Research Assistant", justify="center", style="bold cyan"),
            subtitle=f"Tools: {tools_str}  |  Type [bold]/help[/bold] for commands",
            border_style="cyan",
        )
    )


def _on_tool_call(name: str, inputs: dict, result: str) -> None:
    """Display a compact summary of each tool call."""
    short_result = result[:120].replace("\n", " ")
    if len(result) > 120:
        short_result += "…"
    input_summary = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
    console.print(
        f"  [dim][tool][/dim] [yellow]{name}[/yellow]({input_summary}) "
        f"→ [dim]{short_result}[/dim]"
    )


def _run_turn(client: AssistantClient, session: ConversationSession) -> None:
    """Execute one user turn using the agentic tool loop."""
    from .tools import build_default_registry

    registry = build_default_registry()

    console.print()
    console.rule("[dim]Assistant[/dim]", style="dim")

    for chunk in client.run_with_tools(session, registry, on_tool_call=_on_tool_call):
        console.print(chunk, end="", highlight=False)

    console.print("\n")
    console.rule(style="dim")


def run_repl() -> None:
    client = AssistantClient()
    session = ConversationSession(system_prompt=SYSTEM_PROMPT)
    registry = build_default_registry()

    _print_welcome(registry.names)

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye.[/dim]")
                break
            elif cmd == "/help":
                console.print(HELP_TEXT)
            elif cmd == "/clear":
                session.clear()
                console.print("[dim]Conversation history cleared.[/dim]")
            elif cmd == "/tools":
                console.print("[bold]Available tools:[/bold]")
                for defn in registry.definitions:
                    console.print(f"  [cyan]{defn['name']}[/cyan] — {defn['description'][:80]}…")
            elif cmd == "/usage":
                console.print(f"[dim]{session.usage}[/dim]")
            else:
                console.print(f"[red]Unknown command:[/red] {user_input}")
            continue

        # --- Normal message ---
        session.add_user(user_input)
        try:
            _run_turn(client, session)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
            session.history.pop()


def main() -> None:
    run_repl()


if __name__ == "__main__":
    main()
