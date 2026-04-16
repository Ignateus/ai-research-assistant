"""Interactive CLI for the AI Research Assistant."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .client import AssistantClient, ConversationSession
from .rag import RAGPipeline
from .tools import build_default_registry

console = Console()

SYSTEM_PROMPT = """\
You are an expert research assistant. You help users explore topics deeply, \
summarise information clearly, and produce well-structured reports. \
When asked to research a topic, break your answer into clear sections with headings. \
You have access to tools: use web_search for current information, calculator for maths, \
get_current_datetime for the current date/time, and search_documents to query any \
documents the user has loaded. Always cite limitations in your knowledge where relevant.\
"""

HELP_TEXT = """
[bold]Commands[/bold]
  /help              Show this message
  /clear             Clear conversation history
  /tools             List available tools
  /ingest <path>     Load a file or directory into the document store
  /sources           List all ingested documents
  /cleardb           Wipe the document store
  /usage             Show token usage for this session
  /quit              Exit
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
    """Display a compact inline summary of each tool call."""
    short_result = result[:120].replace("\n", " ")
    if len(result) > 120:
        short_result += "…"
    input_summary = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
    console.print(
        f"  [dim][tool][/dim] [yellow]{name}[/yellow]({input_summary}) "
        f"→ [dim]{short_result}[/dim]"
    )


def _run_turn(
    client: AssistantClient,
    session: ConversationSession,
    pipeline: RAGPipeline,
) -> None:
    """Execute one user turn using the agentic tool loop."""
    registry = build_default_registry(pipeline=pipeline)

    console.print()
    console.rule("[dim]Assistant[/dim]", style="dim")

    for chunk in client.run_with_tools(session, registry, on_tool_call=_on_tool_call):
        console.print(chunk, end="", highlight=False)

    console.print("\n")
    console.rule(style="dim")


def run_repl() -> None:
    client = AssistantClient()
    session = ConversationSession(system_prompt=SYSTEM_PROMPT)
    pipeline = RAGPipeline()
    registry = build_default_registry(pipeline=pipeline)

    _print_welcome(registry.names)

    # Show document count if store already has data from a previous session
    if pipeline.document_count > 0:
        console.print(
            f"[dim]Document store loaded — {pipeline.document_count} chunks "
            f"from {len(pipeline.list_sources())} file(s).[/dim]"
        )

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
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

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

            elif cmd == "/ingest":
                if not arg:
                    console.print("[red]Usage:[/red] /ingest <file_or_directory>")
                else:
                    path = Path(arg.strip())
                    with console.status(f"[dim]Ingesting {path}…[/dim]"):
                        try:
                            if path.is_dir():
                                result = pipeline.ingest_directory(path)
                            else:
                                result = pipeline.ingest_file(path)
                            console.print(f"[green]✓[/green] {result}")
                        except Exception as exc:  # noqa: BLE001
                            console.print(f"[red]Ingest error:[/red] {exc}")

            elif cmd == "/sources":
                sources = pipeline.list_sources()
                if not sources:
                    console.print("[dim]No documents ingested yet.[/dim]")
                else:
                    console.print(f"[bold]{len(sources)} source(s) in document store:[/bold]")
                    for s in sources:
                        console.print(f"  [dim]{s}[/dim]")

            elif cmd == "/cleardb":
                pipeline.clear()
                console.print("[dim]Document store cleared.[/dim]")

            elif cmd == "/usage":
                console.print(f"[dim]{session.usage}[/dim]")

            else:
                console.print(f"[red]Unknown command:[/red] {parts[0]}")
            continue

        # --- Normal message ---
        session.add_user(user_input)
        try:
            _run_turn(client, session, pipeline)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
            session.history.pop()


def main() -> None:
    run_repl()


if __name__ == "__main__":
    main()
