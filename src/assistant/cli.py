"""Interactive CLI for the AI Research Assistant."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from .client import AssistantClient, ConversationSession

console = Console()

SYSTEM_PROMPT = """\
You are an expert research assistant. You help users explore topics deeply, \
summarise information clearly, and produce well-structured reports. \
When asked to research a topic, break your answer into clear sections with headings. \
Always cite limitations in your knowledge where relevant.\
"""

HELP_TEXT = """
[bold]Commands[/bold]
  /help     Show this message
  /clear    Clear conversation history
  /usage    Show token usage for this session
  /quit     Exit
"""


def _print_welcome() -> None:
    console.print(
        Panel(
            Text("AI Research Assistant", justify="center", style="bold cyan"),
            subtitle="Type [bold]/help[/bold] for commands",
            border_style="cyan",
        )
    )


def _stream_response(client: AssistantClient, session: ConversationSession) -> None:
    """Stream the assistant reply and render it as Markdown when done."""
    console.print()
    console.rule("[dim]Assistant[/dim]", style="dim")

    full_text = ""
    # Print raw streaming text so the user sees output immediately
    with console.status("", spinner="dots"):
        pass  # Clear status before streaming

    for chunk in client.chat_stream(session):
        console.print(chunk, end="", highlight=False)
        full_text += chunk

    # Re-render the final response as formatted Markdown
    console.print("\n")
    console.rule(style="dim")


def run_repl() -> None:
    client = AssistantClient()
    session = ConversationSession(system_prompt=SYSTEM_PROMPT)

    _print_welcome()

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
            elif cmd == "/usage":
                console.print(f"[dim]{session.usage}[/dim]")
            else:
                console.print(f"[red]Unknown command:[/red] {user_input}")
            continue

        # --- Normal message ---
        session.add_user(user_input)
        try:
            _stream_response(client, session)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {exc}")
            # Remove the failed user message so history stays clean
            session.history.pop()


def main() -> None:
    run_repl()


if __name__ == "__main__":
    main()
