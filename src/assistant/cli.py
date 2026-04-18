"""Interactive CLI for the AI Research Assistant."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .agent import EventType, run_research_agent
from .client import AssistantClient, ConversationSession
from .cost import calculate_cost, format_cost
from .config import config
from .eval import EvalDataset, HarnessEvent, run_eval, print_summary, print_detail, save_report
from .memory import (
    default_session_path,
    history_token_count,
    list_sessions,
    load_session,
    save_session,
    should_summarize,
    summarize_history,
)
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
  /research <goal>   Run the full agent loop on a research goal
  /eval [path]       Run the eval harness (default dataset if no path given)
  /ingest <path>     Load a file or directory into the document store
  /sources           List all ingested documents
  /cleardb           Wipe the document store
  /memory            Show memory stats (tokens used, message count)
  /save [name]       Save session to disk
  /load [path]       Load a saved session (shows list if no path given)
  /usage             Show token usage for this session
  /quit              Exit
"""

# Auto-summarize when history exceeds this many tokens
SUMMARIZE_THRESHOLD = 6_000


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
    """Execute one user turn: auto-summarize if needed, then run the agentic loop."""
    # Auto-summarize before sending if history is getting long
    if should_summarize(session, threshold=SUMMARIZE_THRESHOLD):
        with console.status("[dim]Compressing conversation history…[/dim]"):
            compressed = summarize_history(session, client, threshold=SUMMARIZE_THRESHOLD)
        if compressed:
            console.print(
                f"  [dim][memory] Summarized {compressed} older messages to save context.[/dim]"
            )

    registry = build_default_registry(pipeline=pipeline)

    console.print()
    console.rule("[dim]Assistant[/dim]", style="dim")

    for chunk in client.run_with_tools(session, registry, on_tool_call=_on_tool_call):
        console.print(chunk, end="", highlight=False)

    console.print("\n")
    console.rule(style="dim")


def _run_research(goal: str, client: AssistantClient, pipeline: RAGPipeline) -> None:
    """Drive the full agent loop and render live progress to the console."""
    from rich.markdown import Markdown

    registry = build_default_registry(pipeline=pipeline)

    console.print()
    console.print(
        Panel(f"[bold]Research goal:[/bold] {goal}", border_style="cyan", expand=False)
    )

    agent_gen = run_research_agent(goal, client, registry)
    report = None

    try:
        while True:
            event = next(agent_gen)

            if event.type == EventType.PLAN_CREATED:
                plan = event.data
                console.print(f"\n[bold cyan]Plan[/bold cyan] ({len(plan.steps)} steps)")
                for step in plan.steps:
                    console.print(f"  [dim]{step}[/dim]")

            elif event.type == EventType.STEP_STARTED:
                step = event.data
                console.print(f"\n[yellow]▶ Step {step.index}:[/yellow] {step.description}")

            elif event.type == EventType.STEP_COMPLETED:
                result = event.data
                preview = result.findings[:120].replace("\n", " ")
                if len(result.findings) > 120:
                    preview += "…"
                console.print(f"  [green]✓[/green] [dim]{preview}[/dim]")

            elif event.type == EventType.REFLECTION:
                reflection = event.data
                if reflection.sufficient:
                    console.print("\n  [green]Reflection: findings are sufficient.[/green]")
                else:
                    console.print(
                        f"\n  [yellow]Reflection: {len(reflection.gaps)} gap(s) found.[/yellow]"
                    )
                    for gap in reflection.gaps:
                        console.print(f"    [dim]• {gap}[/dim]")

            elif event.type == EventType.EXTRA_STEPS_ADDED:
                extra = event.data
                console.print(
                    f"  [yellow]Adding {len(extra)} extra step(s) to fill gaps…[/yellow]"
                )

            elif event.type == EventType.REPORT_READY:
                report = event.data

    except StopIteration as e:
        agent_result = e.value
        report = agent_result.report
        console.print(
            f"\n[dim]Agent finished — {agent_result.total_steps_executed} steps executed.[/dim]"
        )

    if report:
        console.print()
        console.rule("[bold cyan]Report[/bold cyan]", style="cyan")
        console.print(Markdown(report))
        console.rule(style="cyan")


def _run_eval(dataset_path: Path, client: AssistantClient, pipeline: RAGPipeline) -> None:
    """Run the eval harness against a dataset and display results."""
    from datetime import datetime

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found:[/red] {dataset_path}")
        return

    dataset = EvalDataset.load(dataset_path)
    registry = build_default_registry(pipeline=pipeline)

    console.print()
    console.print(
        Panel(
            f"[bold]Evaluating:[/bold] {dataset.name}  "
            f"([dim]{len(dataset)} questions[/dim])",
            border_style="cyan",
            expand=False,
        )
    )

    result = None
    gen = run_eval(dataset, client, client, registry)

    try:
        while True:
            event: HarnessEvent = next(gen)
            score_str = "  ".join(
                f"[{'green' if s.score >= 4 else 'yellow' if s.score >= 3 else 'red'}]{s.metric.value[0].upper()}:{s.score}[/]"
                for s in event.eval_score.scores
            )
            console.print(
                f"  [dim]{event.question.id}[/dim]  {score_str}  "
                f"[dim]mean={event.eval_score.mean:.1f}  ({event.elapsed_seconds:.1f}s)[/dim]"
            )
    except StopIteration as e:
        result = e.value

    if result:
        print_summary(result)

        # Auto-save report
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"data/eval/reports/{dataset.name}_{ts}.json")
        save_report(result, report_path)

        console.print("[dim]Use /eval to re-run or pass a different dataset path.[/dim]")


def _show_memory(session: ConversationSession) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[dim]Messages[/dim]", str(len(session.history)))
    table.add_row("[dim]History tokens (approx)[/dim]", str(history_token_count(session)))
    table.add_row("[dim]Summarize threshold[/dim]", str(SUMMARIZE_THRESHOLD))
    table.add_row("[dim]Cached context[/dim]", f"{len(session.cached_context)} chars")
    console.print(table)


def _show_sessions() -> None:
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No saved sessions found in data/sessions/[/dim]")
        return
    console.print("[bold]Saved sessions:[/bold]")
    for p in sessions:
        console.print(f"  [cyan]{p}[/cyan]")


def run_repl() -> None:
    client = AssistantClient()
    session = ConversationSession(system_prompt=SYSTEM_PROMPT)
    pipeline = RAGPipeline()
    registry = build_default_registry(pipeline=pipeline)

    _print_welcome(registry.names)

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
            arg = parts[1].strip() if len(parts) > 1 else ""

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

            elif cmd == "/research":
                if not arg:
                    console.print("[red]Usage:[/red] /research <goal>")
                else:
                    try:
                        _run_research(arg, client, pipeline)
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"[red]Agent error:[/red] {exc}")

            elif cmd == "/eval":
                dataset_path = Path(arg) if arg else Path("data/eval/research_assistant.json")
                try:
                    _run_eval(dataset_path, client, pipeline)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]Eval error:[/red] {exc}")

            elif cmd == "/ingest":
                if not arg:
                    console.print("[red]Usage:[/red] /ingest <file_or_directory>")
                else:
                    path = Path(arg)
                    with console.status(f"[dim]Ingesting {path}…[/dim]"):
                        try:
                            result = (
                                pipeline.ingest_directory(path)
                                if path.is_dir()
                                else pipeline.ingest_file(path)
                            )
                            # Update cached context with source list for the LLM
                            sources = pipeline.list_sources()
                            session.cached_context = (
                                f"The user has loaded {pipeline.document_count} document chunks "
                                f"from the following files:\n"
                                + "\n".join(f"- {s}" for s in sources)
                            )
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
                session.cached_context = ""
                console.print("[dim]Document store cleared.[/dim]")

            elif cmd == "/memory":
                _show_memory(session)

            elif cmd == "/save":
                path = default_session_path(arg or None)
                try:
                    save_session(session, path)
                    console.print(f"[green]✓[/green] Session saved to [cyan]{path}[/cyan]")
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]Save error:[/red] {exc}")

            elif cmd == "/load":
                if not arg:
                    _show_sessions()
                else:
                    try:
                        loaded = load_session(arg)
                        session.history = loaded.history
                        session.usage = loaded.usage
                        session.cached_context = loaded.cached_context
                        console.print(
                            f"[green]✓[/green] Loaded {len(session.history)} messages "
                            f"from [cyan]{arg}[/cyan]"
                        )
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"[red]Load error:[/red] {exc}")

            elif cmd == "/usage":
                breakdown = calculate_cost(session.usage, config.model)
                console.print(f"[dim]{session.usage}[/dim]")
                console.print(f"[dim]{breakdown}[/dim]")

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
