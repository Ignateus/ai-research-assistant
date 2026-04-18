"""
AI Research Assistant — Live Demo Script
=========================================

Run this script to walk through all key features of the project.
Each section pauses so you can explain what just happened.

Usage:
    python demo.py
"""

from __future__ import annotations

import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()


def pause(msg: str = "Press Enter to continue...") -> None:
    console.print(f"\n[dim]{msg}[/dim]")
    input()


def section(title: str, description: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")
    console.print(f"[dim]{description}[/dim]")
    console.print()


def main() -> None:
    console.print(
        Panel(
            Text("AI Research Assistant — Demo", justify="center", style="bold cyan"),
            subtitle="Python · Claude API · RAG · Agents",
            border_style="cyan",
        )
    )
    console.print(
        "\nThis demo walks through every feature built over the week.\n"
        "Each section shows a different capability of the system.\n"
    )
    pause("Ready? Press Enter to start...")

    # ── Import after env is confirmed loaded ────────────────────────────
    from src.assistant.client import AssistantClient, ConversationSession
    from src.assistant.tools import build_default_registry
    from src.assistant.rag import RAGPipeline
    from src.assistant.memory import history_token_count, save_session, default_session_path
    from src.assistant.agent import run_research_agent, EventType
    from src.assistant.cost import calculate_cost
    from src.assistant.config import config

    client = AssistantClient()
    pipeline = RAGPipeline()

    # ── 1. Basic streaming chat ──────────────────────────────────────────
    section(
        "1 · Streaming Chat",
        "Direct multi-turn conversation with Claude using the Anthropic SDK."
    )

    session = ConversationSession(
        system_prompt="You are an expert AI research assistant. Be concise."
    )
    session.add_user("In two sentences, what makes transformer models powerful?")

    console.print("[bold green]User:[/bold green] In two sentences, what makes transformer models powerful?\n")
    console.print("[bold blue]Assistant:[/bold blue] ", end="")

    for chunk in client.run_with_tools(session, build_default_registry(pipeline=pipeline)):
        console.print(chunk, end="", highlight=False)

    console.print("\n")
    pause()

    # ── 2. Tool use ──────────────────────────────────────────────────────
    section(
        "2 · Tool Use (Function Calling)",
        "The assistant autonomously decides which tools to call — calculator, search, datetime."
    )

    registry = build_default_registry(pipeline=pipeline)
    session2 = ConversationSession(
        system_prompt="You are a helpful assistant. Use tools when appropriate."
    )
    session2.add_user("What is 2 to the power of 24, and what day of the week is it today?")

    console.print("[bold green]User:[/bold green] What is 2 to the power of 24, and what day of the week is it today?\n")

    def show_tool(name, inputs, result):
        short = result[:80].replace("\n", " ")
        console.print(f"  [dim]→ tool:[/dim] [yellow]{name}[/yellow]({', '.join(f'{k}={v!r}' for k,v in inputs.items())}) = [dim]{short}[/dim]")

    console.print("[bold blue]Assistant:[/bold blue] ", end="")
    for chunk in client.run_with_tools(session2, registry, on_tool_call=show_tool):
        console.print(chunk, end="", highlight=False)
    console.print("\n")
    pause()

    # ── 3. RAG — Document ingestion & retrieval ──────────────────────────
    section(
        "3 · RAG — Document Ingestion & Retrieval",
        "Load documents into ChromaDB, then query them via semantic search."
    )

    console.print("Ingesting sample documents...")
    result = pipeline.ingest_directory("data/sample_docs")
    console.print(f"[green]✓[/green] {result}\n")

    query = "What chunking strategies should I use for a RAG pipeline?"
    context = pipeline.search_as_context(query, top_k=3)
    console.print(f"[bold]Query:[/bold] {query}\n")
    console.print("[bold]Retrieved context (top 3 chunks):[/bold]")
    console.print(f"[dim]{context[:600]}…[/dim]\n")
    pause()

    # ── 4. RAG-grounded answer ───────────────────────────────────────────
    section(
        "4 · RAG-Grounded Answer",
        "The assistant uses search_documents to answer from ingested content."
    )

    session3 = ConversationSession(
        system_prompt="You are a research assistant. Use search_documents to answer from loaded documents.",
        cached_context=f"The user has loaded documents: {', '.join(pipeline.list_sources())}",
    )
    session3.add_user("Summarise the main chunking strategies for RAG, based on the loaded documents.")
    console.print("[bold green]User:[/bold green] Summarise the main chunking strategies for RAG.\n")
    console.print("[bold blue]Assistant:[/bold blue] ", end="")
    for chunk in client.run_with_tools(session3, build_default_registry(pipeline=pipeline), on_tool_call=show_tool):
        console.print(chunk, end="", highlight=False)
    console.print("\n")
    pause()

    # ── 5. Memory — conversation summarisation ───────────────────────────
    section(
        "5 · Memory & Prompt Caching",
        "Token usage tracking with cost estimation. Cached system prompts reduce cost on repeat calls."
    )

    console.print(f"[bold]Tokens used so far (session 3):[/bold] {session3.usage}")
    from src.assistant.cost import calculate_cost
    breakdown = calculate_cost(session3.usage, config.model)
    console.print(f"[bold]Cost breakdown:[/bold]\n{breakdown}")
    console.print(f"\n[dim]Prompt caching: cache_write tokens are billed once, then re-reads cost 90% less.[/dim]")
    pause()

    # ── 6. Multi-step Agent ──────────────────────────────────────────────
    section(
        "6 · Multi-step Research Agent",
        "Plan → Execute → Reflect loop. Claude decomposes the goal, runs tools at each step, "
        "checks for gaps, and produces a structured report."
    )

    goal = "What are the key differences between RAG and fine-tuning for LLM customisation?"
    console.print(f"[bold]Research goal:[/bold] {goal}\n")

    registry_for_agent = build_default_registry(pipeline=pipeline)
    gen = run_research_agent(goal, client, registry_for_agent)
    agent_result = None

    try:
        while True:
            event = next(gen)
            if event.type == EventType.PLAN_CREATED:
                plan = event.data
                console.print(f"[cyan]Plan ({len(plan.steps)} steps):[/cyan]")
                for step in plan.steps:
                    console.print(f"  [dim]{step}[/dim]")
            elif event.type == EventType.STEP_STARTED:
                console.print(f"\n[yellow]▶ Step {event.data.index}:[/yellow] {event.data.description}")
            elif event.type == EventType.STEP_COMPLETED:
                preview = event.data.findings[:100].replace("\n", " ")
                console.print(f"  [green]✓[/green] [dim]{preview}…[/dim]")
            elif event.type == EventType.REFLECTION:
                r = event.data
                status = "[green]Sufficient[/green]" if r.sufficient else f"[yellow]{len(r.gaps)} gap(s)[/yellow]"
                console.print(f"\n  Reflection: {status}")
            elif event.type == EventType.REPORT_READY:
                console.print("\n[bold cyan]Report generated.[/bold cyan]")
    except StopIteration as e:
        agent_result = e.value

    pause("Report ready. Press Enter to display it...")

    if agent_result:
        console.print()
        console.rule("[bold cyan]Final Report[/bold cyan]", style="cyan")
        console.print(Markdown(agent_result.report))
        console.rule(style="cyan")

    pause()

    # ── 7. Session save ──────────────────────────────────────────────────
    section(
        "7 · Session Persistence",
        "Conversations can be saved to disk and resumed across sessions."
    )

    path = default_session_path("demo")
    save_session(session3, path)
    console.print(f"[green]✓[/green] Session saved → [cyan]{path}[/cyan]")
    console.print("[dim]Reload with: /load " + str(path) + "[/dim]")
    pause()

    # ── 8. API ───────────────────────────────────────────────────────────
    section(
        "8 · REST API",
        "Every feature is also available over HTTP with Server-Sent Events streaming."
    )
    console.print(
        "Start the API server with:\n"
        "  [bold cyan]serve[/bold cyan]\n\n"
        "Then try:\n"
        "  [bold]GET  /health[/bold]                  — health check\n"
        "  [bold]POST /chat[/bold]                    — streaming chat\n"
        "  [bold]POST /research[/bold]                — agent loop (SSE events)\n"
        "  [bold]POST /documents/ingest[/bold]        — upload a document\n"
        "  [bold]GET  /documents/sources[/bold]       — list ingested files\n"
        "\nInteractive docs: [bold cyan]http://localhost:8000/docs[/bold cyan]"
    )
    pause()

    # ── Summary ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold cyan]Demo Complete[/bold cyan]", style="cyan")
    console.print(
        Panel(
            "[bold]What was demonstrated:[/bold]\n\n"
            "  ✓  Streaming multi-turn chat via the Anthropic SDK\n"
            "  ✓  Tool use / function calling (calculator, search, datetime)\n"
            "  ✓  RAG pipeline — document ingestion, chunking, vector search\n"
            "  ✓  Prompt caching + cost tracking\n"
            "  ✓  Multi-step agent loop (plan → execute → reflect → report)\n"
            "  ✓  Session persistence\n"
            "  ✓  FastAPI REST interface with SSE streaming\n\n"
            f"  [dim]76 tests · Python 3.10 · Anthropic SDK · ChromaDB · FastAPI[/dim]",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    main()
