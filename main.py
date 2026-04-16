"""
Validation Agent CLI

Usage:
    python main.py strategies/<strategy_name>.md

Setup:
    1. Set GEMINI_API_KEY in .env or environment.
    2. Configure model / language in config/settings.yaml.
    3. Customize agent behavior in config/prompts/system.md.

KG state is auto-saved to <strategy>.kg.json next to the input file.
Sessions resume automatically from saved state.

Exit: type 'exit' or press Ctrl+C (auto-saves).
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from google import genai
from google.genai import types

from agent.kg_store import load_kg_state, save_kg_state
from agent.schema_loader import schema_for_stage, load_routing, STAGE_SCHEMA_MAP
from agent.config_loader import load_settings, load_prompt, get_model_config
from agent.session_store import (
    last_model_text,
    load_session_contents,
    prune_session_contents,
    save_session_contents,
)
from agent.stage_summary import build_stage_summary
from agent.runner import build_system_prompt, run_turn
from agent.stage_runtime import sync_runtime_state

console = Console()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_header(strategy_path: Path, stage: int, model_name: str) -> None:
    stage_name = STAGE_SCHEMA_MAP.get(stage, "unknown")
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        Panel(
            f"[bold]Validation Agent[/bold]\n"
            f"Strategy : [cyan]{strategy_path.name}[/cyan]\n"
            f"Model    : [magenta]{model_name}[/magenta]\n"
            f"Stage    : [yellow]{stage}[/yellow] — {stage_name}",
            expand=False,
        )
    )
    console.print(Rule(style="dim"))
    console.print()


def print_kg_summary(kg_state) -> None:
    if kg_state.type_vector:
        console.print("[dim]Type vector:[/dim]", kg_state.type_vector)
    if kg_state.deferred:
        console.print("[dim]Deferred resolved:[/dim]", kg_state.deferred)
    active_checks = kg_state.active_checks.get(f"stage_{kg_state.stage}", [])
    if active_checks:
        console.print("[dim]Active checks:[/dim]", active_checks)
    if kg_state.gates:
        s = kg_state.gate_summary()
        console.print(
            f"[dim]Gates[/dim] — "
            f"[green]pass: {len(s['passed'])}[/green]  "
            f"[yellow]pending: {len(s['pending'])}[/yellow]  "
            f"[red]fail: {len(s['failed'])}[/red]"
        )
    if kg_state.workflow_complete:
        console.print(
            f"[green]Implemented scope complete through Stage {max(STAGE_SCHEMA_MAP)}.[/green]"
        )
    if kg_state.out_of_scope and kg_state.out_of_scope_reason:
        console.print(f"[red]Out of scope:[/red] {kg_state.out_of_scope_reason}")
    console.print()


def display_agent(text: str) -> None:
    if text:
        console.print(
            Panel(
                Text(text),
                title="[bold green]Agent[/bold green]",
                border_style="green",
                expand=False,
            )
        )


def save_state(strategy_path: Path, kg_state, contents: list[types.Content]) -> None:
    save_kg_state(strategy_path, kg_state)
    save_session_contents(strategy_path, contents)


def save_and_print(strategy_path: Path, kg_state, contents: list[types.Content]) -> None:
    save_state(strategy_path, kg_state, contents)
    console.print(f"[green]Saved → {strategy_path.with_suffix('.kg.json')}[/green]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print(
            "[red]GEMINI_API_KEY not found. Set it in .env or environment.[/red]"
        )
        sys.exit(1)

    if len(sys.argv) < 2:
        console.print("[red]Usage: python main.py <strategy_file>[/red]")
        sys.exit(1)

    strategy_path = Path(sys.argv[1])
    if not strategy_path.exists():
        console.print(f"[red]File not found: {strategy_path}[/red]")
        sys.exit(1)

    # Load settings first — needed by print_header
    settings = load_settings()
    model_config = get_model_config()
    model_name = model_config["name"]
    keep_recent_turns = settings.get("session", {}).get("keep_recent_turns", 4)

    strategy_text = strategy_path.read_text(encoding="utf-8")
    kg_state = load_kg_state(strategy_path)
    session_contents = load_session_contents(strategy_path)
    resuming = bool(kg_state.type_vector or kg_state.entities or session_contents)

    routing = load_routing()
    max_implemented_stage = max(STAGE_SCHEMA_MAP)

    print_header(strategy_path, kg_state.stage, model_name)

    client = genai.Client(api_key=api_key)

    try:
        schema = schema_for_stage(kg_state.stage)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Schema error: {e}[/red]")
        sys.exit(1)

    synced_stages = sorted(
        {
            stage_id
            for stage_id in [*kg_state.completed_stages, kg_state.stage]
            if stage_id in STAGE_SCHEMA_MAP
        }
    )
    for stage_id in synced_stages:
        stage_schema = schema if stage_id == kg_state.stage else schema_for_stage(stage_id)
        sync_runtime_state(stage_id, stage_schema, routing, kg_state)

    summaries_updated = False
    for completed_stage in kg_state.completed_stages:
        completed_schema = schema if completed_stage == kg_state.stage else schema_for_stage(completed_stage)
        kg_state.set_stage_summary(
            completed_stage,
            build_stage_summary(completed_stage, completed_schema, kg_state),
        )
        summaries_updated = True

    if resuming:
        console.print("[yellow]Resuming previous session.[/yellow]")
        print_kg_summary(kg_state)
    else:
        console.print("[green]Starting new validation session.[/green]")
        console.print()

    if summaries_updated:
        save_state(strategy_path, kg_state, session_contents or [])

    # Initial trigger — from config/prompts/
    should_run_initial_turn = False
    if session_contents:
        contents = session_contents
        restored_text = last_model_text(contents)
        if restored_text:
            console.print("[dim]Restored last agent turn:[/dim]")
            display_agent(restored_text)
    elif resuming:
        initial = load_prompt("initial_resume").format(stage=kg_state.stage)
        contents = [types.Content(role="user", parts=[types.Part(text=initial)])]
        should_run_initial_turn = True
        console.print(
            "[yellow]Detailed conversation history was not saved for this session, "
            "so the next step will be reconstructed from KG state.[/yellow]"
        )
    else:
        initial = load_prompt("initial_new")
        contents = [types.Content(role="user", parts=[types.Part(text=initial)])]
        should_run_initial_turn = True

    console.print("[dim]Type 'exit' or Ctrl+C to quit (auto-saves).[/dim]")
    console.print()

    current_stage = kg_state.stage

    # ---------------------------------------------------------------------------
    # Helper: run one agent turn, display, save, check stage advance
    # ---------------------------------------------------------------------------
    def agent_turn() -> bool:
        """Execute one agent turn. Returns False if a fatal error occurred."""
        nonlocal schema, current_stage, contents

        turn_stage = kg_state.stage
        turn_schema = schema
        completed_before = set(kg_state.completed_stages)
        sync_runtime_state(turn_stage, turn_schema, routing, kg_state)

        system_prompt = build_system_prompt(
            turn_stage, turn_schema, routing, kg_state, strategy_text, settings
        )
        try:
            agent_text, updated_contents = run_turn(
                client,
                contents,
                system_prompt,
                kg_state,
                stage=turn_stage,
                schema=turn_schema,
                routing=routing,
                max_implemented_stage=max_implemented_stage,
                model_config=model_config,
            )
        except Exception as e:
            console.print(f"[red]API error: {e}[/red]")
            save_state(strategy_path, kg_state, contents)
            return False

        contents = updated_contents

        for completed_stage in kg_state.completed_stages:
            completed_schema = turn_schema
            if completed_stage != turn_stage:
                completed_schema = schema_for_stage(completed_stage)
            kg_state.set_stage_summary(
                completed_stage,
                build_stage_summary(completed_stage, completed_schema, kg_state),
            )

        newly_completed = [
            stage_id
            for stage_id in kg_state.completed_stages
            if stage_id not in completed_before
        ]
        if newly_completed:
            contents = prune_session_contents(contents, keep_recent_turns=keep_recent_turns)

        if kg_state.stage != current_stage:
            current_stage = kg_state.stage
            console.print(f"\n[yellow]► Advancing to Stage {current_stage}[/yellow]\n")
            try:
                schema = schema_for_stage(current_stage)
            except (FileNotFoundError, ValueError) as e:
                console.print(f"[red]Schema error: {e}[/red]")
                save_state(strategy_path, kg_state, contents)
                return False

        sync_runtime_state(kg_state.stage, schema, routing, kg_state)

        if agent_text:
            display_agent(agent_text)
        else:
            # Model completed function calls but generated no text.
            # Show a compact status so the user knows what happened.
            summary = []
            if kg_state.type_vector:
                summary.append(f"Type vector resolved: {list(kg_state.type_vector.keys())}")
            passed = kg_state.gate_summary().get("passed", [])
            if passed:
                summary.append(f"Gates passed: {passed}")
            if kg_state.workflow_complete:
                summary.append("Implemented scope complete")
            if kg_state.out_of_scope:
                summary.append("Out of scope")
            status = " | ".join(summary) if summary else "KG updated."
            console.print(f"[dim]({status})[/dim]")

        save_state(strategy_path, kg_state, contents)
        return True

    # ---------------------------------------------------------------------------
    # Initial agent turn (before any user input)
    # ---------------------------------------------------------------------------
    if should_run_initial_turn:
        if not agent_turn():
            return
    elif kg_state.workflow_complete or kg_state.out_of_scope:
        save_and_print(strategy_path, kg_state, contents)
        return

    if kg_state.workflow_complete:
        console.print(
            f"[green]Stage {max_implemented_stage} is complete. Later stages are not implemented yet.[/green]"
        )
        save_and_print(strategy_path, kg_state, contents)
        return

    if kg_state.out_of_scope:
        console.print(f"[red]{kg_state.out_of_scope_reason}[/red]")
        save_and_print(strategy_path, kg_state, contents)
        return

    # ---------------------------------------------------------------------------
    # Input-driven loop: collect user input FIRST, then call API
    # This ensures 'continue' on empty input never triggers an extra API call.
    # ---------------------------------------------------------------------------
    while True:
        console.print()
        try:
            user_input = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Saving...[/dim]")
            save_and_print(strategy_path, kg_state, contents)
            break

        if user_input.lower() in ("exit", "quit", "q"):
            save_and_print(strategy_path, kg_state, contents)
            break

        if not user_input:
            continue  # safe: API is not called until after input is appended

        contents.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        if not agent_turn():
            break

        if kg_state.workflow_complete:
            console.print(
                f"[green]Stage {max_implemented_stage} is complete. Later stages are not implemented yet.[/green]"
            )
            save_and_print(strategy_path, kg_state, contents)
            break

        if kg_state.out_of_scope:
            console.print(f"[red]{kg_state.out_of_scope_reason}[/red]")
            save_and_print(strategy_path, kg_state, contents)
            break


if __name__ == "__main__":
    main()
