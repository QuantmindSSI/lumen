"""C9: CLI Implementation (Typer + Rich).

All commands execute real business logic.
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from lumen.config import LumenConfig
from lumen.data.schema import ensure_schema, get_connection
from lumen.force.contextual.embed import FallbackEmbedder
from lumen.force.mnemonic.event_buffer import Event
from lumen.force.mnemonic.store import store_memory
from lumen.force.mnemonic.user_profile import get_profile, update_goals, update_values
from lumen.force.mnemonic.value_model import learn_weights_from_feedback
from lumen.lumen.controller import TwinForceController
from lumen.lumen.search import SearchPipeline
from lumen.compliance.safety_forgetting import get_recent_audit_events

app = typer.Typer(name="lumen", help="Twin-force memory and context framework")
memory_app = typer.Typer(name="memory", help="Memory operations")
palace_app = typer.Typer(name="palace", help="Palace topology operations")
tfc_app = typer.Typer(name="tfc", help="Twin-Force Controller operations")
compliance_app = typer.Typer(name="compliance", help="Compliance operations")

app.add_typer(memory_app)
app.add_typer(palace_app)
app.add_typer(tfc_app)
app.add_typer(compliance_app)

console = Console()

_state_tfc = TwinForceController()


def _ensure_conn(config: LumenConfig):
    return get_connection(config)


@app.command()
def init(device: str = typer.Option("generic", "--device", "-d")):
    """Initialize Lumen palace for the given device profile."""
    config = LumenConfig(device=device)
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.model_path.mkdir(parents=True, exist_ok=True)
    conn = get_connection(config)
    ensure_schema(conn)
    # Write default config TOML
    config_toml = config.store_path.parent / "config.toml"
    config_toml.write_text(
        f'device = "{device}"\ncontext_budget = {config.context_budget}\n'
        f'memory_limit_mb = {config.memory_limit_mb}\nvector_index = "{config.vector_index}"\n'
    )
    console.print(f"[bold green]Lumen initialised for device: {device}[/bold green]")
    console.print(f"Store path: {config.store_path}")
    console.print(f"DB path: {config.db_path}")


@app.command()
def status():
    """Show current palace and TFC status."""
    config = LumenConfig()
    conn = _ensure_conn(config)
    room_count = conn.execute("SELECT COUNT(*) FROM room").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]
    console.print("[bold]Lumen Status[/bold]")
    console.print(f"Device: {config.device}")
    console.print(f"Rooms: {room_count}")
    console.print(f"Active chunks: {chunk_count}")
    console.print(f"Context budget: {config.context_budget} tokens")
    env = _state_tfc.to_env()
    console.print(f"TFC → e={env['e']:.2f} a={env['a']:.2f} tau={env['tau']:.1f} r={env['r']}")


@memory_app.command(name="store")
def memory_store(
    content: str = typer.Argument(..., help="Content to store"),
    room: str = typer.Option(..., "--room", "-r"),
    locus: str = typer.Option(None, "--locus", "-l"),
):
    """Store a memory through the full pipeline."""
    config = LumenConfig()
    conn = _ensure_conn(config)
    embedder = FallbackEmbedder(dims=config.embedding_dims)
    embedding = embedder.encode_single(content)
    chunk_id = store_memory(
        conn, content, room_name=room, locus_name=locus,
        embedding=embedding, config=config
    )
    console.print(f"[bold green]Stored chunk_id={chunk_id} in room '{room}'[/bold green]")


@memory_app.command(name="retrieve")
def memory_retrieve(
    query: str = typer.Argument(..., help="Query to retrieve memories for"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
):
    """Run search pipeline and print top results."""
    config = LumenConfig()
    conn = _ensure_conn(config)
    pipeline = SearchPipeline(conn, config, tfc=_state_tfc)
    results = pipeline.execute(query)
    if not results:
        console.print("[yellow]No memories found.[/yellow]")
        return
    table = Table(title="Retrieved Memories")
    table.add_column("Rank", style="cyan")
    table.add_column("Room", style="magenta")
    table.add_column("Locus", style="green")
    table.add_column("Score", justify="right")
    table.add_column("Content", max_width=60)
    for rank, rc in enumerate(results[:top_k], 1):
        table.add_row(
            str(rank), rc.room_name, rc.locus_name,
            f"{rc.final_score:.3f}", rc.content[:200]
        )
    console.print(table)


@palace_app.command(name="rooms")
def palace_rooms():
    """List all rooms."""
    config = LumenConfig()
    conn = _ensure_conn(config)
    rows = conn.execute("SELECT room_id, name, room_type FROM room ORDER BY name").fetchall()
    table = Table(title="Palace Rooms")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Type", style="green")
    for r in rows:
        table.add_row(str(r[0]), r[1], r[2] or "domain")
    console.print(table)


@palace_app.command(name="loci")
def palace_loci(room: str = typer.Argument(..., help="Room name")):
    """List loci in a room."""
    config = LumenConfig()
    conn = _ensure_conn(config)
    row = conn.execute("SELECT room_id FROM room WHERE name = ?", (room,)).fetchone()
    if not row:
        console.print(f"[red]Room '{room}' not found.[/red]")
        raise typer.Exit(1)
    room_id = row[0]
    rows = conn.execute(
        "SELECT locus_id, name, description FROM locus WHERE room_id = ? ORDER BY name",
        (room_id,)
    ).fetchall()
    table = Table(title=f"Loci in '{room}'")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Description", style="green")
    for r in rows:
        table.add_row(str(r[0]), r[1], r[2] or "")
    console.print(table)


@tfc_app.command(name="show")
def tfc_show():
    """Print TFC state."""
    env = _state_tfc.to_env()
    console.print(f"e (mnemonic bias)    = {env['e']:.2f}")
    console.print(f"a (attention temp)   = {env['a']:.2f}")
    console.print(f"tau (temporal horiz) = {env['tau']:.1f} days")
    console.print(f"r (resolution level) = {env['r']}")


@tfc_app.command(name="set")
def tfc_set(
    e: float = typer.Option(None, "--e"),
    a: float = typer.Option(None, "--a"),
    tau: float = typer.Option(None, "--tau"),
    r: int = typer.Option(None, "--r"),
):
    """Manually override TFC state variables."""
    if e is not None:
        _state_tfc.state.e = max(0.0, min(1.0, e))
    if a is not None:
        _state_tfc.state.a = max(0.0, min(1.0, a))
    if tau is not None:
        _state_tfc.state.tau = tau
    if r is not None:
        _state_tfc.state.r = max(0, min(5, r))
    console.print("[bold green]TFC state updated.[/bold green]")
    tfc_show()


@compliance_app.command(name="audit")
def compliance_audit(n: int = typer.Option(10, "--n")):
    """Show last N compliance events from JSONL."""
    events = get_recent_audit_events(n)
    if not events:
        console.print("[yellow]No compliance events found.[/yellow]")
        return
    table = Table(title="Compliance Audit Log")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Event", style="magenta")
    table.add_column("Chunk ID", style="green")
    table.add_column("Reason", style="yellow")
    for ev in events:
        table.add_row(
            ev.get("ts", ""), ev.get("event", ""),
            str(ev.get("chunk_id", "")), ev.get("reason", "")
        )
    console.print(table)


if __name__ == "__main__":
    app()
