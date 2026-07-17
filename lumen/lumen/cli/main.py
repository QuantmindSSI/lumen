"""C9: CLI Implementation stub (Typer + Rich).

Input wire: Typer, Rich
Output wire: All subsystems via API calls
"""

import typer
from rich.console import Console

app = typer.Typer(name="lumen", help="Twin-force memory and context framework")
console = Console()


@app.command()
def status():
    """Show current palace and TFC status."""
    console.print("Lumen status: [bold green]ACTIVE[/bold green]")


@app.command()
def init(device: str = typer.Option("generic", "--device", "-d")):
    """Initialize Lumen palace for the given device profile."""
    from lumen.config import LumenConfig
    from lumen.data.schema import get_connection, init_db

    config = LumenConfig(device=device)
    conn = get_connection(config)
    init_db(conn)
    console.print(f"[bold green]Lumen initialised for device: {device}[/bold green]")
