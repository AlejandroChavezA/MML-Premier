#!/usr/bin/env python3
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import duckdb

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ligamx" / "cleaned"
console = Console()

KEY_COLS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "result", "referee", "stadium", "attendance", "season",
    "home_manager", "away_manager", "home_system_of_play", "away_system_of_play",
]

def show_csv(path: Path):
    conn = duckdb.connect(":memory:")
    df = conn.execute(f"SELECT * FROM read_csv_auto('{path}')").df()
    total_rows = len(df)

    console.print()
    console.print(Panel(
        f"[bold]{path.name}[/bold] — {total_rows} registros, {len(df.columns)} columnas",
        style="cyan"
    ))

    cols_to_show = [c for c in KEY_COLS if c in df.columns]
    if not cols_to_show:
        cols_to_show = list(df.columns[:10])

    df_preview = df[cols_to_show].head(10)

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    for col in df_preview.columns:
        table.add_column(str(col), overflow="fold", max_width=25)
    for _, row in df_preview.iterrows():
        table.add_row(*[str(v)[:24] for v in row])
    console.print(table)

def main():
    console.print(Panel("[bold green]Liga MX — Datos Limpios[/bold green]", style="green"))
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        console.print("[red]No se encontraron archivos CSV en data/ligamx/cleaned/[/red]")
        return
    for f in files:
        show_csv(f)
    console.print(f"\n[bold]{len(files)} archivos mostrados[/bold]")

if __name__ == "__main__":
    main()
