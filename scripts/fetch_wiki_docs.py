"""
Fetch F1 Wikipedia articles and save them as .md files for RAG ingestion.

Pulls three categories:
  - Circuits  : current F1 calendar tracks
  - Drivers   : 2024/2025 season grid
  - Seasons   : 2020-2024 Formula One World Championship articles

Output lands in knowledge_base/raw_docs/{circuits,drivers,seasons}/.
Run this before scripts/ingest_docs.py.

Usage:
    python scripts/fetch_wiki_docs.py
    python scripts/fetch_wiki_docs.py --seasons 2022 2023 2024
    python scripts/fetch_wiki_docs.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

console = Console()
logger = logging.getLogger(__name__)

WIKI_API = "https://en.wikipedia.org/w/api.php"
RATE_LIMIT = 0.5   # seconds between requests — stay polite


# ---------------------------------------------------------------------------
# Article definitions
# ---------------------------------------------------------------------------

class Article(NamedTuple):
    title: str      # Wikipedia page title
    slug: str       # filename stem
    category: str   # subdir: circuits | drivers | seasons


CIRCUITS: list[Article] = [
    Article("Bahrain International Circuit",          "bahrain",       "circuits"),
    Article("Jeddah Street Circuit",                  "jeddah",        "circuits"),
    Article("Albert Park Circuit",                    "albert_park",   "circuits"),
    Article("Suzuka International Racing Course",     "suzuka",        "circuits"),
    Article("Shanghai International Circuit",         "shanghai",      "circuits"),
    Article("Miami International Autodrome",          "miami",         "circuits"),
    Article("Autodromo Enzo e Dino Ferrari",          "imola",         "circuits"),
    Article("Circuit de Monaco",                      "monaco",        "circuits"),
    Article("Circuit Gilles Villeneuve",              "montreal",      "circuits"),
    Article("Circuit de Barcelona-Catalunya",         "barcelona",     "circuits"),
    Article("Red Bull Ring",                          "austria",       "circuits"),
    Article("Silverstone Circuit",                    "silverstone",   "circuits"),
    Article("Hungaroring",                            "hungary",       "circuits"),
    Article("Circuit de Spa-Francorchamps",           "spa",           "circuits"),
    Article("Circuit Park Zandvoort",                 "zandvoort",     "circuits"),
    Article("Autodromo Nazionale Monza",              "monza",         "circuits"),
    Article("Baku City Circuit",                      "baku",          "circuits"),
    Article("Marina Bay Street Circuit",              "singapore",     "circuits"),
    Article("Circuit of the Americas",                "cota",          "circuits"),
    Article("Autodromo Hermanos Rodriguez",           "mexico_city",   "circuits"),
    Article("Autodromo Jose Carlos Pace",             "interlagos",    "circuits"),
    Article("Las Vegas Street Circuit",               "las_vegas",     "circuits"),
    Article("Lusail International Circuit",           "qatar",         "circuits"),
    Article("Yas Marina Circuit",                     "abu_dhabi",     "circuits"),
]

DRIVERS: list[Article] = [
    Article("Max Verstappen",         "verstappen",    "drivers"),
    Article("Sergio Pérez",           "perez",         "drivers"),
    Article("Lewis Hamilton",         "hamilton",      "drivers"),
    Article("George Russell",         "russell",       "drivers"),
    Article("Charles Leclerc",        "leclerc",       "drivers"),
    Article("Carlos Sainz Jr.",       "sainz",         "drivers"),
    Article("Lando Norris",           "norris",        "drivers"),
    Article("Oscar Piastri",          "piastri",       "drivers"),
    Article("Fernando Alonso",        "alonso",        "drivers"),
    Article("Lance Stroll",           "stroll",        "drivers"),
    Article("Esteban Ocon",           "ocon",          "drivers"),
    Article("Pierre Gasly",           "gasly",         "drivers"),
    Article("Valtteri Bottas",        "bottas",        "drivers"),
    Article("Zhou Guanyu",            "zhou",          "drivers"),
    Article("Yuki Tsunoda",           "tsunoda",       "drivers"),
    Article("Daniel Ricciardo",       "ricciardo",     "drivers"),
    Article("Kevin Magnussen",        "magnussen",     "drivers"),
    Article("Nico Hülkenberg",        "hulkenberg",    "drivers"),
    Article("Alexander Albon",        "albon",         "drivers"),
    Article("Logan Sargeant",         "sargeant",      "drivers"),
    Article("Kimi Antonelli",         "antonelli",     "drivers"),
    Article("Oliver Bearman",         "bearman",       "drivers"),
    Article("Isack Hadjar",           "hadjar",        "drivers"),
    Article("Jack Doohan",            "doohan",        "drivers"),
    Article("Gabriel Bortoleto",      "bortoleto",     "drivers"),
    Article("Liam Lawson",            "lawson",        "drivers"),
]

def _season_articles(years: list[int]) -> list[Article]:
    return [
        Article(
            f"{y} Formula One World Championship",
            f"season_{y}",
            "seasons",
        )
        for y in years
    ]


# ---------------------------------------------------------------------------
# Wikipedia fetch
# ---------------------------------------------------------------------------

async def fetch_article(client: httpx.AsyncClient, title: str) -> str | None:
    """Return plain-text extract for a Wikipedia article, or None on failure."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "format": "json",
        "redirects": 1,
    }
    try:
        resp = await client.get(WIKI_API, params=params, timeout=20)
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            return None
        return page.get("extract", "")
    except Exception as exc:
        logger.warning("Failed to fetch '%s': %s", title, exc)
        return None


def _clean(text: str) -> str:
    """Remove excessive blank lines from Wikipedia plain-text extracts."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _save(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{_clean(text)}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(articles: list[Article], out_dir: Path, dry_run: bool) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; f1-intelligence/1.0; mailto:ombharat6361@gmail.com)"}

    ok = skipped = failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching articles…", total=len(articles))

        async with httpx.AsyncClient(headers=headers) as client:
            for article in articles:
                dest = out_dir / article.category / f"{article.slug}.md"
                progress.update(task, description=f"[cyan]{article.slug}")

                if dest.exists():
                    progress.advance(task)
                    skipped += 1
                    continue

                if dry_run:
                    console.print(f"  [dim]dry-run:[/] {dest}")
                    progress.advance(task)
                    ok += 1
                    continue

                text = await fetch_article(client, article.title)
                if text:
                    _save(dest, article.title, text)
                    ok += 1
                else:
                    console.print(f"  [yellow]MISS[/] {article.title}")
                    failed += 1

                progress.advance(task)
                await asyncio.sleep(RATE_LIMIT)

    console.print(
        f"\n[bold green]Done.[/] {ok} fetched, {skipped} skipped (already exist), {failed} failed."
    )
    if ok > 0 and not dry_run:
        console.print(
            f"\nNext step: [bold]python scripts/ingest_docs.py[/] to embed and store in ChromaDB."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch F1 Wikipedia docs for RAG")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(range(2020, 2025)),
        metavar="YEAR",
        help="Season years to fetch (default: 2020-2024)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("knowledge_base/raw_docs"),
        help="Output directory (default: knowledge_base/raw_docs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without downloading",
    )
    parser.add_argument(
        "--only",
        choices=["circuits", "drivers", "seasons"],
        help="Fetch only one category",
    )
    args = parser.parse_args()

    all_articles = CIRCUITS + DRIVERS + _season_articles(args.seasons)
    if args.only:
        all_articles = [a for a in all_articles if a.category == args.only]

    console.print(
        f"[bold]F1 Wikipedia fetcher[/] — {len(all_articles)} articles "
        f"-> [cyan]{args.out_dir}[/]"
        + (" [yellow](dry run)[/]" if args.dry_run else "")
    )

    asyncio.run(run(all_articles, args.out_dir, args.dry_run))


if __name__ == "__main__":
    main()
