"""Public data sources, downloaded at runtime into a local cache (never committed)."""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

SCORPION_BASE = "https://raw.githubusercontent.com/amundfylling/Scorpion-Scraper-2.0/main/data"
SOURCES = {
    "scorpion_matches": f"{SCORPION_BASE}/scraped_matches.parquet",
    "scorpion_players": f"{SCORPION_BASE}/players_data.csv",
    "scorpion_tournaments": f"{SCORPION_BASE}/tournament_metadata.csv",
    "bordshockey_matches": (
        "https://raw.githubusercontent.com/amundfylling/bordshockey.net-scraper/"
        "refs/heads/main/data/bordshockey_results.parquet"
    ),
    "ithf_ranking": "https://stiga.trefik.cz/ithf/ranking/ranking.txt",
}
FILENAMES = {
    "scorpion_matches": "scraped_matches.parquet",
    "scorpion_players": "players_data.csv",
    "scorpion_tournaments": "tournament_metadata.csv",
    "bordshockey_matches": "bordshockey_results.parquet",
    "ithf_ranking": "ranking.txt",
}


def cache_dir() -> Path:
    path = Path(os.environ.get("THSIM_CACHE", Path(__file__).resolve().parents[2] / "data_cache"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch(name: str, refresh: bool = False, required: bool = True) -> Path | None:
    """Return the local path of a source, downloading it when missing.

    Optional sources (``required=False``) return None when the download fails, so a
    blocked host degrades gracefully instead of stopping the pipeline.
    """
    path = cache_dir() / FILENAMES[name]
    if path.exists() and not refresh:
        return path
    url = os.environ.get(f"THSIM_URL_{name.upper()}", SOURCES[name])
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        if required:
            raise
        print(f"Could not download {name} from {url}: {exc}")
        return path if path.exists() else None
    return path
