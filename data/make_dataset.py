"""Generate a deterministic synthetic dataset for polybatch demos and tests.

Writes a CSV with header "order_id,text" where each row is a zero-padded
order_id (e.g. rec_0001) and a short neutral ASCII filler sentence assembled
from seeded word pools. Same --seed + --rows always produces the same file.

Usage:
  python data/make_dataset.py --rows 300 --seed 42 --out records.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ADJECTIVES = [
    "compact",
    "modular",
    "lightweight",
    "flexible",
    "durable",
    "quiet",
    "portable",
    "efficient",
    "sturdy",
    "affordable",
    "elegant",
    "reliable",
]

NOUNS = [
    "widget",
    "planner",
    "tracker",
    "organizer",
    "toolkit",
    "assistant",
    "dashboard",
    "notebook",
    "scheduler",
    "reminder",
    "gadget",
    "workbook",
]

VERBS = [
    "improves",
    "simplifies",
    "streamlines",
    "supports",
    "organizes",
    "accelerates",
    "clarifies",
    "coordinates",
    "enhances",
    "automates",
]

AUDIENCES = [
    "students",
    "small teams",
    "remote workers",
    "hobbyists",
    "commuters",
    "researchers",
    "freelancers",
    "volunteers",
    "travelers",
    "beginners",
]


def make_sentence(rng: random.Random) -> str:
    """Assemble one neutral filler sentence from the seeded word pools."""
    adjective = rng.choice(ADJECTIVES)
    noun = rng.choice(NOUNS)
    verb = rng.choice(VERBS)
    audience = rng.choice(AUDIENCES)
    return f"A {adjective} {noun} that {verb} daily planning for {audience}"


def generate(rows: int, seed: int) -> list[tuple[str, str]]:
    """Return a deterministic list of (order_id, text) rows."""
    rng = random.Random(seed)
    width = max(4, len(str(rows)))
    out: list[tuple[str, str]] = []
    for i in range(1, rows + 1):
        order_id = f"rec_{i:0{width}d}"
        out.append((order_id, make_sentence(rng)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300, help="number of records")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--out",
        type=str,
        default="records.csv",
        help="output filename (relative to this script's directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path

    rows = generate(args.rows, args.seed)
    with out_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "text"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
