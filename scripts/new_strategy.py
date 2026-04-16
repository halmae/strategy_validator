"""
New Strategy Input Generator
Usage:
    python scripts/new_strategy.py

Creates a minimal strategy .md file in strategies/ directory.
The file contains only intuition and hypothesis — no implementation details.
(Those emerge through agent dialogue.)

Optionally: pass --template to pick from preset examples.
    python scripts/new_strategy.py --template
"""
import sys
import re
from pathlib import Path
from datetime import datetime

STRATEGIES_DIR = Path(__file__).parent.parent / "strategies"

# ---------------------------------------------------------------------------
# Preset templates for quick testing
# Different alpha_family / market_scope combinations to exercise routing paths
# ---------------------------------------------------------------------------
TEMPLATES = {
    "1": {
        "name": "KOSPI Earnings Momentum",
        "intuition": (
            "Stocks that beat earnings expectations tend to keep rising for several days "
            "after the announcement. The market seems slow to fully reprice after a surprise."
        ),
        "hypothesis": (
            "Institutional investors accumulate earnings-surprise stocks gradually "
            "due to execution constraints, creating a persistent post-announcement drift "
            "in mid-cap Korean equities where analyst coverage is thin."
        ),
    },
    "2": {
        "name": "US Large-Cap Factor Rotation",
        "intuition": (
            "Value and momentum factors seem to take turns outperforming. "
            "When one has run hard for several months, it tends to mean-revert "
            "while the other picks up."
        ),
        "hypothesis": (
            "Systematic factor crowding causes temporary overshoot and correction cycles. "
            "Rotating between value and momentum on a monthly basis should capture "
            "the mean-reversion of crowded factor exposures."
        ),
    },
    "3": {
        "name": "KOSDAQ Stat-Arb Pairs",
        "intuition": (
            "Some KOSDAQ stocks in the same sub-sector move together most of the time, "
            "but occasionally diverge. The divergence usually corrects within a week or two."
        ),
        "hypothesis": (
            "Pairs in the same supply chain share common revenue drivers. "
            "Short-term price divergence is driven by idiosyncratic order flow rather than "
            "fundamental change, making it mean-reverting over a weekly horizon."
        ),
    },
    "4": {
        "name": "Macro Rate Carry",
        "intuition": (
            "High-yield currencies tend to outperform low-yield currencies over time, "
            "even after accounting for occasional sharp reversals during risk-off episodes."
        ),
        "hypothesis": (
            "The carry premium exists because most investors are risk-averse and require "
            "compensation for holding currencies exposed to sudden stop risk. "
            "The premium is structural and persists despite being well-documented."
        ),
    },
}


def slugify(name: str) -> str:
    """Convert strategy name to a safe filename."""
    slug = re.sub(r"[^a-zA-Z0-9\s-]", "", name.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug


def make_md(name: str, intuition: str, hypothesis: str) -> str:
    return f"""# Strategy: {name}

## Intuition
{intuition.strip()}

## Hypothesis
{hypothesis.strip()}
"""


def pick_template() -> dict:
    print("\nAvailable templates:\n")
    for key, t in TEMPLATES.items():
        print(f"  [{key}] {t['name']}")
    print()
    choice = input("Select template number (or press Enter to write manually): ").strip()
    if choice in TEMPLATES:
        return TEMPLATES[choice]
    return {}


def interactive() -> tuple[str, str, str]:
    print("\n── New Strategy ──────────────────────────────────────")
    print("Keep it sparse: intuition and hypothesis only.")
    print("Implementation details emerge through agent dialogue.\n")

    name = input("Strategy name: ").strip()
    if not name:
        name = f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\nIntuition — what pattern or opportunity do you observe?")
    print("(1-3 sentences, rough observation is fine)\n")
    intuition_lines = []
    while True:
        line = input("> ")
        if line:
            intuition_lines.append(line)
        else:
            if intuition_lines:
                break
            print("Please enter at least one line.")

    print("\nHypothesis — why does this opportunity exist?")
    print("(1-3 sentences, rough idea is fine)\n")
    hypothesis_lines = []
    while True:
        line = input("> ")
        if line:
            hypothesis_lines.append(line)
        else:
            if hypothesis_lines:
                break
            print("Please enter at least one line.")

    return name, " ".join(intuition_lines), " ".join(hypothesis_lines)


def main() -> None:
    use_template = "--template" in sys.argv or "-t" in sys.argv

    if use_template:
        data = pick_template()
        if data:
            name = data["name"]
            intuition = data["intuition"]
            hypothesis = data["hypothesis"]
            print(f"\nUsing template: {name}")
        else:
            name, intuition, hypothesis = interactive()
    else:
        name, intuition, hypothesis = interactive()

    # Write file
    STRATEGIES_DIR.mkdir(exist_ok=True)
    slug = slugify(name)
    out_path = STRATEGIES_DIR / f"{slug}.md"

    # Avoid overwriting
    if out_path.exists():
        ts = datetime.now().strftime("%H%M%S")
        out_path = STRATEGIES_DIR / f"{slug}_{ts}.md"

    out_path.write_text(make_md(name, intuition, hypothesis), encoding="utf-8")

    print(f"\n✓ Created: {out_path}")
    print(f"\nTo start validation:\n  python main.py {out_path}\n")


if __name__ == "__main__":
    main()
