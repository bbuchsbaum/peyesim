#!/usr/bin/env python3
"""Editorial lint for the authored docs surface.

Battle-tested pattern shipped by the quarto-python-docs skill. It keeps a
docs site honest by failing the build on three classes of drift:

1. Banned marketing adjectives — replace with the concrete fact.
2. Identity-by-comparison framing is allowed ONLY under one directory
   (typically evidence/), where comparison is a measurement, not framing.
3. Canonical API tokens must be spelled/cased correctly.

Wire it as a `make` prerequisite so it cannot be skipped (see the skill's
Makefile section). Scope is the *authored* surface only; generated
(reference/), archived, and legacy pages are out of scope.

Exit 0 = clean, 1 = violations. Run: python tools/docs_lint.py

═══════════════════════════════════════════════════════════════════════
PROJECT-SPECIFIC CONFIG — edit the four marked blocks below per project.
The mechanism is generic; the paths and token lists are not. Derive
SIBLINGS / CANON_MISSPELLINGS from the package's real public API
(grep __all__) and from real typos seen in drafts.
═══════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

# ── EDIT 1: authored surface (everything comparison-gated except COMPARE_DIR)
AUTHORED = [
    DOCS / "index.qmd",
    DOCS / "get-started.qmd",
    DOCS / "tutorials",
]
# No measurement/scoreboard directory exists yet, so identity-by-comparison
# is allowed nowhere. (If an evidence/ scoreboard is added later, set this
# to "evidence" and list it in AUTHORED.)
COMPARE_DIR = ""

# ── EDIT 2: banned marketing adjectives (generic; rarely needs changes)
BANNED_ADJECTIVES = [
    "powerful", "easy", "intuitive", "modern", "comprehensive",
    "seamless", "blazing", "effortless", "cutting-edge", "state-of-the-art",
]

# ── EDIT 3: sibling/competitor names this project must not frame against
# outside COMPARE_DIR. NOTE: the R `eyesim` package is this project's
# lineage and is referenced as provenance throughout — it is deliberately
# NOT listed here. These are unrelated eye-movement libraries.
SIBLINGS = r"(?:pygaze|remodnav|multimatch-gaze|scanpath|saccademodel)"
COMPARISON_PHRASES = [
    r"alternative to \w+",
    rf"unlike {SIBLINGS}",
    rf"better than {SIBLINGS}",
    rf"(?:just )?like {SIBLINGS}\b",
    rf"compared to {SIBLINGS}",
    r"drop-in replacement",
]

# ── EDIT 4: canonical API token misspellings -> correct token. Seed from
# real typos in drafts; this is the cheapest guard against API drift in prose.
CANON_MISSPELLINGS = {
    r"\bfixaton\b": "fixation",
    r"\bfixation_groups\(": "fixation_group(",
    r"\beye_densities\(": "eye_density(",
    r"\bmultimatch\(": "multi_match(",
    r"\bscan +path\b": "scanpath",
    r"\bprovenence\b": "provenance",
}


def iter_files():
    for entry in AUTHORED:
        if entry.is_file():
            yield entry
        elif entry.is_dir():
            yield from sorted(entry.rglob("*.qmd"))
            yield from sorted(entry.rglob("*.md"))


def strip_code_blocks(text: str) -> str:
    """Blank out fenced code so prose-only rules don't fire on code."""
    return re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL)


def main() -> int:
    adj_re = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(a) for a in BANNED_ADJECTIVES) + r")(?![\w-])",
        re.IGNORECASE,
    )
    cmp_res = [re.compile(p, re.IGNORECASE) for p in COMPARISON_PHRASES]
    miss_res = [(re.compile(p), good) for p, good in CANON_MISSPELLINGS.items()]

    violations: list[str] = []

    for path in iter_files():
        rel = path.relative_to(DOCS)
        in_compare_dir = bool(
            COMPARE_DIR and rel.parts and rel.parts[0] == COMPARE_DIR
        )
        prose = strip_code_blocks(path.read_text(encoding="utf-8", errors="replace"))

        for i, line in enumerate(prose.splitlines(), 1):
            for m in adj_re.finditer(line):
                violations.append(f"{rel}:{i}: banned adjective '{m.group(1)}' — state the concrete fact instead")
            if not in_compare_dir:
                for cre in cmp_res:
                    m = cre.search(line)
                    if m:
                        where = f"docs/{COMPARE_DIR}/" if COMPARE_DIR else "nowhere"
                        violations.append(
                            f"{rel}:{i}: identity-by-comparison '{m.group(0)}' — "
                            f"comparison framing belongs only under {where}"
                        )
            for mre, good in miss_res:
                m = mre.search(line)
                if m:
                    violations.append(f"{rel}:{i}: '{m.group(0)}' — canonical token is '{good}'")

    if violations:
        print("Docs editorial lint: FAIL\n")
        for v in violations:
            print("  " + v)
        print(f"\n{len(violations)} violation(s).")
        return 1

    print(f"Docs editorial lint: OK ({sum(1 for _ in iter_files())} authored files clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
