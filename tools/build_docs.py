#!/usr/bin/env python3
"""Generate the Fumadocs content tree for the peyesim documentation site.

Two inputs become Markdown pages under ``docs/content/docs/``:

* Authored pages in ``docs/source/`` (Markdown with fenced Python cells). Every
  ```` ```python ```` cell is executed in order, in one namespace per page,
  and its printed output, the value of its last expression and any matplotlib
  figures are written after it. Leading ``#|`` lines set cell options, as in
  Quarto: ``#| eval: false`` (show, do not run), ``#| echo: false`` (run,
  hide the code) and ``#| output: false`` (run, hide the results).
  ``meta.json`` files are copied as they are.
* The package docstrings, read with griffe (NumPy style), become one API
  reference page per section of ``REFERENCE_SECTIONS``. Redirect pages for the
  old Quarto per-object URLs (``reference/<name>.html``) go to
  ``docs/public/reference/``.

Run from anywhere: ``python tools/build_docs.py``. The output directory is
replaced on every run, so it is safe to delete. A failing cell stops the
build with the page and cell number.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "source"
CONTENT = REPO / "docs" / "content" / "docs"
# Static files copied into the site as they are (generated, not committed).
LEGACY = REPO / "docs" / "public" / "reference"

# Reference pages: (slug, title, description, public names). Every name in
# peyesim.__all__ must appear exactly once (tests/test_docs_reference.py).
REFERENCE_SECTIONS = [
    ("fixations", "Fixations", "Build and manipulate ordered fixation sequences.", [
        "FixationGroup", "fixation_group", "concat_fixation_groups", "coords", "center",
        "normalize", "rescale", "rep_fixations", "sample_fixations",
    ]),
    ("eye-tables", "Eye tables", "Trial-level tables that hold many fixation groups.", [
        "EyeTable", "eye_table", "as_eye_table", "simulate_eye_table",
    ]),
    ("scanpaths", "Scanpaths", "Saccades between consecutive fixations.", [
        "Scanpath", "scanpath", "add_scanpath", "cart2pol", "calcangle",
    ]),
    ("density", "Density maps", "Kernel density estimation over a fixation pattern.", [
        "eye_density", "gen_density", "get_density", "density_matrix", "density_by",
        "suggest_sigma", "EyeDensity", "EyeDensityMultiscale",
    ]),
    ("similarity", "Similarity", "Comparisons between densities, fixations and scanpaths.", [
        "similarity", "compute_similarity", "template_similarity", "template_similarity_cv",
        "fixation_similarity", "scanpath_similarity", "sample_density", "sample_density_time",
        "fixation_overlap", "repetitive_similarity",
    ]),
    ("scanpath-comparison", "Scanpath comparison", "MultiMatch and cross-recurrence analysis.", [
        "multi_match", "install_multimatch", "crqa",
    ]),
    ("regression", "Regression and templates", "Regress retrieval gaze onto encoding templates.", [
        "template_regression", "template_multireg", "template_sample",
    ]),
    ("latent-transforms", "Latent transforms", "Align density maps across devices or participants before comparing them.", [
        "latent_pca_transform", "coral_transform", "cca_transform", "contract_transform",
        "affine_transform",
    ]),
    ("entropy-expansion", "Entropy and expansion", "Spatial entropy and scale estimation.", [
        "fixation_entropy", "entropy_from_mass", "estimate_scale", "match_scale",
    ]),
    ("revisitation", "Revisitation and display geometry", "Returns to previously fixated locations, and visual-angle conversion.", [
        "fixation_revisits", "revisit_summary", "RevisitTable", "visual_angle_to_px",
        "px_to_visual_angle",
    ]),
    ("visualization", "Visualization", "Plot fixations, densities and animated scanpaths.", [
        "plot_fixation_group", "plot_eye_density", "anim_scanpath", "eyesim_colours",
        "eyesim_density_cmap", "eyesim_time_cmap", "theme_eyesim", "theme_eyesim_spatial",
    ]),
    ("datasets", "Datasets", "Example data bundled with the package.", [
        "available_datasets", "data_path", "load_dataset",
    ]),
]

FENCE = re.compile(r"^```(\S*)[^\n]*\n(.*?)^```[ \t]*$", re.M | re.S)
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
MAX_TABLE_ROWS = 12
MAX_TEXT_LINES = 60


# ---------------------------------------------------------------------------
# Executed pages
# ---------------------------------------------------------------------------

def _cell_options(code: str) -> tuple[dict, str]:
    opts = {"eval": True, "echo": True, "output": True}
    lines = code.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        m = re.match(r"#\|\s*([\w-]+)\s*:\s*(.+?)\s*$", line)
        if not m:
            break
        key, value = m.group(1), m.group(2).strip("\"'")
        if key in opts:
            opts[key] = value.lower() not in {"false", "no", "0"}
        body_start = i + 1
    return opts, "\n".join(lines[body_start:]).strip("\n")


def _code_block(text: str, lang: str = "text", title: str | None = None) -> str:
    fence = "````" if "```" in text else "```"
    meta = f' title="{title}"' if title else ""
    return f"{fence}{lang}{meta}\n{text.rstrip()}\n{fence}"


def _truncate_lines(text: str) -> str:
    lines = text.rstrip().splitlines()
    if len(lines) > MAX_TEXT_LINES:
        lines = lines[:MAX_TEXT_LINES] + [f"... ({len(lines) - MAX_TEXT_LINES} more lines)"]
    return "\n".join(lines)


def _table_cell(value) -> str:
    import numpy as np

    if isinstance(value, (float, np.floating)):
        text = "NaN" if np.isnan(value) else f"{value:.6g}"
    else:
        text = str(value).replace("\n", " ")
    if len(text) > 40:
        text = text[:37] + "..."
    return text.replace("|", "\\|")


def _frame_markdown(df) -> str:
    """A pandas frame as a GFM table (first rows only, index included)."""
    shown = df.head(MAX_TABLE_ROWS)
    header = [""] + [str(c) for c in shown.columns]
    rows = [[_table_cell(i)] + [_table_cell(v) for v in row]
            for i, row in zip(shown.index, shown.itertuples(index=False))]
    out = ["| " + " | ".join(h.replace("|", "\\|") for h in header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    note = f"\n\n{len(df)} rows × {df.shape[1]} columns" if len(df) > MAX_TABLE_ROWS else ""
    return "\n".join(out) + note


def _display(value) -> str | None:
    import pandas as pd

    if value is None:
        return None
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame) and type(value).__repr__ is pd.DataFrame.__repr__:
        return _frame_markdown(value)
    if isinstance(value, pd.DataFrame):
        # Subclasses with their own repr (FixationGroup, RevisitTable, ...)
        return _code_block(_truncate_lines(repr(value)), title="Output")
    if type(value).__module__.startswith("matplotlib"):
        return None  # Axes, Figure, artists: the figure itself is shown.
    return _code_block(_truncate_lines(repr(value)), title="Output")


def _run_cell(code: str, ns: dict, where: str):
    """Execute code; return (stdout text, value of a trailing expression)."""
    tree = ast.parse(code, filename=where)
    tail = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tail = ast.Expression(tree.body.pop().value)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(tree, where, "exec"), ns)
        value = eval(compile(tail, where, "eval"), ns) if tail is not None else None
    return stdout.getvalue(), value


def _save_figures(fig_dir: Path, stem: str, cell: int) -> list[str]:
    import matplotlib.pyplot as plt

    names = []
    for k, num in enumerate(plt.get_fignums(), 1):
        fig = plt.figure(num)
        fig_dir.mkdir(parents=True, exist_ok=True)
        name = f"{stem}-{cell}-{k}.png"
        fig.savefig(fig_dir / name, dpi=150, bbox_inches="tight", facecolor="white")
        names.append(name)
    plt.close("all")
    return names


def render_page(src: Path, dest: Path) -> int:
    """Execute one authored page and write it as Markdown. Returns cells run."""
    text = src.read_text(encoding="utf-8")
    rel_source = src.relative_to(REPO).as_posix()
    m = FRONTMATTER.match(text)
    front, body = (m.group(1), text[m.end():]) if m else ("", text)
    front = "\n".join(line for line in front.splitlines() if not line.startswith("source:"))
    header = f"---\n{front}\nsource: {rel_source}\n---\n" if front else f"---\nsource: {rel_source}\n---\n"

    ns: dict = {"__name__": "__docs__"}
    out: list[str] = []
    pos = 0
    n_cells = 0
    fig_dir = dest.parent / "figures"
    for cell_no, fence in enumerate(FENCE.finditer(body), 1):
        out.append(body[pos:fence.start()])
        pos = fence.end()
        lang, code = fence.group(1), fence.group(2)
        if lang not in {"python", "{python}"}:
            out.append(fence.group(0))
            continue
        opts, code = _cell_options(code)
        if opts["echo"]:
            out.append(_code_block(code, "python"))
        if not opts["eval"]:
            continue
        n_cells += 1
        where = f"{rel_source} (cell {cell_no})"
        try:
            printed, value = _run_cell(code, ns, where)
        except Exception as exc:  # noqa: BLE001 - report the page and cell
            raise SystemExit(f"build_docs: {where} failed: {type(exc).__name__}: {exc}") from exc
        figures = _save_figures(fig_dir, dest.stem, cell_no)
        if not opts["output"]:
            continue
        blocks = []
        if printed.strip():
            blocks.append(_code_block(_truncate_lines(printed), title="Output"))
        shown = _display(value)
        if shown:
            blocks.append(shown)
        blocks += [f"![]({'./figures/' + name})" for name in figures]
        if blocks:
            out.append("\n\n" + "\n\n".join(blocks))
    out.append(body[pos:])

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + "".join(out), encoding="utf-8")
    return n_cells


# ---------------------------------------------------------------------------
# API reference
# ---------------------------------------------------------------------------

def _anchor(name: str) -> str:
    """The heading id Fumadocs (github-slugger) gives ``## name``."""
    return re.sub(r"[^a-z0-9 _-]", "", name.lower()).replace(" ", "-")


def _reference_index() -> dict[str, str]:
    return {name: f"/reference/{slug}#{_anchor(name)}"
            for slug, _, _, names in REFERENCE_SECTIONS for name in names}


def _rst_to_md(text: str, links: dict[str, str]) -> str:
    """The reST bits used in the docstrings, as Markdown."""

    def role(m):
        target = m.group(2).lstrip("~")
        short = target.split(".")[-1]
        label = f"`{short}()`" if m.group(1) in {"func", "meth"} else f"`{short}`"
        return f"[{label}]({links[short]})" if short in links else label

    text = re.sub(r":(\w+):`([^`]+)`", role, text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    return text


def _sig_text(obj, name: str) -> str:
    import griffe

    func = obj
    prefix = "def"
    if obj.kind is griffe.Kind.CLASS:
        prefix = "class"
        func = obj.all_members.get("__init__") if hasattr(obj, "all_members") else obj.members.get("__init__")
    params = []
    if func is not None and hasattr(func, "parameters"):
        seen_star = False
        for p in func.parameters:
            if p.name in {"self", "cls"}:
                continue
            kind = p.kind.value if p.kind else ""
            piece = p.name
            if kind == "variadic positional":
                piece, seen_star = "*" + p.name, True
            elif kind == "variadic keyword":
                piece = "**" + p.name
            elif kind == "keyword-only" and not seen_star:
                params.append("*")
                seen_star = True
            if p.annotation is not None:
                piece += f": {p.annotation}"
            if p.default is not None and not piece.startswith("*"):
                piece += (" = " if p.annotation is not None else "=") + str(p.default)
            params.append(piece)
    ret = ""
    if prefix == "def" and getattr(func, "returns", None) is not None:
        ret = f" -> {func.returns}"
    one_line = f"{prefix} {name}({', '.join(params)}){ret}"
    if len(one_line) <= 72:
        return one_line
    return f"{prefix} {name}(\n" + "".join(f"    {p},\n" for p in params) + f"){ret}"


def _render_docstring(obj, links: dict[str, str], level: int) -> list[str]:
    import griffe

    out: list[str] = []
    if obj.docstring is None:
        return out
    h = "#" * level
    for section in obj.docstring.parse("numpy"):
        kind = section.kind
        if kind is griffe.DocstringSectionKind.text:
            out.append(_rst_to_md(section.value, links))
        elif kind in {griffe.DocstringSectionKind.parameters,
                      griffe.DocstringSectionKind.other_parameters}:
            title = "Parameters" if kind is griffe.DocstringSectionKind.parameters else "Other parameters"
            items = []
            for p in section.value:
                ann = f" (`{p.annotation}`)" if p.annotation else ""
                desc = _rst_to_md(p.description or "", links).replace("\n", "\n  ")
                items.append(f"- **`{p.name}`**{ann}" + (f": {desc}" if desc else ""))
            out.append(f"{h} {title}\n\n" + "\n".join(items))
        elif kind in {griffe.DocstringSectionKind.returns, griffe.DocstringSectionKind.yields}:
            title = "Returns" if kind is griffe.DocstringSectionKind.returns else "Yields"
            items = []
            for r in section.value:
                label = f"**`{r.name}`** " if r.name else ""
                ann = f"`{r.annotation}`" if r.annotation else ""
                desc = _rst_to_md(r.description or "", links).replace("\n", "\n  ")
                head = (label + ann).strip()
                items.append(f"- {head}" + (f": {desc}" if desc and head else desc))
            out.append(f"{h} {title}\n\n" + "\n".join(items))
        elif kind in {griffe.DocstringSectionKind.raises, griffe.DocstringSectionKind.warns}:
            title = "Raises" if kind is griffe.DocstringSectionKind.raises else "Warns"
            items = [f"- `{e.annotation}`: {_rst_to_md(e.description or '', links)}" for e in section.value]
            out.append(f"{h} {title}\n\n" + "\n".join(items))
        elif kind is griffe.DocstringSectionKind.examples:
            parts = []
            for sub_kind, value in section.value:
                if sub_kind is griffe.DocstringSectionKind.examples:
                    parts.append(_code_block(value, "python"))
                else:
                    parts.append(_rst_to_md(value, links))
            out.append(f"{h} Examples\n\n" + "\n\n".join(parts))
        elif kind is griffe.DocstringSectionKind.admonition:
            adm = section.value
            title = (section.title or adm.kind or "Note").strip().capitalize()
            out.append(f"{h} {title}\n\n{_rst_to_md(adm.contents, links)}")
        elif kind is griffe.DocstringSectionKind.attributes:
            items = [f"- **`{a.name}`**" + (f" (`{a.annotation}`)" if a.annotation else "")
                     + (f": {_rst_to_md(a.description or '', links)}" if a.description else "")
                     for a in section.value]
            out.append(f"{h} Attributes\n\n" + "\n".join(items))
    return out


def _public_methods(cls) -> list:
    import griffe

    methods = []
    for name, member in cls.members.items():
        if name.startswith("_") or member.is_alias:
            continue
        if member.kind is griffe.Kind.FUNCTION and member.docstring is not None:
            methods.append(member)
    return methods


def render_reference(dest_dir: Path) -> int:
    import griffe
    import peyesim

    pkg = griffe.load("peyesim", search_paths=[str(REPO / "src")], docstring_parser="numpy",
                      resolve_aliases=True, resolve_external=False)
    links = _reference_index()
    dest_dir.mkdir(parents=True, exist_ok=True)

    index = ["---", "title: API reference",
             "description: Every public function and class, grouped by task.", "---", ""]
    pages = []
    for slug, title, description, names in REFERENCE_SECTIONS:
        pages.append(slug)
        # No link inside the heading: Fumadocs renders heading links without
        # the site's base path.
        index.append(f"## {title}\n\n{description} [Section page](/reference/{slug}).\n")
        index.append("\n".join(f"- [`{n}`]({links[n]})" for n in names) + "\n")

        front = f"---\ntitle: {json.dumps(title)}\ndescription: {json.dumps(description)}\n---\n\n"
        body = []
        for name in names:
            if name not in peyesim.__all__:
                raise SystemExit(f"build_docs: {name} is listed in REFERENCE_SECTIONS but not exported")
            # Resolve through the live object: a name can also be a submodule
            # (peyesim.similarity is both a module and a function).
            live = getattr(peyesim, name)
            obj = pkg[f"{live.__module__}.{live.__qualname__}".removeprefix("peyesim.")]
            obj = obj.final_target if obj.is_alias else obj
            body.append(f"## {name}\n")
            body.append(_code_block(_sig_text(obj, name), "python"))
            path = obj.filepath.relative_to(REPO).as_posix() if obj.filepath else None
            if path:
                body.append(f"[Source](https://github.com/bbuchsbaum/peyesim/blob/main/{path}#L{obj.lineno})")
            body += _render_docstring(obj, links, level=3)
            if obj.kind is griffe.Kind.CLASS:
                for method in _public_methods(obj):
                    body.append(f"### {name}.{method.name}\n")
                    body.append(_code_block(_sig_text(method, method.name), "python"))
                    body += _render_docstring(method, links, level=4)
            body.append("")
        (dest_dir / f"{slug}.md").write_text(front + "\n\n".join(body), encoding="utf-8")

    (dest_dir / "index.md").write_text("\n".join(index), encoding="utf-8")
    _write_legacy_reference_redirects(links)
    (dest_dir / "meta.json").write_text(json.dumps({"title": "API reference", "pages": pages}, indent=2) + "\n")
    return sum(len(n) for *_, n in REFERENCE_SECTIONS)


def _write_legacy_reference_redirects(links: dict[str, str]) -> None:
    """Redirect the Quarto site's per-object pages (reference/<name>.html).

    The targets are relative, so the pages work under any base path.
    """
    LEGACY.mkdir(parents=True, exist_ok=True)
    for name, url in links.items():
        page, anchor = url.removeprefix("/reference/").split("#")
        target = f"./{page}/#{anchor}"
        (LEGACY / f"{name}.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Moved</title>'
            f'<meta http-equiv="refresh" content="0; url={target}">'
            f'<a href="{target}">{name}</a>\n',
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO / "src"))
    for generated in (CONTENT, LEGACY):
        if generated.exists():
            shutil.rmtree(generated)
    CONTENT.mkdir(parents=True)

    total = 0
    for src in sorted(PAGES.rglob("*")):
        if src.is_dir():
            continue
        dest = CONTENT / src.relative_to(PAGES)
        if src.suffix == ".md":
            # chdir per page so relative paths in a page behave consistently
            cwd = os.getcwd()
            os.chdir(REPO)
            try:
                total += render_page(src, dest)
            finally:
                os.chdir(cwd)
            print(f"  {src.relative_to(REPO)}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    n_ref = render_reference(CONTENT / "reference")
    print(f"build_docs: {total} cells executed; {n_ref} reference entries -> {CONTENT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
