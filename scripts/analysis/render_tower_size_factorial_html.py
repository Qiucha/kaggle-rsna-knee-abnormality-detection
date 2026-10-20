#!/usr/bin/env python3
"""Render the tower-size Markdown record as a styled digest page.

Reads: ``digest/tower-size-factorial-v1.md``.
Writes: ``digest/tower-size-factorial-v1.html``.
Run: ``.venv/bin/python scripts/analysis/render_tower_size_factorial_html.py``
"""

from __future__ import annotations

from pathlib import Path

import argparse
import markdown

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "digest/tower-size-factorial-v1.md"
OUTPUT = ROOT / "digest/tower-size-factorial-v1.html"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    body = markdown.markdown(SOURCE.read_text(), extensions=["tables", "toc"])
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image and text tower size comparison</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #183047; background: #f5f7fa; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
header {{ background: linear-gradient(125deg, #122e4f, #236181); color: white; padding: 3rem max(1.5rem, calc((100vw - 1040px) / 2)); }}
header p {{ max-width: 720px; color: #d6e9f0; line-height: 1.6; }}
main {{ max-width: 1100px; margin: -1.3rem auto 3rem; padding: 2.2rem; border-radius: 16px; background: white; box-shadow: 0 12px 38px #17325014; }}
h1 {{ font-size: clamp(2rem, 4vw, 3rem); line-height: 1.1; margin: 0; }}
main > h1 {{ display: none; }}
h2 {{ color: #173d60; margin: 2.1rem 0 .8rem; border-bottom: 1px solid #dce6ed; padding-bottom: .5rem; }}
h3 {{ color: #275574; }}
p, li {{ line-height: 1.65; }}
table {{ border-collapse: collapse; width: 100%; display: block; overflow-x: auto; margin: 1.1rem 0 1.5rem; }}
th, td {{ padding: .78rem .9rem; border-bottom: 1px solid #dce6ed; text-align: left; white-space: nowrap; }}
th {{ background: #eaf2f6; color: #214661; }}
tr:nth-child(even) td {{ background: #f8fafc; }}
a {{ color: #086a9c; }}
strong {{ color: #163d5e; }}
code {{ background: #eaf2f6; border-radius: 4px; padding: .1em .28em; }}
img {{ max-width: 100%; height: auto; }}
blockquote {{ border-left: 4px solid #3a91af; margin: 1.2rem 0; padding: .1rem 1rem; background: #f0f8fa; }}
@media (max-width: 620px) {{ main {{ margin: 0; padding: 1.2rem; border-radius: 0; }} header {{ padding: 2rem 1.2rem; }} }}
</style>
</head>
<body>
<header><h1>Image and text tower size</h1><p>Four pinned model combinations, the original 56-case gold-only probe, and the 58-case LM-head follow-up.</p></header>
<main>{body}</main>
</body>
</html>
"""
    OUTPUT.write_text(page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
