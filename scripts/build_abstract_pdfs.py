"""
Build symposium-quality PDF abstracts from reports/abstract_{en,tr}.md.

Uses pandoc + xelatex with a conservative academic template:
  - A4, 1-inch margins, single column
  - TeX Gyre Termes (Times-like) for body, sans for metadata
  - Small caps title block, justified body, widow/orphan control

Call signature:
    python scripts/build_abstract_pdfs.py

Writes:
    reports/abstract_en.pdf
    reports/abstract_tr.pdf
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

PREAMBLE = r"""
\usepackage[a4paper, margin=1in]{geometry}
\usepackage{microtype}
\usepackage{parskip}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\usepackage{titling}
\setlength{\droptitle}{-2em}
\pretitle{\begin{center}\Large\bfseries}
\posttitle{\par\end{center}\vskip 0.5em}
\preauthor{\begin{center}\normalsize}
\postauthor{\par\end{center}}
\predate{\begin{center}\small\itshape}
\postdate{\par\end{center}\vskip 1em}
\renewcommand{\baselinestretch}{1.12}
\widowpenalty=10000
\clubpenalty=10000
\usepackage{enumitem}
\setlist{topsep=0.3em, itemsep=0.1em}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdfcreator={pandoc}, pdfproducer={xelatex}}
\usepackage{csquotes}
"""


def build_one(md_path: Path, pdf_path: Path, lang: str) -> None:
    if not md_path.exists():
        print(f"  [skip] missing {md_path.name}")
        return
    md = md_path.read_text(encoding="utf-8")
    # strip first H1 (pandoc will use --metadata title instead)
    lines = md.splitlines()
    title = ""
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and lines[i].startswith("# "):
        title = lines[i][2:].strip()
        i += 1
    body = "\n".join(lines[i:]).lstrip()

    with tempfile.TemporaryDirectory() as td:
        preamble_path = Path(td) / "preamble.tex"
        preamble_path.write_text(PREAMBLE, encoding="utf-8")
        body_path = Path(td) / "body.md"
        body_path.write_text(body, encoding="utf-8")

        cmd = [
            "pandoc",
            str(body_path),
            "-o", str(pdf_path),
            "--pdf-engine=xelatex",
            "-H", str(preamble_path),
            "-V", f"lang={lang}",
            "-V", "papersize=a4",
            "-V", "fontsize=11pt",
            "--metadata", f"title={title}",
            "--standalone",
        ]
        print(f"  [build] {md_path.name} -> {pdf_path.name}  ({lang}, title: "
              f"{title[:60]}{'...' if len(title) > 60 else ''})")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [error] pandoc failed:\n{res.stderr}", file=sys.stderr)
            raise SystemExit(1)
        size_kb = pdf_path.stat().st_size / 1024
        print(f"    wrote {size_kb:.0f} KB")


def main() -> None:
    if shutil.which("pandoc") is None:
        print("error: pandoc is not on PATH", file=sys.stderr)
        raise SystemExit(1)

    build_one(REPORTS / "abstract_en.md",
              REPORTS / "abstract_en.pdf", lang="en")
    build_one(REPORTS / "abstract_tr.md",
              REPORTS / "abstract_tr.pdf", lang="tr")


if __name__ == "__main__":
    main()
