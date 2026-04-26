"""
Build symposium-quality PDF abstracts from reports/abstract_{en,tr}.md.

Uses pandoc + xelatex with an academic template:
  - A4, 0.9-inch margins
  - Times New Roman 11pt body, justified, 1.15 line spacing
  - Centred title with a thin rule underneath
  - Section headings rendered small-caps with a subtle accent colour,
    bold lead-in paragraphs naturally separated
  - No author line (strip any '**Author ...**' block from the md)
  - Proper widow / orphan control, microtype for clean line breaks

Call:
    python scripts/build_abstract_pdfs.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

PREAMBLE = r"""
\usepackage[a4paper, margin=0.9in]{geometry}
\usepackage{microtype}
\usepackage{setspace}
\setstretch{1.15}

\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}

% paragraph spacing: no indent, small space between paragraphs
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}

% widow/orphan control
\widowpenalty=10000
\clubpenalty=10000

% title block: compact, centred, rule underneath
\usepackage{titling}
\setlength{\droptitle}{-1.5em}
\pretitle{\begin{center}\large\bfseries}
\posttitle{\par\vskip 0.2em\hrule height 0.4pt\par\end{center}\vskip 1.2em}
\preauthor{}
\postauthor{}
\predate{}
\postdate{}

% clean, lightly-styled section headers
\usepackage[explicit]{titlesec}
\usepackage{xcolor}
\definecolor{seccol}{HTML}{1f3a5f}
\titleformat{\section}
  {\normalfont\normalsize\bfseries\color{seccol}}
  {}{0pt}{#1}[\vskip 0.15em]
\titlespacing{\section}{0pt}{0.9em}{0.25em}

% lists
\usepackage{enumitem}
\setlist{topsep=0.25em, itemsep=0.1em, leftmargin=1.6em}

% justified text
\sloppy

% hyperlinks
\usepackage[hidelinks]{hyperref}
\hypersetup{pdfcreator={pandoc}, pdfproducer={xelatex}}

% small keywords-style footer emphasis
\usepackage{csquotes}
"""


# paragraphs like `**Kadir Göksel Gündüz¹, ...**` or a single-line
# affiliation `¹ ... ² ...` are stripped before handing off to pandoc
AUTHOR_PAT = re.compile(
    r"^\*\*[^*]*(Gündüz|[Ii]şbirlikçi|Collaborator|Advisor|Danışman)[^*]*\*\*\s*$",
    re.MULTILINE,
)
AFFIL_PAT = re.compile(
    r"^¹[^\n]*$", re.MULTILINE,
)
HR_PAT = re.compile(r"^---\s*$", re.MULTILINE)


def strip_author_block(md: str) -> str:
    """Remove any leftover author/affiliation lines from the body.
    The new markdown already omits them, but this stays as a safety
    net so older copies don't leak names into the PDF."""
    md = AUTHOR_PAT.sub("", md)
    md = AFFIL_PAT.sub("", md)
    # collapse resulting triple blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


def build_one(md_path: Path, pdf_path: Path, lang: str) -> None:
    if not md_path.exists():
        print(f"  [skip] missing {md_path.name}")
        return
    md = md_path.read_text(encoding="utf-8")

    lines = md.splitlines()
    title = ""
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and lines[i].startswith("# "):
        title = lines[i][2:].strip()
        i += 1
    body = "\n".join(lines[i:]).lstrip()
    body = strip_author_block(body)

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
        print(f"  [build] {md_path.name} -> {pdf_path.name}  ({lang})")
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

    # split abstracts (per advisor's request: A and B as separate
    # 1-page submissions). Build all four; build_one tolerates missing
    # source files so partial runs still work.
    build_one(REPORTS / "abstract_a_en.md",
              REPORTS / "abstract_a_en.pdf", lang="en")
    build_one(REPORTS / "abstract_a_tr.md",
              REPORTS / "abstract_a_tr.pdf", lang="tr")
    build_one(REPORTS / "abstract_b_en.md",
              REPORTS / "abstract_b_en.pdf", lang="en")
    build_one(REPORTS / "abstract_b_tr.md",
              REPORTS / "abstract_b_tr.pdf", lang="tr")

    # legacy combined abstracts — keep producing if the source still
    # exists locally, otherwise skip silently
    build_one(REPORTS / "abstract_en.md",
              REPORTS / "abstract_en.pdf", lang="en")
    build_one(REPORTS / "abstract_tr.md",
              REPORTS / "abstract_tr.pdf", lang="tr")


if __name__ == "__main__":
    main()
