"""
Build a self-contained HTML report from reports/sonuclar_hoca.md.

Images are inlined as base64 data URIs so the output HTML is a single
portable file. Open in a browser and Ctrl+P -> Save as PDF for a PDF
version (no extra dependency needed).
"""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "reports" / "sonuclar_hoca.md"
OUT_HTML = ROOT / "reports" / "sonuclar_hoca.html"


def image_to_data_uri(img_path: Path) -> str:
    mime, _ = mimetypes.guess_type(img_path.name)
    mime = mime or "image/png"
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def inline_images(md_text: str, md_dir: Path) -> str:
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def repl(match: re.Match) -> str:
        alt = match.group(1)
        path_str = match.group(2)
        img_path = (md_dir / path_str).resolve()
        if not img_path.exists():
            print(f"[warn] missing image: {img_path}")
            return match.group(0)
        uri = image_to_data_uri(img_path)
        return f"![{alt}]({uri})"

    return pattern.sub(repl, md_text)


CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         max-width: 900px; margin: 2em auto; padding: 0 1.5em;
         color: #222; line-height: 1.55; }
  h1 { border-bottom: 2px solid #222; padding-bottom: .3em; }
  h2 { border-bottom: 1px solid #bbb; padding-bottom: .2em; margin-top: 2em; }
  h3 { margin-top: 1.4em; }
  table { border-collapse: collapse; margin: 1em 0; width: 100%; }
  th, td { border: 1px solid #bbb; padding: .45em .7em; text-align: left;
           vertical-align: top; }
  th { background: #f2f4f8; }
  img { max-width: 100%; height: auto; display: block; margin: .8em auto;
        border: 1px solid #ddd; border-radius: 4px; }
  code { background: #f2f4f8; padding: 1px 4px; border-radius: 3px;
         font-family: Consolas, Menlo, monospace; font-size: 95%; }
  em { color: #555; }
  hr { border: 0; border-top: 1px solid #bbb; margin: 2em 0; }
  blockquote { border-left: 3px solid #bbb; color: #555;
               margin: 1em 0; padding: .1em 1em; }
  @media print {
    body { max-width: none; margin: 0; }
    img { page-break-inside: avoid; }
    h2 { page-break-before: auto; }
  }
</style>
"""


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    md_text = inline_images(md_text, MD_PATH.parent)

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "attr_list"],
    )
    html = (
        "<!DOCTYPE html>\n<html lang='tr'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>Ultrason - Sonuç Raporu</title>\n"
        f"{CSS}\n</head>\n<body>\n{html_body}\n</body>\n</html>\n"
    )
    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"[report] wrote {OUT_HTML}  ({size_kb:.0f} KB)")
    print("[report] Open in browser and Ctrl+P -> 'Save as PDF' for PDF output.")


if __name__ == "__main__":
    main()
