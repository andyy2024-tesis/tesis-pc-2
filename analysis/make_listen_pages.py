"""Build the extra listening pages from analysis/results/slides/listen/index.html (one artifact version holds at most
511 files, so sections with `page: N` go on page N).

For each N given: writes listen/pagina<N>.html (same code, PAGE_NO = N, own title) and listen/files_pagina<N>.json, the
list of audio files that page plays (original and every version of its sections, plus the per-part stems of those models),
relative to listen/, for the publish call.
Usage: uv run python analysis/make_listen_pages.py 2 3
"""
import json
import re
import sys

from tesis.data.gaps import REPO

LISTEN = REPO / "analysis" / "results" / "slides" / "listen"
RECS = ["023_Ypswc", "176_mM1wc", "276_jw1wc", "277_Pw1wc", "306_7c1wc"]
TITLES = {2: ("Ataque y ruido", "El ataque y el ruido: noche del 15 al 16 de noviembre"),
          3: ("Redes y cuerdas", "Redes, cuerdas y contacto: noche del 15 al 16 de noviembre")}


def sections(html):
    body = html[html.index("const ALL_SECTIONS = ["):]
    body = body[:body.index("\n];")]
    out = []
    for chunk in body.split("\n  { id: ")[1:]:
        page = re.search(r"\bpage: (\d+)", chunk)
        files = []
        for key in ("v", "panel"):
            m = re.search(rf"\b{key}: \[([^\]]*)\]", chunk)
            if m:
                files += re.findall(r'"([^"]+)"', m.group(1))
        out.append((int(page.group(1)) if page else 1, files))
    return out


def main(pages):
    html = (LISTEN / "index.html").read_text()
    assert "const PAGE_NO = 1;" in html
    secs = sections(html)
    for n in pages:
        files = sorted({f for p, fs in secs if p == n for f in fs} | {"original"})
        paths = []
        for rid in RECS:
            for f in files:
                p = LISTEN / "audio" / rid / f"{f}.mp4"
                assert p.exists(), p
                paths.append(p)
                if not f.endswith(("-nonoise", "-noise")):
                    paths += sorted((LISTEN / "audio" / rid / "stems" / f).glob("*.mp4"))
        rel = [str(p.relative_to(LISTEN)) for p in paths]
        assert len(rel) <= 500, f"page {n}: {len(rel)} files, over the 511-file limit with the page itself"
        title, h1 = TITLES[n]
        page = html.replace("const PAGE_NO = 1;", f"const PAGE_NO = {n};")
        page = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", page, count=1)
        page = re.sub(r"<h1>.*?</h1>", f"<h1>{h1}</h1>", page, count=1)
        (LISTEN / f"pagina{n}.html").write_text(page)
        (LISTEN / f"files_pagina{n}.json").write_text(json.dumps([{"path": r} for r in rel]))
        print(f"page {n}: {len(files)} versions, {len(rel)} files")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
