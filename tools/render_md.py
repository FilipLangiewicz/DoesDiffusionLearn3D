"""Render a Markdown file with its mermaid diagrams in Chromium (Playwright) and take a screenshot.

The script also reports the mermaid blocks that do not parse and the images that do
not load. It loads marked and mermaid 11 from a CDN, so it needs a network.

Usage: python tools/render_md.py docs/RESULTS.md runs/tmp/results  (writes runs/tmp/results.png)
"""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

md = Path(sys.argv[1]).resolve()
out = sys.argv[2]
html = f"""<!doctype html><html><head><meta charset="utf-8"><base href="file://{md.parent}/">
<style>body{{font-family:sans-serif;max-width:900px;margin:20px auto;font-size:14px}} img{{max-width:100%}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:3px 6px}}</style>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script></head>
<body><div id="c"></div><script>
const src = {json.dumps(md.read_text())};
const r = new marked.Renderer();
r.code = (code, lang) => lang === 'mermaid'
  ? '<pre class="mermaid">' + code + '</pre>' : '<pre>' + code + '</pre>';
document.getElementById('c').innerHTML = marked.parse(src, {{renderer: r}});
window.errs = [];
mermaid.initialize({{startOnLoad: false}});
(async () => {{
  for (const el of document.querySelectorAll('.mermaid')) {{
    try {{ await mermaid.parse(el.textContent); }} catch (e) {{ window.errs.push(String(e).slice(0, 200)); }}
  }}
  await mermaid.run();
  window.done = true;
}})();
</script></body></html>"""
page_file = Path(out + ".html")
page_file.write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1000, "height": 1200})
    pg.goto(f"file://{page_file.resolve()}")
    pg.wait_for_function("window.done === true", timeout=60000)
    pg.wait_for_timeout(500)
    print("mermaid blocks:", pg.evaluate("document.querySelectorAll('.mermaid').length"))
    print("mermaid errors:", pg.evaluate("window.errs"))
    print(
        "broken images:",
        pg.evaluate("[...document.images].filter(i => !i.naturalWidth).map(i => i.getAttribute('src'))"),
    )
    pg.screenshot(path=out + ".png", full_page=True)
    b.close()
