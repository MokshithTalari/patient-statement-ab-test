"""build_dashboard.py - injects results.json into dashboard_template.html -> index.html"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
html = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
results = (ROOT / "results.json").read_text(encoding="utf-8")
(ROOT / "index.html").write_text(html.replace("/*RESULTS_JSON*/null", results), encoding="utf-8")
print("Wrote index.html")
