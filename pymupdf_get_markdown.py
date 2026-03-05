import pymupdf4llm
import json
from pathlib import Path

md_text = pymupdf4llm.to_markdown("Data/Psychology2e_WEB.pdf", page_chunks=True)


def _json_default(obj):
    # Handle PyMuPDF geometry objects like Rect/Point/Matrix.
    name = obj.__class__.__name__
    if name == "Rect" and all(hasattr(obj, k) for k in ("x0", "y0", "x1", "y1")):
        return {
            "x0": float(obj.x0),
            "y0": float(obj.y0),
            "x1": float(obj.x1),
            "y1": float(obj.y1),
        }

    # Fallback for other non-JSON-native objects.
    return str(obj)

out_path = Path("Data/Psychology2e_WEB_with_page_numbers.json")
tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(md_text, f, ensure_ascii=False, indent=4, default=_json_default)

tmp_path.replace(out_path)