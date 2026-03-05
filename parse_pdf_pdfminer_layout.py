import argparse
import json
from statistics import mean
from typing import Any, Dict, List, Optional

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTChar, LTTextContainer, LTTextLine
from tqdm import tqdm


def _line_font_features(line: LTTextLine) -> Dict[str, Any]:
    chars = [obj for obj in line if isinstance(obj, LTChar)]
    if not chars:
        return {
            "avg_font_size": None,
            "max_font_size": None,
            "font_names": [],
            "is_bold_like": False,
        }

    sizes = [float(ch.size) for ch in chars if getattr(ch, "size", None) is not None]
    font_names = [str(ch.fontname) for ch in chars if getattr(ch, "fontname", None)]
    unique_fonts = sorted(set(font_names))

    return {
        "avg_font_size": round(mean(sizes), 3) if sizes else None,
        "max_font_size": round(max(sizes), 3) if sizes else None,
        "font_names": unique_fonts,
        "is_bold_like": any("Bold" in f or "Black" in f or "Semibold" in f for f in unique_fonts),
    }


def _textline_to_record(line: LTTextLine, page_num: int, block_idx: int, line_idx: int) -> Optional[Dict[str, Any]]:
    text = line.get_text().strip()
    if not text:
        return None

    features = _line_font_features(line)
    x0, y0, x1, y1 = [round(v, 3) for v in line.bbox]

    return {
        "page_num": page_num,
        "block_idx": block_idx,
        "line_idx": line_idx,
        "text": text,
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "width": round(x1 - x0, 3),
        "height": round(y1 - y0, 3),
        "avg_font_size": features["avg_font_size"],
        "max_font_size": features["max_font_size"],
        "font_names": features["font_names"],
        "is_bold_like": features["is_bold_like"],
        "char_count": len(text),
        "is_all_caps": text.isupper(),
    }


def extract_layout_records(
    pdf_path: str,
    skip_first_n_pages: int,
    stop_after_page: int,
    laparams: LAParams,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    scanned_likelihood_per_page: List[Dict[str, Any]] = []

    pages_iter = extract_pages(pdf_path, laparams=laparams)
    for raw_page_num, page_layout in enumerate(tqdm(pages_iter, desc="Parsing pages"), start=1):
        if raw_page_num <= skip_first_n_pages:
            continue
        if raw_page_num > stop_after_page:
            break

        relative_page_num = raw_page_num - skip_first_n_pages
        page_text_chars = 0
        text_lines_on_page = 0

        block_idx = 0
        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue

            line_idx = 0
            for line in element:
                if not isinstance(line, LTTextLine):
                    continue
                rec = _textline_to_record(line, relative_page_num, block_idx, line_idx)
                line_idx += 1
                if rec is None:
                    continue

                page_text_chars += rec["char_count"]
                text_lines_on_page += 1
                records.append(rec)

            block_idx += 1

        scanned_likelihood_per_page.append(
            {
                "page_num": relative_page_num,
                "raw_page_num": raw_page_num,
                "text_char_count": page_text_chars,
                "text_line_count": text_lines_on_page,
                "likely_scanned": page_text_chars < 100,
            }
        )

    return {
        "records": records,
        "page_stats": scanned_likelihood_per_page,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="Data/Psychology2e_WEB.pdf")
    parser.add_argument("--out-lines", default="Data/Psychology2e_layout_lines.jsonl")
    parser.add_argument("--out-pages", default="Data/Psychology2e_layout_page_stats.json")
    parser.add_argument("--skip-first", type=int, default=12)
    parser.add_argument("--stop-after", type=int, default=644)
    parser.add_argument("--line-margin", type=float, default=0.5)
    parser.add_argument("--char-margin", type=float, default=2.0)
    parser.add_argument("--word-margin", type=float, default=0.1)
    args = parser.parse_args()

    laparams = LAParams(
        line_margin=args.line_margin,
        char_margin=args.char_margin,
        word_margin=args.word_margin,
    )

    result = extract_layout_records(
        pdf_path=args.pdf,
        skip_first_n_pages=args.skip_first,
        stop_after_page=args.stop_after,
        laparams=laparams,
    )

    with open(args.out_lines, "w", encoding="utf-8") as f:
        for rec in result["records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "pdf": args.pdf,
        "total_lines": len(result["records"]),
        "total_pages": len(result["page_stats"]),
        "likely_scanned_pages": [p["page_num"] for p in result["page_stats"] if p["likely_scanned"]],
        "page_stats": result["page_stats"],
    }

    with open(args.out_pages, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote line-level layout data to: {args.out_lines}")
    print(f"Wrote page-level summary to: {args.out_pages}")


if __name__ == "__main__":
    main()
