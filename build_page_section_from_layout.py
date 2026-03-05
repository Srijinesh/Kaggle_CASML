import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


NOISE_RE = re.compile(
    r"^(Page\s+\d+|Access for free at openstax\.org|FIGURE\s+\d+\.\d+|LINK TO LEARNING|DIG DEEPER)$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_lines(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_toc(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    entries.sort(key=lambda x: x["page"])
    return entries


def is_header_footer(line: Dict[str, Any]) -> bool:
    y0 = float(line["bbox"].get("y0", 0.0))
    y1 = float(line["bbox"].get("y1", 0.0))
    text = normalize_text(line.get("text", ""))
    return y1 >= 748 or y0 <= 35 or bool(NOISE_RE.match(text))


def labels_for_page(page: int, toc_entries: List[Dict[str, Any]]) -> Tuple[str, str]:
    chapter = "Preface"
    section = ""

    for item in toc_entries:
        if item["page"] > page:
            break
        if item["type"] == "chapter":
            chapter = item["title"]
            section = ""
        else:
            section = item["title"]

    return chapter, section


def chapter_for_page(page: int, toc_entries: List[Dict[str, Any]]) -> str:
    chapter = "Preface"
    for item in toc_entries:
        if item["page"] > page:
            break
        if item["type"] == "chapter":
            chapter = item["title"]
    return chapter


def previous_section_before_page(page: int, toc_entries: List[Dict[str, Any]]) -> str:
    section = ""
    for item in toc_entries:
        if item["page"] >= page:
            break
        if item["type"] == "section":
            section = item["title"]
    return section


def sections_starting_on_page(page: int, toc_entries: List[Dict[str, Any]]) -> List[str]:
    return [item["title"] for item in toc_entries if item["type"] == "section" and int(item["page"]) == page]


def find_title_position(text: str, title: str) -> int:
    words = [re.escape(w) for w in title.split()]
    if not words:
        return -1
    pattern = re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE)
    match = pattern.search(text)
    return match.start() if match else -1


def split_page_into_sections(page_text: str, same_page_sections: List[str], fallback_section: str) -> List[Tuple[str, str]]:
    if not same_page_sections:
        return [(fallback_section, page_text)]

    markers: List[Tuple[int, str]] = []
    for title in same_page_sections:
        pos = find_title_position(page_text, title)
        if pos >= 0:
            markers.append((pos, title))

    if not markers:
        return [(same_page_sections[-1], page_text)]

    markers.sort(key=lambda x: x[0])
    blocks: List[Tuple[str, str]] = []

    first_pos = markers[0][0]
    prefix = normalize_text(page_text[:first_pos])
    if prefix and fallback_section:
        blocks.append((fallback_section, prefix))

    for i, (start, title) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(page_text)
        chunk = normalize_text(page_text[start:end])
        if chunk:
            blocks.append((title, chunk))

    return blocks or [(same_page_sections[-1], page_text)]


def build_entries(records: List[Dict[str, Any]], toc_entries: List[Dict[str, Any]], skip_before_page: int) -> List[Dict[str, Any]]:
    page_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        page = int(rec.get("page_num", 0))
        if page >= skip_before_page:
            page_map[page].append(rec)

    entries: List[Dict[str, Any]] = []
    for page in sorted(page_map):
        lines = sorted(page_map[page], key=lambda x: (-float(x["bbox"]["y1"]), float(x["bbox"]["x0"])))
        chapter = chapter_for_page(page, toc_entries)
        fallback_section = previous_section_before_page(page, toc_entries)
        same_page_sections = sections_starting_on_page(page, toc_entries)

        content_lines = [
            normalize_text(line.get("text", ""))
            for line in lines
            if not is_header_footer(line) and normalize_text(line.get("text", ""))
        ]
        page_text = normalize_text(" ".join(content_lines))

        split_blocks = split_page_into_sections(page_text, same_page_sections, fallback_section)

        for section, chunk_text in split_blocks:
            entries.append(
                {
                    "page": page,
                    "heading": chapter,
                    "sub_heading": section,
                    "data": chunk_text,
                }
            )

    return entries


def section_slug(heading: str, sub_heading: str) -> str:
    raw = f"{heading}/{sub_heading or heading}".lower()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_")


def write_outputs(entries: List[Dict[str, Any]], out_page_section: Path, out_kaggle: Path) -> None:
    with out_page_section.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    kaggle_rows = []
    for idx, row in enumerate(entries, start=1):
        kaggle_rows.append(
            {
                "id": idx,
                "context": row["data"],
                "metadata": {
                    "section": section_slug(row["heading"], row["sub_heading"]),
                    "page": str(row["page"]),
                    "heading": row["heading"],
                    "sub_heading": row["sub_heading"],
                },
            }
        )

    with out_kaggle.open("w", encoding="utf-8") as f:
        json.dump(kaggle_rows, f, indent=4, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-lines", default="Data/Psychology2e_layout_lines.jsonl")
    parser.add_argument("--toc", default="Data/toc_pages.json")
    parser.add_argument("--out-page-section", default="Data/page_section_from_layout.json")
    parser.add_argument("--out-kaggle", default="Data/kaggle_section_records.json")
    parser.add_argument("--skip-before-page", type=int, default=1)
    args = parser.parse_args()

    records = load_lines(Path(args.in_lines))
    toc_entries = load_toc(Path(args.toc))
    entries = build_entries(records, toc_entries, args.skip_before_page)
    write_outputs(entries, Path(args.out_page_section), Path(args.out_kaggle))

    print(f"Wrote: {args.out_page_section}")
    print(f"Wrote: {args.out_kaggle}")
    print(f"Total entries generated: {len(entries)}")


if __name__ == "__main__":
    main()
