import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


NOISE_RE = re.compile(r"^(Page\s+\d+|Access for free at openstax\.org|FIGURE\s+\d+\.\d+|LINK TO LEARNING|DIG DEEPER)$", re.IGNORECASE)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def slug(heading, sub_heading):
    return re.sub(r"[^a-z0-9]+", "_", f"{heading}/{sub_heading or heading}".lower()).strip("_")


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_toc(path):
    with Path(path).open("r", encoding="utf-8") as f:
        toc = json.load(f)
    toc.sort(key=lambda x: x["page"])
    return toc


def chapter_for(page, toc):
    chapter = "Preface"
    for item in toc:
        if item["page"] > page:
            break
        if item["type"] == "chapter":
            chapter = item["title"]
    return chapter


def prev_section(page, toc):
    section = ""
    for item in toc:
        if item["page"] >= page:
            break
        if item["type"] == "section":
            section = item["title"]
    return section


def same_page_sections(page, toc):
    return [i["title"] for i in toc if i["type"] == "section" and int(i["page"]) == page]


def text_from_lines(lines):
    clean = []
    for line in lines:
        y0, y1 = float(line["bbox"].get("y0", 0.0)), float(line["bbox"].get("y1", 0.0))
        txt = norm(line.get("text", ""))
        if txt and y1 < 748 and y0 > 35 and not NOISE_RE.match(txt):
            clean.append(txt)
    return norm(" ".join(clean))


def split_blocks(page_text, sections, fallback):
    if not sections:
        return [(fallback, page_text)]
    markers = []
    for title in sections:
        pattern = re.compile(r"\b" + r"\s+".join(re.escape(w) for w in title.split()) + r"\b", re.IGNORECASE)
        m = pattern.search(page_text)
        if m:
            markers.append((m.start(), title))
    if not markers:
        return [(sections[-1], page_text)]
    markers.sort(key=lambda x: x[0])
    out = []
    prefix = norm(page_text[: markers[0][0]])
    if prefix and fallback:
        out.append((fallback, prefix))
    for idx, (start, title) in enumerate(markers):
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(page_text)
        chunk = norm(page_text[start:end])
        if chunk:
            out.append((title, chunk))
    return out or [(sections[-1], page_text)]


def build_entries(records, toc, skip_before_page):
    pages = defaultdict(list)
    for rec in records:
        p = int(rec.get("page_num", 0))
        if p >= skip_before_page:
            pages[p].append(rec)

    entries = []
    for page in sorted(pages):
        lines = sorted(pages[page], key=lambda x: (-float(x["bbox"]["y1"]), float(x["bbox"]["x0"])))
        page_text = text_from_lines(lines)
        for section, data in split_blocks(page_text, same_page_sections(page, toc), prev_section(page, toc)):
            entries.append({"page": page, "heading": chapter_for(page, toc), "sub_heading": section, "data": data})
    return entries


def write_outputs(entries, out_page_section, out_kaggle):
    with Path(out_page_section).open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)
    rows = []
    for idx, row in enumerate(entries, start=1):
        rows.append({
            "id": idx,
            "context": row["data"],
            "metadata": {
                "section": slug(row["heading"], row["sub_heading"]),
                "page": str(row["page"]),
                "heading": row["heading"],
                "sub_heading": row["sub_heading"],
            },
        })
    with Path(out_kaggle).open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-lines", default="Data/Psychology2e_layout_lines.jsonl")
    ap.add_argument("--toc", default="Data/toc_pages.json")
    ap.add_argument("--out-page-section", default="Data/page_section_from_layout_modular.json")
    ap.add_argument("--out-kaggle", default="Data/kaggle_section_records_modular.json")
    ap.add_argument("--skip-before-page", type=int, default=1)
    a = ap.parse_args()

    entries = build_entries(load_jsonl(a.in_lines), load_toc(a.toc), a.skip_before_page)
    write_outputs(entries, a.out_page_section, a.out_kaggle)
    print(f"Wrote: {a.out_page_section}")
    print(f"Wrote: {a.out_kaggle}")
    print(f"Total entries generated: {len(entries)}")


if __name__ == "__main__":
    main()
