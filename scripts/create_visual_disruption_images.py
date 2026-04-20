import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import read_jsonl, write_jsonl


INGREDIENT_WORDS = {
    "salt", "pepper", "sugar", "butter", "milk", "cream", "oil", "olive oil",
    "vinegar", "lemon", "lemon juice", "lime", "lime juice", "egg", "eggs",
    "flour", "water", "cheese", "chicken", "beef", "garlic", "onion", "celery",
    "apple", "apples", "raisins", "cranberries", "nuts", "bread", "breadcrumbs",
    "parmesan", "sauce", "hot sauce", "mustard", "mayo", "mayonnaise",
}


def _default_inputs() -> List[Path]:
    return [
        ROOT / "data" / "processed" / "benchmark.jsonl",
        ROOT / "data" / "splits" / "train.jsonl",
        ROOT / "data" / "splits" / "dev.jsonl",
        ROOT / "data" / "splits" / "test.jsonl",
    ]


def _project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _target_ingredient(step_text: str) -> str:
    step_l = (step_text or "").lower()
    for term in sorted(INGREDIENT_WORDS, key=len, reverse=True):
        if term in step_l:
            return term
    return "key ingredient"


def _wrap(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) <= max_chars:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _font(size: int):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _make_visual_disruption(source_path: Path, output_path: Path, ingredient: str, step_text: str) -> None:
    image = Image.open(source_path).convert("RGB")
    image.thumbnail((960, 720))

    panel_width = max(300, image.width // 3)
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (248, 248, 245))
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    x0 = image.width
    draw.rectangle((x0, 0, canvas.width, canvas.height), fill=(245, 245, 242))
    draw.line((x0, 0, x0, canvas.height), fill=(40, 40, 40), width=3)

    title_font = _font(30)
    body_font = _font(22)
    small_font = _font(17)

    margin = 24
    y = 34
    draw.text((x0 + margin, y), "MISSING", fill=(150, 24, 24), font=title_font)
    y += 40
    draw.text((x0 + margin, y), ingredient.upper(), fill=(20, 20, 20), font=title_font)
    y += 66

    icon_box = (x0 + margin, y, x0 + margin + 120, y + 120)
    draw.ellipse(icon_box, outline=(150, 24, 24), width=8)
    draw.line((icon_box[0] + 18, icon_box[3] - 18, icon_box[2] - 18, icon_box[1] + 18), fill=(150, 24, 24), width=9)
    y += 152

    for line in _wrap("This ingredient is unavailable for the next step.", 23):
        draw.text((x0 + margin, y), line, fill=(20, 20, 20), font=body_font)
        y += 29
    y += 18

    draw.text((x0 + margin, y), "Step:", fill=(80, 80, 80), font=small_font)
    y += 24
    for line in _wrap(step_text, 31)[:7]:
        draw.text((x0 + margin, y), line, fill=(45, 45, 45), font=small_font)
        y += 22

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _is_visual_missing_ingredient(row: Dict) -> bool:
    return (
        row.get("disruption_type") == "missing_ingredient"
        and row.get("metadata", {}).get("disruption_modality") == "vision"
        and bool(row.get("image_path"))
    )


def _update_rows(rows: Iterable[Dict], output_dir: Path, overwrite: bool) -> tuple[List[Dict], int]:
    updated_rows: List[Dict] = []
    count = 0
    for row in rows:
        if not _is_visual_missing_ingredient(row):
            updated_rows.append(row)
            continue

        metadata = row.setdefault("metadata", {})
        original_image_path = metadata.get("original_image_path") or row["image_path"]
        source_path = _project_path(original_image_path)
        if not source_path.exists():
            updated_rows.append(row)
            continue

        output_path = output_dir / f"{row['example_id']}.jpg"
        if overwrite or not output_path.exists():
            ingredient = metadata.get("missing_ingredient") or _target_ingredient(row.get("disrupted_step_text", ""))
            _make_visual_disruption(source_path, output_path, ingredient, row.get("disrupted_step_text", ""))

        metadata["original_image_path"] = original_image_path
        metadata["visual_disruption_image_path"] = _portable_path(output_path)
        row["image_path"] = _portable_path(output_path)
        count += 1
        updated_rows.append(row)
    return updated_rows, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Create explicit visual missing-ingredient disruption images.")
    parser.add_argument("--input", dest="inputs", action="append", help="JSONL artifact to update. Can be passed multiple times.")
    parser.add_argument("--output-dir", default="data/processed/visual_disruptions")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = [Path(p).resolve() for p in args.inputs] if args.inputs else _default_inputs()
    output_dir = _project_path(args.output_dir)

    for input_path in inputs:
        if not input_path.exists():
            continue
        rows = read_jsonl(input_path)
        updated_rows, count = _update_rows(rows, output_dir, args.overwrite)
        write_jsonl(input_path, updated_rows)
        print(f"Updated {count} visual disruption rows in {input_path}")


if __name__ == "__main__":
    main()
