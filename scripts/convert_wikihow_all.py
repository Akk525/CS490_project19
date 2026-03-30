import json
from pathlib import Path

INPUT_DIR = Path("data/raw/wikihow")
OUTPUT_FILE = INPUT_DIR / "procedures.jsonl"


def extract_steps(item: dict) -> list[str]:
    steps = []
    raw_steps = item.get("Steps", [])

    if not isinstance(raw_steps, list):
        return steps

    for step in raw_steps:
        if not isinstance(step, dict):
            continue

        headline = (step.get("Headline") or "").strip()
        description = (step.get("Description") or "").strip()

        if headline and description:
            steps.append(f"{headline} {description}")
        elif headline:
            steps.append(headline)
        elif description:
            steps.append(description)

    return steps


def process_file(file_path: Path, out) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} does not contain a top-level list")

    count = 0

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        goal = (item.get("MainTask") or "").strip()
        summary = (item.get("MainTaskSummary") or "").strip()
        steps = extract_steps(item)

        if goal and len(steps) >= 3:
            record = {
                "source_id": f"{file_path.stem}_{idx}",
                "goal": goal,
                "summary": summary,
                "steps": steps,
                "url": item.get("URL"),
                "categories": item.get("Categories", []),
                "ingredients": item.get("Ingredients", []),
                "requirements": item.get("Requirements", []),
                "tips": item.get("Tips", []),
                "source_file": file_path.name,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    files = sorted(INPUT_DIR.glob("wikiHow*.json"))

    if not files:
        raise FileNotFoundError(f"No wikiHow*.json files found in {INPUT_DIR}")

    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for file in files:
            print(f"Processing {file.name}...")
            count = process_file(file, out)
            print(f"  -> {count} valid procedures")
            total += count

    print(f"\nSaved {total} procedures to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()