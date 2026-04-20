import csv
from pathlib import Path
from typing import Dict, List

from src.dataset.normalizer import normalize_steps
from src.dataset.visual_context import index_frame_paths, index_step_frame_paths
from src.utils.io import read_json


def load_foodtype_map(csv_path: Path) -> Dict[str, str]:
    """Map YouCook2 recipe_type id (string) to human-readable food label."""
    mapping: Dict[str, str] = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                recipe_id = row[0].strip()
                label = row[1].strip()
                if recipe_id:
                    mapping[recipe_id] = label
    return mapping


def _extract_steps(annotations: Dict) -> List[str]:
    steps: List[str] = []
    if 'annotations' in annotations and isinstance(annotations['annotations'], list):
        for seg in annotations['annotations']:
            sentence = seg.get('sentence') or seg.get('text') or ''
            if sentence:
                steps.append(sentence)
    elif 'segments' in annotations and isinstance(annotations['segments'], list):
        for seg in annotations['segments']:
            sentence = seg.get('sentence') or seg.get('description') or ''
            if sentence:
                steps.append(sentence)
    return normalize_steps(steps)


def _extract_step_times(annotations: Dict) -> List[Dict[str, float | None]]:
    segments = annotations.get('annotations') or annotations.get('segments') or []
    out: List[Dict[str, float | None]] = []
    if not isinstance(segments, list):
        return out

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start, end = _segment_bounds(seg)
        out.append({'start': start, 'end': end})
    return out


def _segment_bounds(seg: Dict) -> tuple[float | None, float | None]:
    if isinstance(seg.get('segment'), list) and len(seg['segment']) >= 2:
        return _to_float(seg['segment'][0]), _to_float(seg['segment'][1])
    if isinstance(seg.get('timestamp'), list) and len(seg['timestamp']) >= 2:
        return _to_float(seg['timestamp'][0]), _to_float(seg['timestamp'][1])
    start = seg.get('start') or seg.get('start_time') or seg.get('begin')
    end = seg.get('end') or seg.get('end_time') or seg.get('finish')
    return _to_float(start), _to_float(end)


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _discover_annotation_file(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / 'annotations' / 'youcookii_annotations_trainval.json',
        raw_dir / 'annotations' / 'youcook2_annotations_trainval.json',
        raw_dir / 'annotations' / 'youcookii_annotations_train.json',
        raw_dir / 'youcookii_annotations_trainval.json',
        raw_dir / 'youcook2_annotations_trainval.json',
    ]
    for path in candidates:
        if path.exists():
            return path

    # Fallback: any annotation-like JSON inside expected locations.
    search_roots = [raw_dir / 'annotations', raw_dir]
    discovered: List[Path] = []
    for sr in search_roots:
        if sr.exists():
            discovered.extend(sorted(sr.glob('*annotation*.json')))
            discovered.extend(sorted(sr.glob('*annotations*.json')))

    if discovered:
        return discovered[0]

    found_json: List[str] = []
    for sr in search_roots:
        if sr.exists():
            found_json.extend([str(p.relative_to(raw_dir)) for p in sorted(sr.glob('*.json'))])
    raise FileNotFoundError(
        'Missing YouCook2 annotation file. Expected one of: '
        'annotations/youcookii_annotations_trainval.json, '
        'annotations/youcook2_annotations_trainval.json, '
        'youcookii_annotations_trainval.json. '
        f'Looked under: {raw_dir}. '
        f'Found JSON files: {found_json if found_json else "none"}'
    )


def load_youcook2(raw_dir: Path, min_steps: int = 3) -> List[Dict]:
    ann_path = _discover_annotation_file(raw_dir)
    foodtype_path = raw_dir / 'annotations' / 'label_foodtype.csv'
    foodtype_map = load_foodtype_map(foodtype_path)
    project_root = raw_dir.parents[2] if len(raw_dir.parents) >= 3 else raw_dir
    frame_index = index_frame_paths(raw_dir / 'frames', project_root)
    step_frame_index = index_step_frame_paths(raw_dir / 'frames', project_root)

    payload = read_json(ann_path)
    database = payload.get('database') if isinstance(payload, dict) else None
    if not isinstance(database, dict):
        raise ValueError('Unexpected YouCook2 format: expected key "database" in annotations JSON.')

    rows: List[Dict] = []
    for video_id, item in database.items():
        steps = _extract_steps(item)
        if len(steps) < min_steps:
            continue
        step_times = _extract_step_times(item)

        recipe_type = str(item.get('recipe_type', '')).strip()
        goal = foodtype_map.get(recipe_type)
        if not goal:
            if steps:
                first = steps[0].strip()
                goal = f'Prepare dish: {first}'
            else:
                goal = 'Prepare dish'
        goal = goal.strip().lower()

        rows.append(
            {
                'source_dataset': 'youcook2',
                'source_item_id': video_id,
                'domain': 'cooking',
                'image_path': frame_index.get(video_id),
                'goal': goal,
                'steps': steps,
                'metadata': {
                    'subset': item.get('subset'),
                    'duration': item.get('duration'),
                    'step_times': step_times,
                    'step_image_paths': step_frame_index.get(video_id, {}),
                },
            }
        )
    return rows
