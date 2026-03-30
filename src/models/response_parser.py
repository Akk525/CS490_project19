import json
from typing import Any, Dict


def parse_model_output(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith('{') and raw.endswith('}'):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {'final_plan': raw}
