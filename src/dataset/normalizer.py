from typing import List


def normalize_steps(steps: List[str]) -> List[str]:
    out = []
    for s in steps:
        t = ' '.join(str(s).strip().split())
        if t:
            out.append(t)
    return out
